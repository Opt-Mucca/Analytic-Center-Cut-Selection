import argparse
import yaml
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, top_k_accuracy_score, accuracy_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from scipy.stats import gmean
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns
import pdb

from utilities import is_file, is_dir, get_instances, get_random_seeds, get_permutation_seeds, get_filename

def load_data(results_yml, data_key='num_nodes'):
    """
    Function for loading the conctanedated results and then determining the instances, rand seeds,
    permutation seeds, and cutoff_options.

    Args:
        results_yml (file): The concatenated results file
        data_key (str): The metric we are interested in.

    Returns:
        Loaded concatenated results. Instances. Permutation seeds. Random seeds. Cutoff options.
    """

    # Load all the results from results_yml
    with open(results_yml, 'r') as s:
        cutoff_data = yaml.safe_load(s)

    # Initialise data structures that will contain instances and random seeds
    instances = sorted(list(cutoff_data.keys()))
    permutation_seeds = sorted(list(cutoff_data[instances[0]].keys()))
    rand_seeds = sorted(list(cutoff_data[instances[0]][permutation_seeds[0]].keys()))
    cutoff_options = sorted(list(cutoff_data[instances[0]][permutation_seeds[0]][rand_seeds[0]].keys()))

    # Remove any instances when doing num_nodes comparison and a cutoff hit timelimit
    time_limit_instances = set()
    if data_key == 'num_nodes':
        for instance in instances:
            for permutation_seed in permutation_seeds:
                for rand_seed in rand_seeds:
                    for cutoff in cutoff_options:
                        if cutoff_data[instance][permutation_seed][rand_seed][cutoff]['status'] == 'timelimit':
                            time_limit_instances.add(instance)
                            break
        # print('Removing {} instances. Hit time limit: {}'.format(len(time_limit_instances), time_limit_instances))
        instances = sorted(list(set(instances) - time_limit_instances))

    return cutoff_data, instances, permutation_seeds, rand_seeds, cutoff_options


def load_data_and_create_regressor(results_yml, features_dir, data_key='num_nodes', model_type='svr'):
    """
    Function for loading the data and creating the base regression model
    Args:
        results_yml (file): File of conctaneated results
        features_dir (dir): Directory containing results from feature generation runs
        data_key (str): The key used for training. 'num_nodes' or 'solve_time'. TODO: Extend to primal-dual-difference
        model_type (str): The type of model we want to create. 'svr' or 'random_forest'

    Returns:
        The loaded data in an sklearn friendly format, and the created model
    """

    assert model_type in ['svr', 'random_forest'], model_type
    assert data_key in ['num_nodes', 'solve_time'], data_key

    # Load the data
    cutoff_data, instances, permutation_seeds, rand_seeds, cutoff_options = load_data(results_yml, data_key)

    # For now let each (instance, rand_seed) combination be a data point
    data_dict = {}
    data = []
    regression_labels = []
    data_key_val_labels = []
    # Iterate over all instance-seed pairs and append a new data point to appropriate data structure
    for instance in instances:
        data_dict[instance] = {}
        for permutation_seed in permutation_seeds:
            data_dict[instance][permutation_seed] = {}
            for rand_seed in rand_seeds:
                # Get the label-wise information (each point is best / observed)
                c = cutoff_data[instance][permutation_seed][rand_seed]
                # Get the best value for this instance-seed pair
                best_cutoff_val = None
                for cutoff in cutoff_options:
                    cutoff_val = c[cutoff][data_key]
                    if data_key == 'solve_time' and c[cutoff]['status'] != 'timelimit':
                        cutoff_val -= c[cutoff]['cut_selection_time']
                    if best_cutoff_val is None:
                        best_cutoff_val = cutoff_val
                    else:
                        best_cutoff_val = min(best_cutoff_val, cutoff_val)
                regression_label = []
                data_key_val_label = []
                for cutoff in cutoff_options:
                    cutoff_val = c[cutoff][data_key]
                    # TODO: Explore how much an effect cut-selection-time actually has the learning process
                    if data_key == 'solve_time' and c[cutoff]['status'] != 'timelimit':
                        cutoff_val -= c[cutoff]['cut_selection_time']
                    # We add a shift of 10 (for both num_nodes and solve_time). It flattens out extreme examples.
                    regression_label.append((best_cutoff_val + 10) / (cutoff_val + 10))
                    # Remove the cut-selection-time shift for the final evaluation data
                    if data_key == 'solve_time' and c[cutoff]['status'] != 'timelimit':
                        cutoff_val += c[cutoff]['cut_selection_time']
                    data_key_val_label.append(cutoff_val)
                # Filter out the instance if it is [1, 1, 1, 1, ...., 1]
                if min(regression_label) >= 1:
                    # print('Instance {} - {} - {} has always same {} - {}'.format(
                    # instance, permutation_seed, rand_seed, data_key, best_cutoff_val))
                    continue
                # Append the regression label to the larger list
                regression_labels.append(regression_label)
                data_key_val_labels.append(data_key_val_label)

                # Get the feature information. Place it into data
                feature_file = get_filename(features_dir, instance, rand_seed=rand_seed,
                                            permutation_seed=permutation_seed, root=False, cutoff=None, ext='yml')
                assert os.path.isfile(feature_file)
                with open(feature_file, 'r') as s:
                    feature_data = yaml.safe_load(s)
                data_dict[instance][permutation_seed][rand_seed] = feature_data
                data.append([])
                for feature_key in ['dual_degeneracy', 'primal_degeneracy', 'lp_solution_fractionality',
                                    'ratio_of_equality_constraints', 'density_of_constraint_matrix']:
                    assert feature_key in feature_data
                    data[-1].append(feature_data[feature_key])

    # Initialise the regression model
    if model_type == 'random_forest':
        reg = RandomForestRegressor()
    if model_type == 'svr':
        reg = MultiOutputRegressor(SVR(kernel="poly"))

    return data, regression_labels, data_key_val_labels, cutoff_options, reg


def train_model_print_results(data, regression_labels, data_key_val_labels, cutoff_options, reg, test_size=0.1,
                              print_stats=False):
    """
    Function for training the model and printing out key statistics
    Args:
        data (list): Our input data
        regression_labels (list): The regression labels
        data_key_val_labels (list): Non-normalised data
        cutoff_options (list): List of cutoff options
        reg (): The regression model
        test_size (float): How much of the test set should be used for for test data

    Returns:
        A trained model
    """

    # Initialise dictionaries for mapping from id to cutoff_option (ids are necessary for classification ranking)
    cutoff_mapping = {cutoff: i for i, cutoff in enumerate(cutoff_options)}
    cutoff_reverse_mapping = {i: cutoff for i, cutoff in enumerate(cutoff_options)}

    # Split the data sets into training and test
    data_train, data_test, regression_train, regression_test, data_key_train, data_key_test = train_test_split(
        data, regression_labels, data_key_val_labels, test_size=test_size)
    # Perform 5-fold cross validation over the training data set
    scores = cross_validate(reg, data_train, regression_train, return_train_score=True, return_estimator=True,
                            scoring='neg_mean_squared_error')
    # Get the best performaing of the 5-models w.r.t their training sets
    best_reg_id = np.argmax(scores['test_score'])
    best_reg = scores['estimator'][best_reg_id]
    # Evaluate the best model over the test set
    test_pred = best_reg.predict(data_test)
    mse = mean_squared_error(regression_test, test_pred)
    # TODO: Accuracy calculations ignore cases where there are multiple best choices
    accuracy = accuracy_score(np.argmax(regression_test, axis=1), np.argmax(test_pred, axis=1))
    top_k_accuracy = top_k_accuracy_score(np.argmax(regression_test, axis=1), test_pred)
    print('MSE: {}. Accuracy: {}. Top-k-Accuracy: {}'.format(mse, accuracy, top_k_accuracy))

    # Get statistics on how each method performs
    if print_stats:
        shift = 10
        for d, r, v, s in [(data_train, regression_train, data_key_train, 'train'),
                           (data_test, regression_test, data_key_test, 'test')]:
            # Get lists containing the virtual best values, and the cutoff specific values
            virtual_best_vals = []
            cutoff_data_key_dict = {cutoff: [] for cutoff in cutoff_options}
            cutoff_wins = {cutoff: 0 for cutoff in cutoff_options}
            for i in range(len(d)):
                virtual_best_vals.append(min(v[i]))
            for i, cutoff in enumerate(cutoff_options):
                for j, label in enumerate(r):
                    cutoff_data_key_dict[cutoff].append(v[j][i] + shift)
                    if max(label) == label[i]:
                        cutoff_wins[cutoff] += 1
                cutoff_wins[cutoff] /= len(r)
            print('{} wins: {}'.format(s, cutoff_wins))
            # Get geomtric-mean results
            shifted_virtual_best_g_mean = gmean(virtual_best_vals) - shift
            shifted_eff_g_mean = gmean(cutoff_data_key_dict['efficacy']) - shift
            for i, cutoff in enumerate(cutoff_options):
                print('{}: Method {} Geometric-Mean {}'.format(
                    s, cutoff, round((gmean(cutoff_data_key_dict[cutoff]) - shift) / shifted_eff_g_mean, 2)))
            # Get regressor specific results
            num_wins = 0
            cutoff_reg_dict = {cutoff: 0 for cutoff in cutoff_options}
            reg_best_vals = []
            for i, label in enumerate(r):
                best_reg_cutoff_i = np.argmax(best_reg.predict([d[i]])[0])
                if label[best_reg_cutoff_i] == max(label):
                    num_wins += 1
                cutoff_reg_dict[cutoff_reverse_mapping[best_reg_cutoff_i]] += 1
                reg_best_vals.append(v[i][best_reg_cutoff_i] + shift)
            print('{}: Reg Wins {}'.format(s, round(num_wins / len(r), 2)))
            best_cutoff_gmean = min([gmean(cutoff_data_key_dict[cutoff]) for cutoff in cutoff_options]) - shift
            print('{}: Reg Geometric-Mean {}'.format(s, round(
                (gmean(reg_best_vals) - shift) / best_cutoff_gmean, 2)))
            print('{}: Reg Predictions {}'.format(s, cutoff_reg_dict))

    return best_reg, data_train, data_test, regression_train, regression_test, data_key_train, data_key_test


def perform_pca(data):
    """
    Function for performaing PCA
    Args:
        data (list): List of original feature space data

    Returns:
        PCA
    """

    pca = PCA(n_components=2)

    return pca


def plot_pca_data(data_train, data_test, regression_labels_train, regression_labels_test, cutoff_options, reg, pca):
    """
    Function for performing PCA and plotting results
    Args:
        data (list): List of features
        regression_labels (list): List of labels
        cutoff_options (list): The list of cutoff options
        reg (): The trained regression model
        pca (): The PCA object

    Returns:
        PCA object. Plots data.
    """

    # Now do PCA
    t_d_train = pca.fit_transform(data_train)

    # Make a grid of points in [-1, 1]**2
    sqrt_num_grid_points = 500
    min_x = min(t_d_train[:, 0])
    max_x = max(t_d_train[:, 0])
    min_y = min(t_d_train[:, 1])
    max_y = max(t_d_train[:, 1])
    gap_size = 0.05
    grid_x = np.linspace(min_x - gap_size, max_x + gap_size, sqrt_num_grid_points - 1)
    grid_y = np.linspace(min_y - gap_size, max_y + gap_size, sqrt_num_grid_points - 1)
    grid_labels = {}
    for i, x_val in enumerate(grid_x):
        grid_labels[i] = {}
        for j, y_val in enumerate(grid_y):
            original_data = pca.inverse_transform([x_val, y_val])
            pred = reg.predict([original_data])
            best_cutoff_id = np.argmax(pred[0])
            grid_labels[i][j] = best_cutoff_id

    # Make a colormap for cutoff measures using seaborn colour paletee (as in other plots)
    cmap = ListedColormap(sns.color_palette('colorblind')[:8])
    fig, ax = plt.subplots()
    grid_labels_array = np.array(
        [np.array([grid_labels[i][j] for i in range(len(grid_x))]) for j in range(len(grid_y))])
    colourbar_input = ax.pcolormesh(grid_x, grid_y, grid_labels_array, cmap=cmap, rasterized=True, vmin=0, vmax=7)

    # Plot the test set as black dots. Their transparency is the relative performance of their prediction.
    t_d_test = pca.transform(data_test)
    for i in range(len(t_d_test)):
        point = [t_d_test[i, 0], t_d_test[i, 1]]
        original_data = pca.inverse_transform(point)
        pred = reg.predict([original_data])
        best_cutoff_id = np.argmax(pred[0])
        alpha = regression_labels_test[i][best_cutoff_id]
        ax.scatter(t_d_test[:, 0][i], t_d_test[:, 1][i], c='black', alpha=alpha)

    # Get strings for axis_labels
    axis_label_dict = {0: 'dual_degeneracy', 1: 'primal_degeneracy', 2: 'fractionality', 3: 'thinness',
                       4: 'density'}
    # x_label = ''
    # y_label = ''
    # for i in range(len(pca.components_[0])):
    #     x_multiplier = round(pca.components_[0][i], 3)
    #     y_multiplier = round(pca.components_[1][i], 3)
    #     if len(x_label) > 0 and x_multiplier > 0:
    #         x_label += ' + '
    #     else:
    #         x_label += '  '
    #     if len(y_label) > 0 and y_multiplier > 0:
    #         y_label += ' + '
    #     else:
    #         y_label += '  '
    #     x_label += '{} {}'.format(str(round(pca.components_[0][i], 3)), axis_label_dict[i])
    #     y_label += '{} {}'.format(str(round(pca.components_[1][i], 3)), axis_label_dict[i])

    x_label = 'Component 1'
    y_label = 'Component 2'
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)

    # Make the legend
    cutoff_mapping = {'efficacy': 'eff',
                      'directed_cutoff': 'dcd',
                      'analytic_efficacy': 'a-eff',
                      'analytic_directed_cutoff': 'a-dcd',
                      'approximate_analytic_directed_cutoff': 'app-a-dcd',
                      'average_efficacy': 'avgeff',
                      'minimum_efficacy': 'mineff',
                      'expected_improvement': 'exp-improv'}
    legend_elements = []
    for i, cutoff in enumerate(cutoff_options):
        legend_elements.append(Patch(facecolor=cmap.colors[i], label=cutoff_mapping[cutoff]))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.show()

    return pca


def see_model_transfer(data_train_nodes, data_test_nodes, regression_train_nodes, regression_test_nodes,
                       data_key_train_nodes, data_key_test_nodes, data_train_time, data_test_time,
                       regression_train_time, regression_test_time, data_key_train_time, data_key_test_time,
                       cutoff_options, reg_nodes, reg_time):
    """
    See how the trained model on one metric performs on the data set of another metric.
    """

    shift = 10
    for d, r, v, s, reg in [(data_test_nodes, regression_test_nodes, data_key_test_nodes, 'nodes-nodes', reg_nodes),
                            (data_test_time, regression_test_time, data_key_test_time, 'nodes-time', reg_nodes),
                            (data_test_nodes, regression_test_nodes, data_key_test_nodes, 'time-nodes', reg_time),
                            (data_test_time, regression_test_time, data_key_test_time, 'time-time', reg_time)]:
        # Get comparisons of how our model (trained on nodes or solve_time) performs on the test sets
        cutoff_data_key_dict = {cutoff: [] for cutoff in cutoff_options}
        for i, cutoff in enumerate(cutoff_options):
            for j, label in enumerate(r):
                cutoff_data_key_dict[cutoff].append(v[j][i] + shift)
        reg_best_vals = []
        for i, label in enumerate(r):
            best_reg_cutoff_i = np.argmax(reg.predict([d[i]])[0])
            reg_best_vals.append(v[i][best_reg_cutoff_i] + shift)
        best_cutoff_gmean = min([gmean(cutoff_data_key_dict[cutoff]) for cutoff in cutoff_options]) - shift
        print('{}: Reg Geometric-Mean {}'.format(s, round(
            (gmean(reg_best_vals) - shift) / best_cutoff_gmean, 2)))

def get_predictions_from_regressor(reg, results_yml, features_dir, data_key='num_nodes'):
    """
    Function for printing out predictions per (instance-seed) pair of the regression model.
    This lets us then compare the regression model to other individual measures.
    Args:
        reg (): The scikit-learn regression model
        results_yml (file): The conctenated results file
        features_dir (dir): The directory containing all feature files
        data_key (str): The data key we are interested in. E.g. num_nodes

    Returns:
        Nothing. It prints the predictions out to a file.
    """

    # Load the data
    cutoff_data, instances, permutation_seeds, rand_seeds, cutoff_options = load_data(results_yml, data_key)

    # Initialise the reverse cutoff mapping
    cutoff_reverse_mapping = {i: cutoff for i, cutoff in enumerate(cutoff_options)}

    # Initialise the dictionary of all results
    prediction_dict = {}
    # Iterate over all instance-seed pairs and get the prediction of the regression model of that instance-seed pair.
    for instance in instances:
        prediction_dict[instance] = {}
        for permutation_seed in permutation_seeds:
            prediction_dict[instance][permutation_seed] = {}
            for rand_seed in rand_seeds:
                prediction_dict[instance][permutation_seed][rand_seed] = {}

                # Get the feature information. Place it into data
                feature_file = get_filename(features_dir, instance, rand_seed=rand_seed,
                                            permutation_seed=permutation_seed, root=False, cutoff=None, ext='yml')
                assert os.path.isfile(feature_file)
                with open(feature_file, 'r') as s:
                    feature_data = yaml.safe_load(s)
                features = []
                for feature_key in ['dual_degeneracy', 'primal_degeneracy', 'lp_solution_fractionality',
                                    'ratio_of_equality_constraints', 'density_of_constraint_matrix']:
                    assert feature_key in feature_data
                    features.append(feature_data[feature_key])
                prediction = reg.predict([features])
                pred_cutoff = cutoff_reverse_mapping[np.argmax(prediction, axis=1)[0]]
                c = cutoff_data[instance][permutation_seed][rand_seed]
                prediction_dict[instance][permutation_seed][rand_seed][data_key] = c[pred_cutoff][data_key]

    with open(data_key + '_model_results.yml', 'w') as s:
        yaml.dump(prediction_dict, s)

def load_data_and_train_model(results_yml, features_dir, data_key='num_nodes'):
    """
    Main function for building a sklearn SupportVectorRegression / RandomForestRegressot.
    It loads data from individual feature ymls, and from the concatenated results yml.
    It then ranks all cutoff methods per instance, weighting them by performance.
    It finally creates a regression model that learns which cutoff distance measure to use for which features.
    Args:
        results_yml (file): Concatenated results file from scan_results.py
        features_dir (dir): Directory where all the feature files are stored

    Returns:
        Nothing. It just prints out the performance of what the learned regression model would do
    """

    # Set the random seed for reproducibility
    np.random.seed(42)

    model_type = 'svr'

    # Get information on num_nodes
    data_nodes, regression_labels_nodes, data_key_val_labels_nodes, cutoff_options, reg_nodes = \
        load_data_and_create_regressor(results_yml, features_dir, data_key='num_nodes', model_type=model_type)

    # Build a regression model based on num_nodes
    reg_nodes, data_train_nodes, data_test_nodes, regression_train_nodes, regression_test_nodes, \
    data_key_train_nodes, data_key_test_nodes = train_model_print_results(data_nodes, regression_labels_nodes,
                                                                          data_key_val_labels_nodes, cutoff_options,
                                                                          reg_nodes, test_size=0.1)

    # get_predictions_from_regressor(reg_nodes, results_yml, features_dir, 'num_nodes')
    # get_predictions_from_regressor(reg_nodes, results_yml, features_dir, 'solve_time')
    # quit()

    # Perform PCA on the feature space using training samples for the num_nodes model
    pca_nodes = perform_pca(data_nodes)

    # Get information of solve time
    data_time, regression_labels_time, data_key_val_labels_time, cutoff_options, reg_time = \
        load_data_and_create_regressor(results_yml, features_dir, data_key='solve_time', model_type=model_type)

    # Build a regression model based on solve time
    reg_time, data_train_time, data_test_time, regression_train_time, regression_test_time, \
    data_key_train_time, data_key_test_time = train_model_print_results(data_time, regression_labels_time,
                                                                        data_key_val_labels_time, cutoff_options,
                                                                        reg_time, test_size=0.1)

    # Perform PCA on the feature space using training samples for the solve_time model
    pca_time = perform_pca(data_time)

    # See how well the model transfers learning results from one metric to another
    see_model_transfer(data_train_nodes, data_test_nodes, regression_train_nodes, regression_test_nodes,
                       data_key_train_nodes, data_key_test_nodes, data_train_time, data_test_time,
                       regression_train_time, regression_test_time, data_key_train_time, data_key_test_time,
                       cutoff_options, reg_nodes, reg_time)

    # Plot the transformed space and visualise the test set in this space.
    pca_nodes = plot_pca_data(data_train_nodes, data_test_nodes, regression_train_nodes, regression_test_nodes,
                              cutoff_options, reg_nodes, pca_nodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_yml', type=is_file)
    parser.add_argument('features_dir', type=is_dir)
    args = parser.parse_args()

    load_data_and_train_model(args.results_yml, args.features_dir)
