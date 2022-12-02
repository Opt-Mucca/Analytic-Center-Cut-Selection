#! /usr/bin/env python
import os
from pyscipopt import Model, SCIP_PROPTIMING, SCIP_PRESOLTIMING
import argparse
import yaml
from CutSelectors.CutoffDistanceCutsel import CutoffDistanceCutsel
from ConstraintHandlers.RepeatSepaConshdlr import RepeatSepaConshdlr
from utilities import is_dir, is_file, build_scip_model, get_filename, str_to_bool
import parameters


def run_instance(results_dir, instance_path, instance, rand_seed, permutation_seed, time_limit, root, print_stats,
                 solution_path=None, dir_cut_off=1, efficacy=0, int_support=0, obj_parallelism=0, cutoff='e',
                 max_density=1, fixed_cuts=False):
    """
    The call to solve a single instance. A model will be created and an instance file (and potentially solution file)
    loaded in. Appropriate settings as defined by this function call are then set and the model solved.
    After the model is solved, found infeasible, or some time limit hit, information is extracted and put into
    a yml file. All calls to solve an instance should go through this function and 'run_python_slurm_job' in
    utilities.py.
    Args:
        results_dir: The directory in which all all result files will be stored
        instance_path: The path to the MIP instance
        instance: The instance base name of the MIP file
        rand_seed: The random seed which will be used to shift all SCIP randomisation
        permutation_seed: The random seed which will be used to permute the problem before solving
        time_limit: The time limit, if it exists for our SCIP instance (in seconds).
        root: A boolean for whether we should restrict our solve to the root node or not
        print_stats: Whether the .stats file from the run should be printed or not
        solution_path: The path to the solution file which will be loaded
        dir_cut_off: Directed cut off distance weight
        efficacy: Efficacy weight
        int_support: Integer support weight
        obj_parallelism: Objective parallelism weight
        cutoff: The type of distance measure used. Please see parameters.py for a complete list
        max_density: The maximum density of a cut allowed into the LP
        fixed_cuts: If parameters.NUM_CUT_ROUNDS should be forced. Includes custom constraint handler and cut selector

    Returns:
        Nothing. All results from this run should be output to a file in results_dir.
        The results should contain all information about the run, (e.g. solve_time, dual_bound etc)
    """

    # Make sure the input is of the right type
    assert type(time_limit) == int and time_limit > 0
    assert is_dir(results_dir)
    assert is_file(instance_path)
    assert instance == os.path.split(instance_path)[-1].split('.mps')[0]
    assert type(rand_seed) == int and rand_seed >= 0
    assert isinstance(print_stats, bool)
    if solution_path is not None:
        assert is_file(solution_path) and instance == os.path.split(solution_path)[-1].split('.sol')[0]
    assert cutoff in parameters.CUTOFF_OPTIONS

    # Set the time limit if None is provided. Set the node_limit to 1 and dummy_branch to True if root is True.
    time_limit = None if time_limit < 0 else time_limit
    node_lim = 1 if root else -1
    dummy_branch = True if root else False

    # Build the actual SCIP model from the information now
    scip, cut_selector = build_scip_model(instance_path, node_lim, rand_seed, True, True, True, True,
                                          dummy_branch, permutation_seed, time_limit=time_limit, sol_path=solution_path,
                                          dir_cut_off=dir_cut_off, efficacy=efficacy, int_support=int_support,
                                          obj_parallelism=obj_parallelism, cutoff=cutoff,
                                          max_density=max_density, fixed_cuts=fixed_cuts)

    # Solve the SCIP model and extract all solve information
    solve_model_and_extract_solve_info(scip, dir_cut_off, efficacy, int_support, obj_parallelism, rand_seed,
                                       permutation_seed, instance, results_dir, root=root, cutoff=cutoff,
                                       print_stats=print_stats, cut_selector=cut_selector)

    # Free the SCIP instance
    scip.freeProb()

    return


def solve_model_and_extract_solve_info(scip, dir_cut_off, efficacy, int_support, obj_parallelism, rand_seed,
                                       permutation_seed, instance, results_dir, root=True, cutoff=None,
                                       print_stats=False, cut_selector=None):
    """
    Solves the given SCIP model and after solving creates a YML file with all potentially interesting
    solve information. This information will later be read and used to update the neural_network parameters
    Args:
        scip: The PySCIPOpt model that we want to solve
        dir_cut_off: The coefficient for the directed cut-off distance
        efficacy: The coefficient for the efficacy
        int_support: The coefficient for the integer support
        obj_parallelism: The coefficient for the objective function parallelism (see also the cosine similarity)
        rand_seed: The random seed used in the scip parameter settings
        permutation_seed: The random seed used to permute the problems rows and columns before solving
        instance: The instance base name of our problem
        results_dir: The directory in which all all result files will be stored
        root: A kwarg that informs if the solve is restricted to the root node. Used for naming the yml file
        cutoff: The type of distance measure used. Please see parameters.py for a complete list
        print_stats: A kwarg that informs if the .stats file from the run should be saved
        cut_selector: The cut selector object attached to the scip Model

    Returns:

    """

    # Solve the MIP instance. All parameters should be pre-set
    scip.optimize()

    # Check if measure all methods was used by the cut selector. If so, then print all the cut statistics
    if cut_selector is not None:
        if cutoff == 'compare_all_methods':
            yml_file = get_filename(results_dir, instance, rand_seed=rand_seed, permutation_seed=permutation_seed,
                                    root=root, cutoff=cutoff, ext='yml')
            cut_selector.print_cut_scores(yml_file)
            return

    # Initialise the dictionary that will store our solve information
    data = {}

    # Get the solve_time
    data['solve_time'] = scip.getSolvingTime()
    # Get the number of cuts applied
    data['num_cuts'] = scip.getNCutsApplied()
    # Get the number of nodes in our branch and bound tree
    data['num_nodes'] = scip.getNNodes()
    # Get the best primal solution if available
    data['primal_bound'] = scip.getObjVal() if len(scip.getSols()) > 0 else 1e+20
    # Get the gap provided a primal solution exists
    data['gap'] = scip.getGap() if len(scip.getSols()) > 0 else 1e+20
    # Get the best dual bound
    data['dual_bound'] = scip.getDualbound()
    # Get the number of LP iterations
    data['num_lp_iterations'] = scip.getNLPIterations()
    # Get the status of the solve
    data['status'] = scip.getStatus()
    # Get the primal-dual difference
    data['primal_dual_difference'] = data['primal_bound'] - data['dual_bound'] if len(scip.getSols()) > 0 else 1e+20
    # Get the number of separation rounds
    data['num_sepa_rounds'] = scip.getNSepaRounds()

    # Get the percentage of integer variables with fractional values. This includes implicit integer variables
    scip_vars = scip.getVars()
    non_cont_vars = [var for var in scip_vars if var.vtype() != 'CONTINUOUS']
    assert len(non_cont_vars) > 0
    if root:
        cont_valued_non_cont_vars = [var for var in non_cont_vars if not scip.isZero(scip.frac(var.getLPSol()))]
    else:
        if len(scip.getSols()) > 0:
            scip_sol = scip.getBestSol()
            cont_valued_non_cont_vars = [var for var in non_cont_vars if not scip.isZero(scip.frac(scip_sol[var]))]
        else:
            cont_valued_non_cont_vars = [var for var in non_cont_vars if not scip.isZero(scip.frac(var.getLPSol()))]
    data['solution_fractionality'] = len(cont_valued_non_cont_vars) / len(non_cont_vars)

    # Add the cut-selector parameters
    data['dir_cut_off'] = dir_cut_off
    data['efficacy'] = efficacy
    data['int_support'] = int_support
    data['obj_parallelism'] = obj_parallelism

    # Get the primal dual integral. This is not really needed for root solves, but might be important to have
    # It is only accessible through the solver statistics. TODO: Write a wrapper function for this
    stat_file = get_filename(results_dir, instance, rand_seed, root=root, permutation_seed=permutation_seed,
                             cutoff=cutoff, ext='stats')
    assert not os.path.isfile(stat_file)
    scip.writeStatistics(filename=stat_file)
    with open(stat_file) as s:
        stats = s.readlines()
    # TODO: Make this safer to access.
    for line_i, line in enumerate(stats):
        if 'primal-dual' in line:
            data['primal_dual_integral'] = float(line.split(':')[1].split('     ')[1])
        # Get the initial LP value. Calculate the dual-bound closed. (Same as the primal-dual difference when optimal)
        if 'First LP value' in line:
            data['initial_dual_bound'] = float(line.split(':')[-1].split('\n')[0])
            data['dual_bound_difference'] = data['dual_bound'] - data['initial_dual_bound']
    # If we haven't asked to save the file, then remove it.
    if not print_stats:
        os.remove(stat_file)

    # Extract data from the cut selector if it was also included
    if cut_selector is not None:
        cut_statistics = cut_selector.get_statistics()
        for feature in cut_statistics:
            data[feature] = cut_statistics[feature]

    # Dump the yml file containing all of our solve info into the right place
    yml_file = get_filename(results_dir, instance, rand_seed=rand_seed, permutation_seed=permutation_seed,
                            root=root, cutoff=cutoff, ext='yml')
    with open(yml_file, 'w') as s:
        yaml.dump(data, s)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=is_dir)
    parser.add_argument('instance_path', type=is_file)
    parser.add_argument('instance', type=str)
    parser.add_argument('rand_seed', type=int)
    parser.add_argument('permutation_seed', type=int)
    parser.add_argument('time_limit', type=int)
    parser.add_argument('root', type=str_to_bool)
    parser.add_argument('print_stats', type=str_to_bool)
    parser.add_argument('solution_path', type=str)
    parser.add_argument('dir_cut_off', type=float)
    parser.add_argument('efficacy', type=float)
    parser.add_argument('int_support', type=float)
    parser.add_argument('obj_parallelism', type=float)
    parser.add_argument('cutoff', type=str)
    parser.add_argument('max_density', type=float)
    parser.add_argument('fixed_cuts', type=str_to_bool)
    args = parser.parse_args()

    # Check if the solution file exists
    if args.solution_path == 'None':
        args.solution_path = None
    else:
        assert os.path.isfile(args.solution_path)

    # The main function call to run a SCIP instance with cut-sel params
    run_instance(args.results_dir, args.instance_path, args.instance, args.rand_seed, args.permutation_seed,
                 args.time_limit, args.root, args.print_stats, args.solution_path, args.dir_cut_off, args.efficacy,
                 args.int_support, args.obj_parallelism, args.cutoff, args.max_density, args.fixed_cuts)
