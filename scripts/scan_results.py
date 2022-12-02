import os
import argparse
import yaml
import numpy as np
import logging
import pdb

from utilities import is_dir, str_to_bool, get_filename, get_instances, get_random_seeds, get_permutation_seeds
import parameters


def scan_results(instance_dir, results_dir, root, remove_small_primal_dual_diff):
    """
    Function for scanning over the individual yml result files for each instance, seed, and cutoff combination.
    It then filters out instances that failed for any combination and concatenates all data into a dictionary
    and prints that into a yaml file.
    Args:
        instance_dir (dir): Directory containing all instance files
        results_dir (dir): Directory containing all yml result files from runs
        root (bool): Whether the runs that are being scanned over were restricted to the root node (naming convention)
        remove_small_primal_dual_diff (bool): Whether instances with small primal-dual should be removed

    Returns:
        Prints a yaml file containing all the information that is distributed over individual result files
    """
    assert os.path.isdir(results_dir)
    assert type(root) == bool

    # Get all the cutoff options (excluding the one that compares all methods as this is just used for statistics later)
    cutoff_options = parameters.DISTANCE_CUTOFF_OPTIONS

    # Initialise data structures that will contain instances and random seeds
    permutation_seeds = get_permutation_seeds(results_dir, cutoff_options)
    rand_seeds = get_random_seeds(results_dir, cutoff_options, permutation_seeds)
    instances = get_instances(instance_dir, files_are_gzipped=True)

    # Remove any runs from consideration that failed. Categorise the reasons that they failed
    invalid_primal_instances = set()
    invalid_mem_instances = set()
    invalid_time_instances = set()
    lp_error_instances = set()
    for instance in instances:
        for permutation_seed in permutation_seeds:
            for rand_seed in rand_seeds:
                for cutoff in cutoff_options:
                    permutation_seed_dir = os.path.join(os.path.join(results_dir, cutoff), str(permutation_seed))
                    yml_file = get_filename(permutation_seed_dir, instance, rand_seed=rand_seed,
                                            permutation_seed=permutation_seed, root=root, cutoff=cutoff, ext='yml')
                    log_file = get_filename(permutation_seed_dir, instance, rand_seed=rand_seed,
                                            permutation_seed=permutation_seed, root=root, cutoff=cutoff, ext='log')
                    assert os.path.isfile(log_file), 'Instance {} P-seed {} R-seed {} Cutoff {} no log {} in {}'.format(
                        instance, permutation_seed, rand_seed, cutoff, log_file, permutation_seed_dir)
                    if not os.path.isfile(yml_file):
                        instance_failed = True
                    else:
                        instance_failed = False
                    invalid_primal_instances, invalid_mem_instances, invalid_time_instances, lp_error_instances = \
                        check_if_invalid_reason_from_log_file(instance, log_file, invalid_primal_instances,
                                                              invalid_mem_instances, invalid_time_instances,
                                                              lp_error_instances, root=root,
                                                              know_instance_failed=instance_failed)

    # Remove all invalid instances from contention
    invalid_instances = invalid_primal_instances | invalid_mem_instances | invalid_time_instances | lp_error_instances
    instances = sorted(list(set(instances) - invalid_instances))

    # Now check if any instance failed for specific conditions (root optimal, number of cuts etc)
    root_optimal_instances = set()
    invalid_cut_instances = set()
    root_timeout_instances = set()
    full_timeout_instances = set()
    small_primal_dual_difference_instances = set()
    for instance in instances:
        root_optimal = False
        valid_num_cuts = False
        root_timeout = False
        full_timeout = True if not root else False
        small_primal_dual = False
        for permutation_seed in permutation_seeds:
            for rand_seed in rand_seeds:
                for cutoff in cutoff_options:
                    permutation_seed_dir = os.path.join(os.path.join(results_dir, cutoff),
                                                        str(permutation_seed))
                    yml_file = get_filename(permutation_seed_dir, instance, rand_seed=rand_seed,
                                            permutation_seed=permutation_seed, root=root, cutoff=cutoff,
                                            ext='yml')
                    assert os.path.isfile(yml_file), 'File {} doesnt exist'.format(yml_file)
                    with open(yml_file, 'r') as s:
                        run_data = yaml.safe_load(s)
                    # Flag the instance if it is root optimal
                    if (root and run_data['status'] == 'optimal') or \
                            (run_data['num_nodes'] <= 1 and run_data['status'] == 'optimal'):
                        root_optimal = True
                    # Flag the instance is no cuts are ever added
                    if 'cuts_added_per_round' in run_data and run_data['cuts_added_per_round'] > 0:
                        valid_num_cuts = True
                    # If we want to remove instances with stationary dual bounds. Filter by some tolerance
                    if remove_small_primal_dual_diff and root and \
                            run_data['primal_dual_difference'] < parameters.MIN_PRIMAL_DUAL_DIFFERENCE:
                        small_primal_dual = True
                    # Flag instance if hit a root timelimit
                    if root and run_data['status'] == 'timelimit':
                        root_timeout = True
                    # If the return code is not expected then flag the instance and quit the script
                    if run_data['status'] not in ['timelimit', 'nodelimit', 'optimal']:
                        print('Instance {} p-seed {} r-seed {} failed for reason {}'.format(
                            instance, permutation_seed, rand_seed, run_data['status']), flush=True)
                        quit()
                    # Flag that the instance doesn't always time-out. Don't use this to blankelty filter.
                    if not root and run_data['status'] == 'optimal':
                        full_timeout = False
        if root_optimal:
            root_optimal_instances.add(instance)
        if not valid_num_cuts:
            invalid_cut_instances.add(instance)
        if root_timeout:
            root_timeout_instances.add(instance)
        if small_primal_dual:
            small_primal_dual_difference_instances.add(instance)
        if full_timeout:
            full_timeout_instances.add(instance)
    # Print instances that timed out for all experiments. These may be interesting and shouldn't blanketly be removed.
    print(full_timeout_instances)

    # Now remove the root optimal instances (empty if not focused on root node solves)
    instances = sorted(list(set(instances) - (root_optimal_instances | invalid_cut_instances | root_timeout_instances
                                              | small_primal_dual_difference_instances)))

    # Print out reasons for why some instances failed
    print_invalid_instance_reasons(invalid_primal_instances, invalid_mem_instances, invalid_time_instances,
                                   lp_error_instances, root_optimal_instances, invalid_cut_instances,
                                   root_timeout_instances, small_primal_dual_difference_instances)

    # Now create a single data structure to store all run data
    full_data = {instance: {permutation_seed: {rand_seed: {} for rand_seed in rand_seeds}
                            for permutation_seed in permutation_seeds} for instance in instances}

    # Populate the complete data structure containing data on all valid runs
    for instance in instances:
        for permutation_seed in permutation_seeds:
            for rand_seed in rand_seeds:
                for cutoff in cutoff_options:
                    permutation_seed_dir = os.path.join(os.path.join(results_dir, cutoff), str(permutation_seed))
                    yml_file = get_filename(permutation_seed_dir, instance, rand_seed=rand_seed,
                                            permutation_seed=permutation_seed, root=root, cutoff=cutoff, ext='yml')
                    with open(yml_file, 'r') as s:
                        run_data = yaml.safe_load(s)
                    full_data[instance][permutation_seed][rand_seed][cutoff] = run_data

    # Dump the data from all instances into a single yml file
    result_file = get_filename(results_dir, 'full_results', rand_seed=None, root=False, cutoff=None, ext='yml')
    with open(result_file, 'w') as s:
        yaml.dump(full_data, s)


def check_if_invalid_reason_from_log_file(instance, logfile, invalid_primal_instances, invalid_mem_instances,
                                          invalid_time_instances, lp_error_instances,
                                          root=True, ignore_time_limit=False, know_instance_failed=False):
    """
    Gets the reason that a run was invalid. A run could be invalid for the following reasons:
    1. Solution not accepted
    2. Slurm memory limit has been hit
    3. Time limit of the run was hit
    4. LP Error

    Args:
        instance (set): The instance name
        logfile (is_file): The .out file produced by the run (transformed into a .log file for analysis)
        invalid_primal_instances (set): Set containing instances that did not accept pre-loaded solutions
        invalid_mem_instances (set): Set containing instances that exceeded memory limits
        invalid_time_instances (set): Instances that hit their time limits
        lp_error_instances (set): Instances that failed for other reasons
        root (bool): Whether the run only was concerned with the root node
        ignore_time_limit (bool): Whether the time limit should be ignored as a failure reason
        know_instance_failed (bool): Whether we know if the instance failed for sure or not

    Returns:
        Updated appropriate lists with the invalid instances of the correct type
    """
    assert os.path.isfile(logfile) and logfile.endswith('.log'), print('{} is not log file'.format(logfile))

    with open(logfile, 'r') as f:
        lines = f.readlines()

    found_reason = False
    for line in lines:
        if 'Invalid input line' in line and 'solution file' in line:
            invalid_primal_instances.add(instance)
            found_reason = True
            break
        elif 'all 1 solutions given by solution candidate storage are infeasible' in line:
            invalid_primal_instances.add(instance)
            found_reason = True
            break
        elif 'error: Exceeded job memory limit' in line:
            invalid_mem_instances.add(instance)
            found_reason = True
            break
        elif 'error: Job' in line and 'exceeded memory limit' in line:
            invalid_mem_instances.add(instance)
            found_reason = True
            break
        elif 'CANCELLED' in line and 'DUE TO TIME LIMIT' in line:
            invalid_time_instances.add(instance)
            found_reason = True
            break
        elif not root and 'WARNING: LP solver reached time limit, but SCIP time limit is not' in line:
            invalid_time_instances.add(instance)
            found_reason = True
            break
        elif root and 'solving was interrupted [time limit reached]' in line and not ignore_time_limit:
            # TODO: Decide if this should be placed here.
            invalid_time_instances.add(instance)
            found_reason = True
            break

    if not found_reason and know_instance_failed:
        lp_error_instances.add(instance)

    return invalid_primal_instances, invalid_mem_instances, invalid_time_instances, lp_error_instances


def print_invalid_instance_reasons(invalid_primal_instances, invalid_mem_instances, invalid_time_instances,
                                   lp_error_instances, root_optimal_instances, invalid_cut_instances,
                                   root_timeout_instances, small_primal_dual_difference_instances):
    """
    Args:
        invalid_primal_instances (set): Set containing instances that did not accept pre-loaded solutions
        invalid_mem_instances (set): Set containing instances that exceeded memory limits
        invalid_time_instances (set): Instances that hit their time limits
        lp_error_instances (set): All instances with unresolved solve errors
        root_optimal_instances (set): All instances that were optimal at the root node
        invalid_cut_instances (set): All instances that used too few or too many cuts
        root_timeout_instances (set): All instances that timed out for root related solves
        small_primal_dual_difference_instances (set): All instances that have too small of a primal-dual difference

    Returns: Nothing, just prints information about the sets
    """
    assert type(invalid_primal_instances) == set
    assert type(invalid_mem_instances) == set
    assert type(invalid_time_instances) == set
    assert type(lp_error_instances) == set
    assert type(root_optimal_instances) == set
    assert type(invalid_cut_instances) == set
    assert type(root_timeout_instances) == set
    assert type(small_primal_dual_difference_instances) == set

    print('Invalid Solution: {} many instances. {}'.format(len(invalid_primal_instances), invalid_primal_instances),
          flush=True)
    print('Too much memory: {} many instances. {}'.format(len(invalid_mem_instances), invalid_mem_instances),
          flush=True)
    print('Too much time. {} many instances. {}'.format(len(invalid_time_instances), invalid_time_instances),
          flush=True)
    print('Unresolved Errors. {} many instances. {}'.format(len(lp_error_instances), lp_error_instances),
          flush=True)
    print('Root optimal. {} many instances. {}'.format(len(root_optimal_instances), root_optimal_instances),
          flush=True)
    print('Invalid number of cuts. {} many instances. {}'.format(len(invalid_cut_instances), invalid_cut_instances),
          flush=True)
    print('Root hit time limit. {} many instances. {}'.format(len(root_timeout_instances), root_timeout_instances),
          flush=True)
    print('Small primal dual difference. {} many instances. {}'.format(
        len(small_primal_dual_difference_instances), small_primal_dual_difference_instances), flush=True)

    overlap = [invalid_primal_instances, invalid_mem_instances, invalid_time_instances, lp_error_instances,
               root_optimal_instances, invalid_cut_instances, root_timeout_instances,
               small_primal_dual_difference_instances]

    def instance_set_name(idx):
        if idx == 0:
            return 'invalid_primal_instances'
        elif idx == 1:
            return 'invalid_mem_instances'
        elif idx == 2:
            return 'invalid_time_instances'
        elif idx == 3:
            return 'lp_error_instances'
        elif idx == 4:
            return 'root_optimal_instances'
        elif idx == 5:
            return 'invalid_cut_instances'
        elif idx == 6:
            return 'root_timeout_instances'
        elif idx == 7:
            return 'small_primal_dual_diff'
        else:
            logging.warning('index out of range [0,7]: {}'.format(idx))
            return ''

    for i in range(8):
        for j in range(i + 1, 8):
            intersection = overlap[i].intersection(overlap[j])
            if len(intersection) > 0:
                print('Overlap of {} and {} has {} many instances'.format(instance_set_name(i), instance_set_name(j),
                                                                          len(intersection)), flush=True)
                for instance in list(intersection):
                    print('Instance {} had multiple unique fail reasons over its seeds. Reasons: {}, {}'.format(
                        instance, instance_set_name(i), instance_set_name(j)), flush=True)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=is_dir)
    parser.add_argument('results_dir', type=is_dir)
    parser.add_argument('root', type=str_to_bool)
    parser.add_argument('remove_small_primal_dual_diff', type=str_to_bool)
    args = parser.parse_args()

    scan_results(args.instance_dir, args.results_dir, args.root, args.remove_small_primal_dual_diff)
