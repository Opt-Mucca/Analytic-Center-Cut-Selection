#! /usr/bin/env python
import os
import argparse
import time
from utilities import is_dir, run_python_slurm_job, str_to_bool, get_slurm_output_file, get_filename, remove_slurm_files
import parameters

"""
IDEA: write a wrapper function for writeMIP from scip_lp.c. You can then include a dummy branch rule after root node
experiments to spit out the MILP w/ the added cuts. This problem can then be reloaded and solved without separators
to get accurate running time comparisons. Would need to also disable presolve
"""


def run_instances(instance_dir, solution_dir, results_dir, outfile_dir, solutions_are_compressed, num_rand_seeds,
                  num_permutation_seeds, time_limit, root, fixed_cuts, use_provided_sols, print_stats,
                  single_cutoff_option=None):
    """
    The main function for iterating over all instance, random seed, permutation seed, and cutoff combinations.
    For each combination a slurm job is issued, which creates the appropriate solve files for each run.

    Args:
        instance_dir (dir): The directory containing all instances
        solution_dir (dir): The directory containing all solution files
        results_dir (dir): The directory where all result files will be dumped
        outfile_dir (dir): The directory where all out files will be dumped
        num_rand_seeds (int): The number of random seeds used in the experiments
        num_permutation_seeds (int): The number of permutation seeds used in the experiments
        solutions_are_compressed (bool): Whether the solutions are gzipped or not
        time_limit (int): The time limit on each individual run
        root (bool): Whether the runs should be restricted to the root node or not
        fixed_cuts (bool): Whether we should restrict ourselves to the fixed amount of cuts added
        use_provided_sols (bool): Whether pre-calculated solutions should be used when starting the run
        print_stats (bool): Whether a stats file should be output from each solve call
        single_cutoff_option (str): A potential single cutoff option to reproduce a single set of runs

    Returns:
        Nothing at all. This is the main function call and it will just produce all appropriate run files
    """

    # Get all instance files
    instance_files = sorted(os.listdir(instance_dir))
    instances = [instance_path.split('.mps')[0] for instance_path in instance_files]

    # Iterate through all different cutoff options
    for cutoff in parameters.CUTOFF_OPTIONS:

        # If a single cutoff option is provided then skip all other options
        if single_cutoff_option in parameters.CUTOFF_OPTIONS:
            if cutoff != single_cutoff_option:
                continue

        # Create cutoff specific outfile and results subdirectories
        cutoff_outfile_dir = os.path.join(outfile_dir, cutoff)
        cutoff_results_dir = os.path.join(results_dir, cutoff)
        for cutoff_dir in [cutoff_outfile_dir, cutoff_results_dir]:
            if os.path.isdir(cutoff_dir):
                remove_slurm_files(cutoff_dir)
            else:
                os.mkdir(cutoff_dir)

        # Adjust the run time limit
        run_time_limit = time_limit if cutoff != 'compare_all_methods' else len(parameters.CUTOFF_OPTIONS) * time_limit
        # Give the slurm time limit a time comfortably larger than the SCIP time limit
        slurm_time_limit = 2 * run_time_limit

        # Either set efficacy or directed cutoff distance parameters to 1.0
        efficacy = 1.0 if cutoff in parameters.EFFICACY_CUTOFF_OPTIONS else 0.0
        dir_cut_off = 1.0 - efficacy
        # Get the max density. This can be extracted from the cutoff identification string
        max_density = 0.05 if ('05' in cutoff) else 0.1 if ('10' in cutoff) else 0.2 if ('20' in cutoff) else 0.4 if \
            ('40' in cutoff) else 0.8 if ('80' in cutoff) else 1

        # Iterate over all permutation seeds.
        for permutation_seed in range(0, num_permutation_seeds):

            # Create outfile and result specific subdirectories for the permutation seed
            permutation_outfile_dir = os.path.join(cutoff_outfile_dir, str(permutation_seed))
            permutation_results_dir = os.path.join(cutoff_results_dir, str(permutation_seed))
            assert not os.path.isdir(permutation_outfile_dir)
            assert not os.path.isdir(permutation_results_dir)
            os.mkdir(permutation_outfile_dir)
            os.mkdir(permutation_results_dir)

            # Iterate over all random seeds
            for seed_i in range(1, num_rand_seeds + 1):

                # Say what runs are being started
                print('Starting runs for cutoff {} p-seed {} and r-seed {}!'.format(cutoff, permutation_seed,
                                                                                    seed_i), flush=True)

                # Initialise a list containing all slurm job ids (so we can wait on them)
                slurm_job_ids = []
                # Iterate over all instances
                for i, instance in enumerate(instances):
                    # Create the instance path from the file and directory combination
                    instance_path = os.path.join(instance_dir, instance_files[i])
                    # If there are provided solutions, then check if they exists them and add them
                    if use_provided_sols:
                        if solutions_are_compressed:
                            solution_path = os.path.join(solution_dir, instance + '.sol.gz')
                        else:
                            solution_path = os.path.join(solution_dir, instance + '.sol')
                        assert os.path.isfile(solution_path), 'Not found solution file is {}'.format(solution_path)
                    else:
                        solution_path = 'None'
                    # Call the solve_instance.py run and append the returned slurm job id
                    ji = run_python_slurm_job(python_file='Slurm/solve_instance.py',
                                              job_name='{}--{}--{}'.format(instance, cutoff, seed_i),
                                              outfile=os.path.join(permutation_outfile_dir, '%j__{}__{}__{}.out'.format(
                                                  instance, cutoff, seed_i)),
                                              time_limit=slurm_time_limit,
                                              arg_list=[permutation_results_dir, instance_path, instance, seed_i,
                                                        permutation_seed, run_time_limit, root, print_stats,
                                                        solution_path,
                                                        dir_cut_off, efficacy, 0.0, 0.0, cutoff, max_density,
                                                        fixed_cuts],
                                              exclusive=True
                                              )
                    slurm_job_ids.append(ji)
                # Now submit the checker job that has dependencies slurm_job_ids
                safety_file_root = os.path.join(permutation_outfile_dir, '{}__{}'.format(permutation_seed, seed_i))
                _ = run_python_slurm_job(python_file='Slurm/safety_check.py',
                                         job_name='cleaner',
                                         outfile=safety_file_root + '.out',
                                         time_limit=10,
                                         arg_list=[safety_file_root + '.txt'],
                                         dependencies=slurm_job_ids)
                # Put the program to sleep until all of slurm jobs are complete
                time.sleep(10)
                while not os.path.isfile(safety_file_root + '.txt'):
                    time.sleep(10)

                # Now move all the log files from the out files to the results directory
                for instance in instances:
                    out_file = get_slurm_output_file(permutation_outfile_dir, instance, cutoff, seed_i)
                    new_out_file = get_filename(permutation_results_dir, instance, rand_seed=seed_i,
                                                permutation_seed=permutation_seed, root=root,
                                                cutoff=cutoff, ext='log')
                    assert os.path.isfile(out_file) and not os.path.isfile(new_out_file)
                    os.rename(out_file, new_out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=is_dir)
    parser.add_argument('solution_dir', type=is_dir)
    parser.add_argument('results_dir', type=is_dir)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('solutions_are_compressed', type=str_to_bool)
    parser.add_argument('num_rand_seeds', type=int)
    parser.add_argument('num_permutation_seeds', type=int)
    parser.add_argument('time_limit', type=int)
    parser.add_argument('root', type=str_to_bool)
    parser.add_argument('fixed_cuts', type=str_to_bool)
    parser.add_argument('print_stats', type=str_to_bool)
    parser.add_argument('remove_previous_runs', type=str_to_bool)
    parser.add_argument('single_cutoff_option', type=str)
    args = parser.parse_args()

    # Single cutoff is an option if we just want to repeat one type of run (because of some bug or something)
    if args.single_cutoff_option in parameters.CUTOFF_OPTIONS:
        single_cutoff = True
        print('Doing Single set of runs for cutoff option {}'.format(args.single_cutoff_option), flush=True)
    else:
        assert args.single_cutoff_option == 'None', args.single_cutoff_option
        single_cutoff = False
        print('Doing runs for all cutoff options', flush=True)

    # The main function call to run a SCIP instance with cut-sel params
    # Set provide_solutions to False if you don't want to run experiments with a pre-loaded solution
    for provide_solutions in [True]:
        dir_substring = 'root_' if args.root else 'full_'
        dir_substring += 'with_solution_start' if provide_solutions else 'no_solution_start'
        global_outfile_dir = os.path.join(args.outfile_dir, dir_substring)
        global_results_dir = os.path.join(args.results_dir, dir_substring)
        # Remove all previous results and out files from previous runs. Otherwise create the appropriate directories
        for global_dir in [global_outfile_dir, global_results_dir]:
            if not single_cutoff and args.remove_previous_runs:
                if os.path.isdir(global_dir):
                    print('Would remove all directories!', flush=True)
                    remove_slurm_files(global_dir)
                else:
                    os.mkdir(global_dir)
            else:
                if not os.path.isdir(global_dir):
                    os.mkdir(global_dir)
        print('Starting runs where the solution is {} provided!'.format(provide_solutions), flush=True)
        run_instances(args.instance_dir, args.solution_dir, global_results_dir, global_outfile_dir,
                      args.solutions_are_compressed, args.num_rand_seeds,
                      args.num_permutation_seeds, args.time_limit, args.root, args.fixed_cuts, provide_solutions,
                      args.print_stats, args.single_cutoff_option)
