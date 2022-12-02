#! /usr/bin/env python
import os
import argparse
import time
from utilities import is_dir, run_python_slurm_job, remove_slurm_files


def generate_features(instance_dir, solution_dir, features_dir, outfile_dir, num_rand_seeds, num_permutation_seeds,
                      time_limit):
    """
    Driver function for iterating over all instances and calling the actual slurm job to extract feature instances
    Args:
        instance_dir (dir): The directory containing all instances
        solution_dir (dir): The directory containing all solutions
        features_dir (dir): The directory where we will store all feature information (feature YMLs)
        outfile_dir (dir): The directory where we will dump all slurm outfiles
        num_rand_seeds (int): The number of random seeds
        num_permutation_seeds (int): The number of permutation seeds the variables and constraints before solving
        time_limit (int): The time limit per job (in seconds)

    Returns:
        Nothing. It just issues all slurm jobs.
    """

    # Get all the instance files and the instance names
    instance_files = sorted(os.listdir(instance_dir))
    instances = [instance_path.split('.')[0] for instance_path in instance_files]

    # Iterate over the instances
    for i, instance in enumerate(instances):
        instance_path = os.path.join(instance_dir, instance_files[i])
        # TODO: Don't assume the solution is gzipped
        solution_path = os.path.join(solution_dir, instance + '.sol.gz')
        assert os.path.isfile(solution_path)
        for permutation_seed in range(0, num_permutation_seeds):
            for seed_i in range(1, num_rand_seeds + 1):
                _ = run_python_slurm_job(python_file='Slurm/extract_features.py',
                                         job_name='{}--{}'.format(instance, seed_i),
                                         outfile=os.path.join(outfile_dir, '%j__{}__{}.out'.format(
                                             instance, seed_i)),
                                         time_limit=3600,
                                         arg_list=[features_dir, instance_path, instance, seed_i, permutation_seed,
                                                   time_limit, solution_path],
                                         exclusive=False
                                         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=is_dir)
    parser.add_argument('solution_dir', type=is_dir)
    parser.add_argument('features_dir', type=is_dir)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('num_rand_seeds', type=int)
    parser.add_argument('num_permutation_seeds', type=int)
    parser.add_argument('time_limit', type=int)
    args = parser.parse_args()

    # Remove all slurm .out files produced by previous runs
    args.outfile_dir = os.path.join(args.outfile_dir, 'generate_features')
    if not os.path.isdir(args.outfile_dir):
        os.mkdir(args.outfile_dir)
    else:
        remove_slurm_files(args.outfile_dir)

    # Clear out all previous files in features_dir and outfile_dir
    print('The following directories will be emptied: {}, {}'.format(args.features_dir, args.outfile_dir))
    for directory in [args.features_dir, args.outfile_dir]:
        assert directory != '/' and directory != '~'
    time.sleep(5)
    for directory in [args.features_dir, args.outfile_dir]:
        for old_file in os.listdir(directory):
            os.remove(os.path.join(directory, old_file))

    # The main function call to run a SCIP instance that will extract root node features
    generate_features(args.instance_dir, args.solution_dir, args.features_dir, args.outfile_dir,
                      args.num_rand_seeds, args.num_permutation_seeds, args.time_limit)
