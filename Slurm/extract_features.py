#! /usr/bin/env python
import os
import argparse
import yaml
from utilities import is_dir, is_file, build_scip_model, get_filename
from BranchRules.RootNodeFeatureExtractorBranchRule import RootNodeFeatureExtractor


def extract_features(features_dir, instance_path, instance, rand_seed, permutation_seed, time_limit,
                     solution_path=None):
    """
    The call to extract statistic information from the instance. A model will be created and a branch rule
    included that extracts information from the root node. This information is printed to a file and the solve is
    then terminated. All calls to solve an instance should go through this function and 'run_python_slurm_job' in
    utilities.py.
    Args:
        features_dir: The directory in which all result files will be stored (feature YMLs)
        instance_path: The path to the MIP instance
        instance: The instance base name of the MIP file
        rand_seed: The random seed which will be used to shift all SCIP randomisation
        permutation_seed: The random seed used to permute the variables and constraints before solving
        time_limit: The time limit, if it exists for our SCIP instance (in seconds).
        solution_path: The path to the solution file which will be loaded

    Returns:
        Nothing. All results from this run should be output to a file in features_dir.
        The results should contain all information about the presolved instance
    """

    assert is_dir(features_dir)
    assert is_file(instance_path)
    assert instance == os.path.split(instance_path)[-1].split('.')[0]
    assert type(rand_seed) == int and rand_seed >= 0
    assert type(time_limit) == int and time_limit > 0
    if solution_path is not None:
        assert is_file(solution_path) and instance == os.path.split(solution_path)[-1].split('.')[0]

    # Build the actual SCIP model from the information now
    scip, _ = build_scip_model(instance_path, 2, rand_seed, True, True, False, False, False,
                               permutation_seed=permutation_seed, time_limit=time_limit,
                               sol_path=solution_path, fixed_cuts=False)

    feature_extractor = RootNodeFeatureExtractor()
    scip.includeBranchrule(feature_extractor, "feature_extractor", "extract features of LP solution at root node",
                           priority=10000000, maxdepth=-1, maxbounddist=1)
    scip.optimize()
    feature_dict = feature_extractor.features

    if len(feature_dict) == 0:
        scip.freeProb()
        print('Root either did not complete within time limit or was root optimal')
        quit()

    feature_file = get_filename(features_dir, instance, rand_seed, permutation_seed=permutation_seed,
                                root=False, cutoff=None, ext='yml')
    with open(feature_file, 'w') as s:
        yaml.dump(feature_dict, s)

    scip.freeProb()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('features_dir', type=is_dir)
    parser.add_argument('instance_path', type=is_file)
    parser.add_argument('instance', type=str)
    parser.add_argument('rand_seed', type=int)
    parser.add_argument('permutation_seed', type=int)
    parser.add_argument('time_limit', type=int)
    parser.add_argument('solution_path', type=str)
    args = parser.parse_args()

    if args.solution_path == 'None':
        args.solution_path = None
    else:
        assert os.path.isfile(args.solution_path)

    # The main function call to run a SCIP instance with cut-sel params
    extract_features(args.features_dir, args.instance_path, args.instance, args.rand_seed, args.permutation_seed,
                     args.time_limit, args.solution_path)
