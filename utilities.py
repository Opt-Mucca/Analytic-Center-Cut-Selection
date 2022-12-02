#! /usr/bin/env python
import os
import sys
import numpy as np
import subprocess
import shutil
import logging
import argparse
from pyscipopt import Model, SCIP_RESULT, SCIP_PARAMSETTING, Branchrule, SCIP_PRESOLTIMING, SCIP_PROPTIMING
from ConstraintHandlers.RepeatSepaConshdlr import RepeatSepaConshdlr
from CutSelectors.CutoffDistanceCutsel import CutoffDistanceCutsel
import parameters


def build_scip_model(instance_path, node_lim, rand_seed, pre_solve, propagation, separators, heuristics,
                     dummy_branch_rule, permutation_seed=0, time_limit=None, sol_path=None, dir_cut_off=0.0,
                     efficacy=1.0, int_support=0.1, obj_parallelism=0.1, cutoff='efficacy',
                     max_density=1, fixed_cuts=False):
    """
    General function to construct a PySCIPOpt model.

    Args:
        instance_path: The path to the instance
        node_lim: The node limit
        rand_seed: The random seed for all SCIP plugins (and LP solver)
        pre_solve: Whether pre-solve should be enabled or disabled
        propagation: Whether propagators should be enabled or disabled
        separators: Whether separators should be enabled or disabled
        heuristics: Whether heuristics should be enabled or disabled
        dummy_branch_rule: This is to cover a 'feature' of SCIP where by default strong branching is done and this can
                           give information about nodes beneath the node limit. So we add a branch-rule that can't.
        permutation_seed: The random seed used to permute the rows and columns before solving
        time_limit: The time_limit of the model
        sol_path: An optional path to a valid solution file containing a primal solution to the instance
        dir_cut_off: The directed cut off weight that is applied to the custom cut-selector
        efficacy: The efficacy weight that is applied to the custom cut-selector
        int_support: The integer support weight that is applied to the custom cut-selector
        obj_parallelism: The objective parallelism weight that is applied to the custom cut-selector
        cutoff: The type of distance measure used. Please see parameters.py for a complete list
        max_density: The maximum density of a cut that is allowed into the LP
        fixed_cuts: If parameters.NUM_CUT_ROUNDS should be forced. Includes custom constraint handler and cut selector

    Returns:
        pyscipopt model

    """
    assert os.path.exists(instance_path)
    assert type(node_lim) == int and type(rand_seed) == int
    assert all([type(param) == bool for param in [pre_solve, propagation, separators, heuristics]])

    # Create the base PySCIPOpt model and start setting parameters
    scip = Model()
    scip.setParam('limits/nodes', node_lim)

    # We do not want the solve process to restart and potentially trigger another presolve or remove cuts.
    scip.setParam('estimation/restarts/restartlimit', 0)
    scip.setParam('estimation/restarts/restartpolicy', 'n')
    scip.setParam('presolving/maxrestarts', 0)

    if permutation_seed > 0:
        scip.setParam('randomization/permutevars', True)
        scip.setParam('randomization/permutationseed', rand_seed)
    scip.setParam('randomization/randomseedshift', rand_seed)
    if not pre_solve:
        scip.setPresolve(SCIP_PARAMSETTING.OFF)
    if not propagation:
        scip.disablePropagation()
    if not separators:
        scip.setSeparating(SCIP_PARAMSETTING.OFF)
    if not heuristics:
        scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    if dummy_branch_rule:
        scip.setParam("branching/leastinf/priority", 5000000)
    if time_limit is not None:
        scip.setParam('limits/time', time_limit)
    if not fixed_cuts:
        cut_selector = None
        scip = set_scip_cut_selector_params(scip, dir_cut_off, efficacy, int_support, obj_parallelism)
    else:
        num_rounds = parameters.NUM_CUT_ROUNDS
        cuts_per_round = parameters.NUM_CUTS_PER_ROUND
        # Create a dummy constraint handler that forces the num_rounds amount of separation rounds
        constraint_handler = RepeatSepaConshdlr(scip, num_rounds)
        scip.includeConshdlr(constraint_handler, "RepeatSepa", "Forces a certain number of separation rounds",
                             sepapriority=-1, enfopriority=1, chckpriority=-1, sepafreq=-1, propfreq=-1,
                             eagerfreq=-1, maxprerounds=-1, delaysepa=False, delayprop=False, needscons=False,
                             presoltiming=SCIP_PRESOLTIMING.FAST, proptiming=SCIP_PROPTIMING.AFTERLPNODE)
        # Add the cut selector to the model
        cut_selector = CutoffDistanceCutsel(dir_cutoff_dist_weight=dir_cut_off,
                                            efficacy_weight=efficacy, int_support_weight=int_support,
                                            obj_parallel_weight=obj_parallelism,
                                            analytic_dir_cutoff=(cutoff == 'analytic_directed_cutoff'),
                                            analytic_efficacy=(cutoff == 'analytic_efficacy'),
                                            approximate_analytic_dir_cutoff=(
                                                    cutoff == 'approximate_analytic_directed_cutoff'),
                                            min_efficacy=(cutoff == 'minimum_efficacy'),
                                            average_efficacy=(cutoff == 'average_efficacy'),
                                            average_multiple_primal_solutions=(
                                                        cutoff == 'average_primal_directed_cutoff'),
                                            efficacy_check_projection=(
                                                    cutoff == 'efficacy_check_projection'),
                                            expected_improvement=(cutoff == 'expected_improvement'),
                                            compare_all_methods=(cutoff == 'compare_all_methods'),
                                            max_density=max_density)
        scip.includeCutsel(cut_selector, 'CutoffDistanceCutSel', 'Hybrid method with different cutoff measurements',
                           1000000)
        # Set the separator parameters
        scip = set_scip_separator_params(scip, num_rounds, -1, cuts_per_round, cuts_per_round, 0)

    # read in the problem
    scip.readProblem(instance_path)

    if sol_path is not None:
        assert os.path.isfile(sol_path) and '.sol' in sol_path
        # Create the solution to add to SCIP
        sol = scip.readSolFile(sol_path)
        # Add the solution. This automatically frees the loaded solution
        scip.addSol(sol)

    return scip, cut_selector


def set_scip_cut_selector_params(scip, dir_cut_off, efficacy, int_support, obj_parallelism):
    """
    Sets the SCIP hybrid cut-selector parameter values in the weighted sum
    Args:
        scip: The PySCIPOpt model
        dir_cut_off: The coefficient of the directed cut-off distance
        efficacy: The coefficient of the efficacy
        int_support: The coefficient of the integer support
        obj_parallelism: The coefficient of the objective value parallelism (cosine similarity)

    Returns:
        The PySCIPOpt model with set parameters
    """
    scip.setParam("cutselection/hybrid/dircutoffdistweight", max(dir_cut_off, 0))
    scip.setParam("cutselection/hybrid/efficacyweight", max(efficacy, 0))
    scip.setParam("cutselection/hybrid/intsupportweight", max(int_support, 0))
    scip.setParam("cutselection/hybrid/objparalweight", max(obj_parallelism, 0))

    return scip


def set_scip_separator_params(scip, max_rounds_root=-1, max_rounds=-1, max_cuts_root=10000, max_cuts=10000,
                              frequency=10):
    """
    Function for setting the separator params in SCIP.
    We enable more separators than default in an attempt to generate more cuts during solving.
    Certain sepeatrors are kept disabled due to their long running time.

    Args:
        scip: The SCIP Model object
        max_rounds_root: The max number of separation rounds that can be performed at the root node
        max_rounds: The max number of separation rounds that can be performed at any non-root node
        max_cuts_root: The max number of cuts that can be added per round in the root node
        max_cuts: The max number of cuts that can be added per node at any non-root node
        frequency: The separators will be called each time the tree hits a new multiple of this depth
    Returns:
        The SCIP Model with all the appropriate parameters now set
    """

    assert type(max_cuts) == int and type(max_rounds) == int
    assert type(max_cuts_root) == int and type(max_rounds_root) == int

    # First for the aggregation heuristic separator
    scip.setParam('separating/aggregation/freq', frequency)
    scip.setParam('separating/aggregation/maxrounds', -1)
    scip.setParam('separating/aggregation/maxroundsroot', -1)
    scip.setParam('separating/aggregation/maxsepacuts', 1000)
    scip.setParam('separating/aggregation/maxsepacutsroot', 1000)

    # Now the Chvatal-Gomory w/ MIP separator
    # scip.setParam('separating/cgmip/freq', frequency)
    # scip.setParam('separating/cgmip/maxrounds', max_rounds)
    # scip.setParam('separating/cgmip/maxroundsroot', max_rounds_root)

    # The clique separator
    scip.setParam('separating/clique/freq', frequency)
    scip.setParam('separating/clique/maxsepacuts', 1000)

    # The close-cuts separator
    # scip.setParam('separating/closecuts/freq', frequency)

    # The CMIR separator
    scip.setParam('separating/cmir/freq', frequency)

    # The Convex Projection separator
    # scip.setParam('separating/convexproj/freq', frequency)
    # scip.setParam('separating/convexproj/maxdepth', -1)

    # The disjunctive cut separator
    scip.setParam('separating/disjunctive/freq', frequency)
    scip.setParam('separating/disjunctive/maxrounds', -1)
    scip.setParam('separating/disjunctive/maxroundsroot', -1)
    scip.setParam('separating/disjunctive/maxinvcuts', 1000)
    scip.setParam('separating/disjunctive/maxinvcutsroot', 1000)
    scip.setParam('separating/disjunctive/maxdepth', -1)

    # The separator for edge-concave function
    # scip.setParam('separating/eccuts/freq', frequency)
    # scip.setParam('separating/eccuts/maxrounds', -1)
    # scip.setParam('separating/eccuts/maxroundsroot', -1)
    # scip.setParam('separating/eccuts/maxsepacuts', 1000)
    # scip.setParam('separating/eccuts/maxsepacutsroot', 1000)
    # scip.setParam('separating/eccuts/maxdepth', -1)

    # The flow cover cut separator
    scip.setParam('separating/flowcover/freq', frequency)

    # The gauge separator
    # scip.setParam('separating/gauge/freq', frequency)

    # Gomory MIR cuts
    scip.setParam('separating/gomory/freq', frequency)
    scip.setParam('separating/gomory/maxrounds', -1)
    scip.setParam('separating/gomory/maxroundsroot', -1)
    scip.setParam('separating/gomory/maxsepacuts', 1000)
    scip.setParam('separating/gomory/maxsepacutsroot', 1000)

    # The implied bounds separator
    scip.setParam('separating/impliedbounds/freq', frequency)

    # The integer objective value separator
    # scip.setParam('separating/intobj/freq', frequency)

    # The knapsack cover separator
    scip.setParam('separating/knapsackcover/freq', frequency)

    # The multi-commodity-flow network cut separator
    scip.setParam('separating/mcf/freq', frequency)
    scip.setParam('separating/mcf/maxsepacuts', 1000)
    scip.setParam('separating/mcf/maxsepacutsroot', 1000)

    # The odd cycle separator
    # scip.setParam('separating/oddcycle/freq', frequency)
    # scip.setParam('separating/oddcycle/maxrounds', -1)
    # scip.setParam('separating/oddcycle/maxroundsroot', -1)
    # scip.setParam('separating/oddcycle/maxsepacuts', 1000)
    # scip.setParam('separating/oddcycle/maxsepacutsroot', 1000)

    # The rapid learning separator
    scip.setParam('separating/rapidlearning/freq', frequency)

    # The strong CG separator
    scip.setParam('separating/strongcg/freq', frequency)

    # The zero-half separator
    scip.setParam('separating/zerohalf/freq', frequency)
    scip.setParam('separating/zerohalf/maxcutcands', 100000)
    scip.setParam('separating/zerohalf/maxrounds', -1)
    scip.setParam('separating/zerohalf/maxroundsroot', -1)
    scip.setParam('separating/zerohalf/maxsepacuts', 1000)
    scip.setParam('separating/zerohalf/maxsepacutsroot', 1000)

    # The rlt separator
    scip.setParam('separating/rlt/freq', frequency)
    scip.setParam('separating/rlt/maxncuts', 1000)
    scip.setParam('separating/rlt/maxrounds', -1)
    scip.setParam('separating/rlt/maxroundsroot', -1)

    # Now the general cut and round parameters
    scip.setParam("separating/maxroundsroot", max_rounds_root - 1)
    scip.setParam("separating/maxstallroundsroot", max_rounds_root - 1)
    scip.setParam("separating/maxcutsroot", 10000)

    # WARNING: We currently disable all cuts after the root node!
    scip.setParam("separating/maxrounds", 0)
    scip.setParam("separating/maxstallrounds", 0)
    scip.setParam("separating/maxcuts", 0)

    return scip


def remove_slurm_files(outfile_dir):
    """
    Removes all files from outfile_dir.
    Args:
        outfile_dir: The output directory containing all of our slurm .out files

    Returns:
        Nothing. It simply deletes the files
    """

    assert not outfile_dir == '/' and not outfile_dir == ''

    # Delete everything
    shutil.rmtree(outfile_dir)

    # Make the directory itself again
    os.mkdir(outfile_dir)

    return


def remove_temp_files(temp_dir):
    """
    Removes all files from the given directory
    Args:
        temp_dir: The directory containing all information that is batch specific

    Returns:
        Nothing, the function deletes all files in the given directory
    """

    # Get all files in the directory
    files = os.listdir(temp_dir)

    # Now cycle through the files and delete them
    for file in files:
        os.remove(os.path.join(temp_dir, file))

    return


def run_python_slurm_job(python_file, job_name, outfile, time_limit, arg_list, exclusive=True, dependencies=None):
    """
    Function for calling a python file through SLURM. This offloads the job from the current call, and lets multiple
    jobs run simulataneously. Please make sure to set up an appropriate dependency using safety_check.py so
    the main process does not continue to run while the other jobs are still running. All information
    between jobs is communicated through input / output files.
    Note: Spawned processes cannot directly communicate with each other
    Args:
        python_file: The python file that wil be run
        job_name: The name to give the python run in slurm
        outfile: The file in which all output from the python run will be stored
        time_limit: The time limit on the slurm job in seconds
        arg_list: The list containing all args that will be added to the python call
        exclusive: Whether the run should be exclusive
        dependencies: A list of slurm job ID dependencies that must first complete before this job starts
    Returns:
        Nothing. It simply starts a python job through the command line that will be run in slurm
    """

    if dependencies is None:
        dependencies = []
    assert os.path.isfile(python_file) and python_file.endswith('.py')
    assert not os.path.isfile(outfile) and outfile.endswith('.out'), '{}'.format(outfile)
    assert os.path.isdir(os.path.dirname(outfile)), '{}'.format(outfile)
    assert type(time_limit) == int and 0 <= time_limit <= 1e+8
    assert type(arg_list) == list
    assert dependencies is None or (type(dependencies) == list and
                                    all(type(dependency) == int for dependency in dependencies))

    # Get the current working environment.
    my_env = os.environ.copy()

    # Give the base command line call for running a single slurm job through shell.
    cmd_1 = ['sbatch',
             '--job-name={}'.format(job_name),
             '--time=0-00:00:{}'.format(time_limit)]

    # This flag makes the timing reproducible, as no memory is shared between it and other jobs.
    cmd_2 = ['--exclusive'] if exclusive else []
    # If you wanted to add memory limits; '--mem={}'.format(mem), where mem is in MB, e.g. 8000=8GB
    if parameters.SLURM_MEMORY is not None:
        cmd_2 += ['--mem={}'.format(parameters.SLURM_MEMORY)]
    if dependencies is not None and len(dependencies) > 0:
        # Add the dependencies if they exist
        dependency_str = ''.join([str(dependency) + ':' for dependency in dependencies])[:-1]
        cmd_2 += ['--dependency=afterany:{}'.format(dependency_str)]

    cmd_3 = ['-p',
             parameters.SLURM_QUEUE,
             '--output',
             outfile,
             '--error',
             outfile,
             '{}'.format(python_file)]

    cmd = cmd_1 + cmd_2 + cmd_3

    # Add all arguments of the python file afterwards
    for arg in arg_list:
        cmd.append('{}'.format(arg))

    # Run the command in shell.
    p = subprocess.Popen(cmd, env=my_env, stdout=subprocess.PIPE)
    p.wait()

    # Now access the stdout of the subprocess for the job ID
    job_line = ''
    for line in p.stdout:
        job_line = str(line.rstrip())
        break
    assert 'Submitted batch job' in job_line, print(job_line)
    job_id = int(job_line.split(' ')[-1].split("'")[0])

    del p

    return job_id


def get_filename(parent_dir, instance, rand_seed=None, permutation_seed=None, root=False, cutoff=None, ext='yml'):
    """
    The main function for retrieving the file names for all non-temporary files. It is a shortcut to avoid constantly
    rewriting the names of the different files, such as the .yml, .sol, .stats, and .mps files
    Args:
        parent_dir: The parent directory where the file belongs
        instance: The instance name of the SCIP problem
        rand_seed: The random seed used in the SCIP run
        permutation_seed: The random seed used to permute the variables and constraints before solving
        root: If root should be included in the file name
        cutoff: The type of distance measure used. Please see parameters.py for a complete list
        ext: The extension of the file, e.g. yml or sol
    Returns:
        The filename e.g. 'parent_dir/toll-like__trans__seed__2__permute__0__efficacy.mps'
    """

    # Initialise the base_file name. This always contains the instance name
    base_file = instance
    if root:
        base_file += '__root'
    if rand_seed is not None:
        base_file += '__seed__{}'.format(rand_seed)
    if permutation_seed is not None:
        base_file += '__permute__{}'.format(permutation_seed)
    if cutoff is not None:
        # assert cutoff in parameters.CUTOFF_OPTIONS
        base_file += '__{}'.format(cutoff)

    # Add the extension to the base file
    if ext is not None:
        base_file += '.{}'.format(ext)

    # Now join the file with its parent dir
    return os.path.join(parent_dir, base_file)


def get_slurm_output_file(outfile_dir, instance, cutoff=None, rand_seed=None):
    """
    Function for getting the slurm output log for the current run.
    # TODO: Currently we never run multiple permutation_seeds. Probably wouldnt run at once, but would collpase if did.
    Args:
        outfile_dir: The directory containing all slurm .log files
        instance: The instance name
        cutoff: The type of distance measure used. Please see parameters.py for a complete list
        rand_seed: The instance random seed
    Returns:
        The slurm .out file which is currently being used
    """

    assert os.path.isdir(outfile_dir)
    assert type(instance) == str
    assert rand_seed is None or type(rand_seed) == int
    assert cutoff is None or cutoff in parameters.CUTOFF_OPTIONS

    # Get all slurm out files
    out_files = os.listdir(outfile_dir)

    # Get a unique substring that will only be contained for a single run
    file_substring = '__{}'.format(instance)
    if cutoff is not None:
        file_substring += '__{}'.format(cutoff)
    if rand_seed is not None:
        file_substring += '__{}'.format(rand_seed)

    unique_file = [out_file for out_file in out_files if file_substring in out_file]
    assert len(unique_file) == 1, 'Instance {} with rand_seed {} has no outfile in {}'.format(instance, rand_seed,
                                                                                              outfile_dir)
    return os.path.join(outfile_dir, unique_file[0])


def str_to_bool(word):
    """
    This is used to check if a string is trying to represent a boolean True.
    We need this because argparse doesnt by default have such a function, and using bool('False') evaluates to True
    Args:
        word: The string we want to convert to a boolean
    Returns:
        Whether the string is representing True or not.
    """
    assert type(word) == str
    return word.lower() in ["yes", "true", "t", "1"]


def is_dir(path):
    """
    This is used to check if a string is trying to represent a directory when we parse it into argparse.
    Args:
        path: The path to a directory
    Returns:
        The string path if it is a valid directory else we raise an error
    """
    assert type(path) == str, print('{} is not a string!'.format(path))
    exists = os.path.isdir(path)
    if not exists:
        raise argparse.ArgumentTypeError('{} is not a valid directory'.format(path))
    else:
        return path


def is_file(path):
    """
    This is used to check if a string is trying to represent a file when we parse it into argparse.
    Args:
        path: The path to a file
    Returns:
        The string path if it is a valid file else we raise an error
    """
    assert type(path) == str, print('{} is not a string!'.format(path))
    exists = os.path.isfile(path)
    if not exists:
        raise argparse.ArgumentTypeError('{} is not a valid file'.format(path))
    else:
        return path


def get_instances(instance_dir, files_are_lps=False, files_are_gzipped=False, ignore_file_type_check=False):
    """
    Function for getting all instance strings. These are pulled from the instance directory names.
    Args:
        instance_dir (dir): The directory in which all of our instance files are stored
        files_are_lps: Whether the files are .lp files instead of .mps files
        files_are_gzipped: Whether the files are stored in compressed .gz format
        ignore_file_type_check: Whether to ignore the assertion checks of file types

    Returns:
        A list of all instances
    """

    # Make sure all files in this directory are instance files
    instance_files = os.listdir(instance_dir)

    if not ignore_file_type_check:
        for instance_file in instance_files:
            if files_are_lps:
                file_ending = '.lp'
            else:
                file_ending = '.mps'
            if files_are_gzipped:
                file_ending += '.gz'
            assert instance_file.endswith(file_ending), 'File {} in directory {} is not an instance file'.format(
                instance_file, instance_dir)

    instances = set()
    for instance_file in instance_files:
        if files_are_gzipped:
            instance_name = os.path.splitext(os.path.splitext(instance_file)[0])[0].split('__')[0]
        else:
            instance_name = os.path.splitext(instance_file)[0].split('__')[0]
        assert '__' not in instance_name, 'Instance {} is invalid'.format(instance_name)
        instances.add(instance_name)
    return sorted(list(instances))


def get_random_seeds(results_dir, cutoff_options, permutation_seeds):
    """
    Function for getting all random seeds. These are pulled from the result file names.
    Args:
        results_dir (dir): The directory in which all results are stored
        cutoff_options (list): The cutoff options that we are interested in (see parameters.py)
        permutation_seeds (list): The list of permutation seeds

    Returns:
        A list of all random seeds
    """

    assert os.path.isdir(results_dir)
    assert type(cutoff_options) == list and len(cutoff_options) > 0
    assert type(permutation_seeds) == list and len(permutation_seeds) > 0

    # Get the result files from a single cutoff / permutation seed combination
    result_files = os.listdir(os.path.join(results_dir, cutoff_options[0], str(permutation_seeds[0])))
    # Get the rand seeds from the file names
    rand_seeds = set()
    for result_file in result_files:
        rand_seed_str = os.path.splitext(result_file)[0].split('__')[-4]
        assert '__' not in rand_seed_str, 'Rand seed {} is invalid'.format(rand_seed_str)
        rand_seeds.add(int(rand_seed_str))

    # Make sure the random seeds are the same for all such cutoff and permutation seed combinations
    for cutoff in cutoff_options:
        for permutation_seed in permutation_seeds:
            cutoff_seeds = set()
            result_files = os.listdir(os.path.join(results_dir, cutoff, str(permutation_seed)))
            for result_file in result_files:
                rand_seed_str = os.path.splitext(result_file)[0].split('__')[-4]
                assert '__' not in rand_seed_str, 'Rand seed {} is invalid'.format(rand_seed_str)
                cutoff_seeds.add(int(rand_seed_str))
            assert rand_seeds == cutoff_seeds, 'Cutoff {} P-seed {} has different R-seeds than {}, {}'.format(
                cutoff, permutation_seed, cutoff_options[0], permutation_seeds[0])

    # Make sure the random seeds are in order
    rand_seeds = sorted(list(rand_seeds))
    for rand_seed_i, rand_seed in enumerate(rand_seeds):
        if rand_seed_i < len(rand_seeds) - 1:
            assert rand_seed == rand_seeds[
                rand_seed_i + 1] - 1, 'Random seeds {} do not represent a python range'.format(rand_seeds)
    return rand_seeds


def get_permutation_seeds(results_dir, cutoff_options):
    """
    Function for getting all permutation seeds. These are pulled from the result directory names.
    Args:
        results_dir (dir): The directory in which all of our results are stored
        cutoff_options (list): A list of all cutoff options (see parameters.py)

    Returns:
        A list of all permutation seeds
    """

    assert os.path.isdir(results_dir)
    assert type(cutoff_options) == list and len(cutoff_options) > 0

    # Get the permutation seed directory names (the seeds themselves). Make sure they're the same for all cutoffs
    permutation_seeds = [int(p) for p in sorted(os.listdir(os.path.join(results_dir, cutoff_options[0])))]
    for cutoff_option in cutoff_options:
        cutoff_seeds = [int(p) for p in sorted(os.listdir(os.path.join(results_dir, cutoff_option)))]
        assert cutoff_seeds == permutation_seeds, 'Cutoff Option {} has different permutation seeds to {}'.format(
            cutoff_options[0], cutoff_option)

    # Make sure the random seeds are in order
    for permutation_seed_i, permutation_seed in enumerate(permutation_seeds):
        if permutation_seed_i < len(permutation_seeds) - 1:
            assert permutation_seed == permutation_seeds[
                permutation_seed_i + 1] - 1, 'Random seeds {} do not represent a python range'.format(permutation_seeds)
    return permutation_seeds
