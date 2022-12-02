# Analytic-Center-Cut-Selection

If this software was used for academic purposes, please cite our paper with the below information:

TODO: Put the bibtex entry here


## Install Guide
Requirements: Python 3.8 / Ubuntu (18.04 / 20.04) (Probably installs on other Ubuntu versions, and
if you change the virtual environment a bit it should work on Python 3.{6,7,9}).
We use SLURM https://slurm.schedmd.com/overview.html as a job manager. 
All calls go through a central function however, and in theory SLURM could be replaced by python's 
default multiprocessing package.

Run the bash script init_venv. If you don't use bash, configure the shebang (first line of the script) 
to be your shell interpreter.

`./init_venv`

After installing the virtual environment, make sure to always activate it with the script printed beneath. 
This is so your python path is appended and files at different directory levels can import from each other.

`source ./set_venv`

Now go and install SCIP from (https://www.scipopt.org/index.php#download / https://github.com/scipopt/scip)
For Ubuntu / Debian, the .sh installer is the easiest choice if you don't want to configure it yourself). 
The cut selector plugin, which features heavily in this research is available from SCIP 8.0+. We used SCIP 8.0.2

You can test if SCIP is installed by locating /bin/scip and calling it through the command line. 
SCIP should hopefully open.

One then needs to install PySCIPOpt https://github.com/scipopt/PySCIPOpt. 
I would recommend following the INSTALL.md guide. Make sure to have set your environment variable pointing to SCIP! 
You can test if this has been properly installed by running one of the tests, or by trying to import Model. 
This research was done using PySCIPOpt 4.2.0. 

How to run the software
We use Nohup https://en.wikipedia.org/wiki/Nohup to run all of our jobs and to capture output 
of the main function calls. It also allows jobs to be started through a SHH connection. 
Using this is not necessary however, so feel free to use whatever software you prefer. 
An example call to redirect output to nohup/nohup.out and to run the process in the background would be

`nohup python dir/example.py > nohup/nohup.out &`

## Run Guide

- Download instances (and potentially solutions) that you want to perform experiments on 

There's an included script for downloading MIPLIB 2017 https://miplib.zib.de/ instances / solutions.
One can run the following python call to download all instances and the best currently available solutions:

`python Instances/MIPLIB2017/get_instances.py`

It is then possible to check the solutions you've downloaded and if they are feasible for SCIP using

`python Instances/MIPLIB2017/try_solutions.py`

Note: This repository is not constrained to MIPLIB2017 instances. One can use any instance set they want.

- Run experiments using distance measures over the instance set

The main script for running all instances is `Slurm/run_all_methods.py`. It is the main function that ceates
individual jobs for solving instances, and iterates over all instance-permutation_seed-random_seed pairs. 
It stores all results files from these runs as well. It will use distance measures as defined in 
`parameters.py:CUTOFF_OPTIONS`. An example of how to call this function is as below:

`nohup python Slurm/run_all_methods.py instance_dir solution_dir results_dir outfile_dir solutions_are_compressed
num_rand_seeds num_permutation_seeds time_limit root fixed_cuts print_stats remove_previous_runs single_cutoff_option >
nohup/run_all_methods.out &`

instance_dir - The directory where all your instances are stored \
solution_dir - The directory where all your solutions are stored \
results_dir - The directory where you want to store all results \
outfile_dir - The directory where you want to store all output files (superset of SCIP logs) \
solutions_are_compressed - Whether you have gzipped your solution files \
num_rand_seeds - The number of random seeds in SCIP to use. Will use {1, 2, ...., n} (We used 3)\
num_permutation_seeds - The number of SCIP permutation seeds to use. Will use {0, 1, ....., n-1}. (We used 1) \
time_limit - What time limit do you want to set for each run in (s). (We used 600 for root node and 7200 for tree) \
root - Whether you want to fix results to the root node \
fixed_cuts - Whether you want to use the custom cut selector (Always set to True. Is an outdated option) \
print_stats - Whether you want to store the SCIP .stats file (Recommended if you have the space) \
remove_previous_runs - Whether you want to delete all previous run data. (Set True if previous runs have a bug) \
single_cutoff_option - If you want to run a single distance measure from `CUTOFF_OPTIONS`

- Generate Features

If you want to generate a feature set for each instance and try and create a mapping from the feature space to
some solving method, then you require this step (For example, if you want to train the support vector regression model).
This solves the root node of the instance, and disables nearly all SCIP plugins, such as cutting and heuristics.
It then includes a branching rule with the highest priority that extracts all relevant root node information, which
would be the information available to internal SCIP before the separation rounds begin. You can run the script as:

`nohup python Slurm/generate_features.py instance_dir solution_dir features_dir outfile_dir num_rand_seeds
num_permutation_seeds time_limit > nohup/generate_features.out &`

- Concatenate Result Files

The output of `run_all_methods` is a large amount of individual result files. Many of these result files
may also be undesirable. We use the script `scan_results` to concatenate all instances that have desirable
properites into a single large results file. We can then use this for easier analysis and visualisation. 

`python scripts/scan_results.py instance_dir results_dir root remove_small_primal_dual_diff`

The option remove_small_primal_dual_diff was always set to False for our analysis. It can be enabled if you want
to remove instances for which a minimum movement requirement of the root node dual bound is not met.

- Train a Multi-output Regression Model

Now that we have all the results file from individual runs, and generated features for all instances, we
can train a regression model for learning different distance measures. To do this, there is a script `regression_model`.
This script loads the features and results data, build either a support vector regression model or a random forest 
regressor, trains the model using cross validation, and then visualises the learned model. You can run the
script using the following command:

`python scripts/regression_model.py results_yml features_dir`

#### Thanks for Reading!! I hope this code helps you with your research. Please feel free to send any issues.
