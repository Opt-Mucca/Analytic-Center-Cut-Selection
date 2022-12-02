"""File containing the settings for the experiments you want to perform.
Each parameter here affects different bits of the experiments. The individual comments outline how.
"""

# The number of cut-selection rounds we will apply. Note that this is only used when restricted to the root node.
# In the paper this was set to 50
NUM_CUT_ROUNDS = 50

# The number of cuts that are attempted to be applied at each cut-selection round. Note that if not enough cuts are
# presented to the cut-selector, then all those presented are applied. This was set to 10 for the paper.
NUM_CUTS_PER_ROUND = 10

# The minimum allowed primal-dual difference. Any instance that has a primal-dual difference lower than this
# value under standard settings or for any other setting is discarded.
# In the paper this was not used. It was only to see instances with bound unaffected by cuts. (If used we recommend 0.5)
MIN_PRIMAL_DUAL_DIFFERENCE = 0.5

# Make sure to set this before you begin any runs. This is the SLURM queue that'll be used
# TODO: The user must define this by themselves!!!!!
SLURM_QUEUE = ''

# If you want to implement memory limits on the individual slurm jobs (units are megabytes). Set to None to ignore.
# This was set to None in the paper.
SLURM_MEMORY = None  # 96000

# This is a list of cutoff options that are used for various analysis of results
DENSITY_CUTOFF_OPTIONS = ['efficacy', 'efficacy_05', 'efficacy_10', 'efficacy_20',
                          'efficacy_40', 'efficacy_80']
DISTANCE_CUTOFF_OPTIONS = ['analytic_directed_cutoff', 'approximate_analytic_directed_cutoff', 'analytic_efficacy',
                           'minimum_efficacy', 'average_efficacy', 'efficacy', 'directed_cutoff',
                           'expected_improvement']
CHECK_PROJ_CUTOFF_OPTIONS = ['efficacy_check_projection']
EFFICACY_CUTOFF_OPTIONS = ['efficacy', 'efficacy_05', 'efficacy_10', 'efficacy_20', 'efficacy_40', 'efficacy_80',
                           'analytic_efficacy', 'average_efficacy', 'minimum_efficacy', 'expected_improvement']
DIR_CUTOFF_OPTIONS = ['approximate_analytic_directed_cutoff', 'analytic_directed_cutoff', 'directed_cutoff',
                      'average_primal_directed_cutoff']

# This is the list of distance measures used in the actual paper.
# We removed 'average_primal_directed_cutoff' due to pre-loading an optimal solution.
# We removed 'compare_all_methods' as it is not a single measure.
CUTOFF_OPTIONS = ['approximate_analytic_directed_cutoff', 'analytic_efficacy',
                  'analytic_directed_cutoff', 'directed_cutoff',
                  'average_efficacy', 'minimum_efficacy', 'expected_improvement',
                  'efficacy', 'efficacy_05', 'efficacy_10', 'efficacy_20', 'efficacy_40', 'efficacy_80']
