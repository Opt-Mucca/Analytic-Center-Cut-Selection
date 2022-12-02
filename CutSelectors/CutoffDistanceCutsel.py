#! /usr/bin/env python
from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING
from pyscipopt.scip import Cutsel
import random
import yaml
from parameters import *
from time import time
import numpy as np
import pdb


class CutoffDistanceCutsel(Cutsel):

    def __init__(self, good_score_factor=0.9, bad_score_factor=0.0, min_orthogonality_root=0.9,
                 min_orthogonality=0.9, dir_cutoff_dist_weight=0.0, int_support_weight=0.1,
                 obj_parallel_weight=0.1, efficacy_weight=1.0, analytic_dir_cutoff=False, analytic_efficacy=False,
                 approximate_analytic_dir_cutoff=False,
                 min_efficacy=False, average_efficacy=False,
                 average_multiple_primal_solutions=False, efficacy_check_projection=False,
                 expected_improvement=False, compare_all_methods=False, num_lp_solutions=3, max_density=1,
                 num_cuts_per_round=10):
        super().__init__()
        self.good_score_factor = good_score_factor
        self.bad_score_factor = bad_score_factor
        self.min_orthogonality_root = min_orthogonality_root
        self.min_orthogonality = min_orthogonality
        self.dir_cutoff_dist_weight = dir_cutoff_dist_weight
        self.int_support_weight = int_support_weight
        self.obj_parallel_weight = obj_parallel_weight
        self.efficacy_weight = efficacy_weight
        self.analytic_dir_cutoff = analytic_dir_cutoff
        self.analytic_efficacy = analytic_efficacy
        self.approximate_analytic_dir_cutoff = approximate_analytic_dir_cutoff
        self.min_efficacy = min_efficacy
        self.average_efficacy = average_efficacy
        self.average_multiple_primal_solutions = average_multiple_primal_solutions
        self.efficacy_check_projection = efficacy_check_projection
        self.expected_improvement = expected_improvement
        self.compare_all_methods = compare_all_methods
        self.num_lp_solutions = num_lp_solutions
        self.max_density = max_density
        self.analytic_dcd_sol = None
        self.average_primal_sol = None
        self.objective_parallelism_good_cuts = [0, 0]
        self.objective_parallelism_cuts_added = [0, 0]
        self.orthogonality_cuts_added = [0, 0]
        self.density_cuts_added = [0, 0]
        self.cuts_added_per_round = [0, 0]
        self.density_max_cuts_added = 0
        self.infeasible_lp_projections = [0, 0]
        self.new_approximate_analytic_dcd_sols = [0, 0]
        self.time_taken = 0
        self.num_primals = 0
        self.num_cuts_per_round = num_cuts_per_round
        self.cut_scores = {cutoff: [] for cutoff in CUTOFF_OPTIONS if ('0' not in cutoff and
                                                                       cutoff != 'compare_all_methods')}
        random.seed(42)
        np.random.seed(42)

        assert analytic_dir_cutoff + analytic_efficacy + approximate_analytic_dir_cutoff + \
               min_efficacy + average_efficacy + \
               average_multiple_primal_solutions + efficacy_check_projection + expected_improvement <= 1
        if compare_all_methods:
            assert analytic_dir_cutoff + analytic_efficacy + approximate_analytic_dir_cutoff + \
                   min_efficacy + average_efficacy + \
                   average_multiple_primal_solutions + efficacy_check_projection + expected_improvement == 0

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """
        This is the main function used to select cuts. It must be named cutselselect and is called by default when
        SCIP performs cut selection if the associated cut selector has been included (assuming no cutsel with higher
        priority was called successfully before).
        Args:
            cuts (list): List of pyscipopt Rows (The cuts that we sort, and the first nselectedcuts get added)
            forcedcuts (list): List of pyscipopt Row (The cuts that will be added no matter what)
            root (bool): Whether we are the root node or not
            maxnselectedcuts (int): The maximum number of selected cuts that can be selected from the list cuts

        Returns:
            The sorted list of cuts and the amount of cuts that we want to select
        """

        # If no cuts can be added, don't bother scoring or sorting them.
        if maxnselectedcuts <= 0 or len(cuts) <= 0:
            return {'cuts': cuts, 'nselectedcuts': 0,
                    'result': SCIP_RESULT.SUCCESS}

        # Initialise the timing measure
        start_time = time()

        # Initialise number cuts, number of selected cuts, and number of cuts that we will forcefully select
        n_cuts = len(cuts)
        nselectedcuts = 0
        num_cuts_to_select = min(maxnselectedcuts, max(self.num_cuts_per_round - len(forcedcuts), 0), n_cuts)

        # Initialises parallel thresholds.
        # good_max_parallel is allowed parallelism of any two added cuts that are rated as x% compared to the best cut
        # max_parallel is the allowed parallelism of two added cuts when at-least one of them is below x% the best cut
        max_parallel = 1 - self.min_orthogonality_root if root else 1 - self.min_orthogonality
        good_max_parallel = max(0.5, max_parallel)

        # This is if we want to compare the score of different cut measures. Ignore for single use cases
        if self.compare_all_methods:
            if len(cuts) > 0:
                for cutoff in CUTOFF_OPTIONS:
                    if '0' not in cutoff and cutoff != 'compare_all_methods':
                        self.set_cutoff_parameters(cutoff)
                        # Calculate the directed cutoff distance solution, efficacy solution and alternate lp solutions
                        dcd_sol, eff_sol, alternate_lp_solutions = self.get_dcd_eff_alternate_lp_sols()
                        _, scores = self.scoring(cuts, dcd_sol=dcd_sol, eff_sol=eff_sol,
                                                 alternate_lp_solutions=alternate_lp_solutions)
                        self.cut_scores[cutoff].append(scores)
                        if self.analytic_dir_cutoff:
                            self.free_solution(self.analytic_dcd_sol)
                        self.free_solution(eff_sol)
                        if alternate_lp_solutions is not None:
                            for lp_sol in alternate_lp_solutions:
                                self.free_solution(lp_sol)
            self.set_cutoff_parameters('efficacy')
            self.compare_all_methods = True

        # Filter out cuts by maximum density
        n_cuts, cuts = self.filter_with_density(n_cuts, nselectedcuts, cuts, self.max_density)

        if n_cuts <= 0:
            return {'cuts': cuts, 'nselectedcuts': 0, 'result': SCIP_RESULT.SUCCESS}

        # Calculate the directed cutoff distance solution, efficacy solution, and alternate lp solutions
        dcd_sol, eff_sol, alternate_lp_solutions = self.get_dcd_eff_alternate_lp_sols()

        # Generate the scores of each cut and thereby the maximum score
        max_non_forced_score, scores = self.scoring(cuts, dcd_sol=dcd_sol, eff_sol=eff_sol,
                                                    alternate_lp_solutions=alternate_lp_solutions)

        good_score = max_non_forced_score

        # Good_score allows a greater level of cut parallelism if score[cut] > good_score.
        good_score = good_score * self.good_score_factor

        # This filters out all cuts in cuts who are parallel to a forcedcut.
        for forced_cut in forcedcuts:
            n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, forced_cut, cuts,
                                                                scores, max_parallel, good_max_parallel, good_score)

        # Enter a while loop and keep adding cuts provided they exist and a limit has not been hit
        if maxnselectedcuts > 0:
            while n_cuts > 0:
                # Break the loop if we have selected the required amount of cuts
                if nselectedcuts == num_cuts_to_select:
                    break
                # re-sorts cuts and scores by putting the best cut at the beginning
                cuts, scores = self.select_best_cut(n_cuts, nselectedcuts, cuts, scores)
                nselectedcuts += 1
                n_cuts -= 1
                n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, cuts[nselectedcuts - 1],
                                                                    cuts,
                                                                    scores, max_parallel, good_max_parallel,
                                                                    good_score)

        # If we want to check the LP projection of each added cut and their LP feasibility.
        if self.efficacy_check_projection:
            cols = self.model.getLPColsData()
            t_vars = [col.getVar() for col in cols]
            for cut_i in range(nselectedcuts):
                _ = self.check_projected_lp_solution_on_cut_is_feasible(cuts[cut_i], eff_sol, t_vars)

        # Free the solutions we calculated
        self.free_solution(eff_sol)
        if alternate_lp_solutions is not None:
            for lp_sol in alternate_lp_solutions:
                self.free_solution(lp_sol)

        # Add the statistics of the added cuts
        self.add_statistics(cuts[:nselectedcuts])

        self.time_taken += time() - start_time

        return {'cuts': cuts, 'nselectedcuts': nselectedcuts,
                'result': SCIP_RESULT.SUCCESS}

    def scoring(self, cuts, dcd_sol=None, eff_sol=None, alternate_lp_solutions=None):
        """
        Scores each cut in cuts. The current rule is a weighted sum combination of the efficacy,
        directed cutoff distance, integer support, and objective function parallelism.
        Args:
            cuts (list): The list of cuts we want to find scores for
            dcd_sol (PySCIPOpt Solution): A solution we use for directed cutoff distance calculations
            eff_sol (PySCIPOpt Solution): A solution (maybe None for the LP Solution) that we use for efficacy calculations
            alternate_lp_solutions (list): A list of LP optimal PySCIPOpt Solutions

        Returns:
            The max score over all cuts, as well as the individual scores
        """

        # Initialise the scoring of each cut as well as the max_score
        scores = [0] * len(cuts)
        max_score = 0.0

        # Score the cuts
        for i in range(len(cuts)):

            # Get the int-support and obj-parallelism. (Default SCIP scoring measures)
            int_support = self.int_support_weight * self.model.getRowNumIntCols(cuts[i]) / cuts[i].getNNonz()
            obj_parallel = self.obj_parallel_weight * self.model.getRowObjParallelism(cuts[i])

            # Now go through the various efficacy calculations
            # Take the minimum efficacy over all LP solutions if this is the case
            if self.min_efficacy:
                if alternate_lp_solutions is not None:
                    efficacy = min([self.model.getCutEfficacy(cuts[i], sol=sol) for sol in alternate_lp_solutions])
                else:
                    efficacy = self.model.getCutEfficacy(cuts[i])
            # Take the average efficacy over all LP solutions if this is the case
            elif self.average_efficacy:
                if alternate_lp_solutions is not None:
                    efficacy = sum([self.model.getCutEfficacy(cuts[i], sol=sol) for sol in alternate_lp_solutions])
                    efficacy /= len(alternate_lp_solutions)
                else:
                    efficacy = self.model.getCutEfficacy(cuts[i])
            # Check if the projection of the LP solution onto the cut is LP feasible (for statistics)
            elif self.efficacy_check_projection:
                efficacy = self.model.getCutEfficacy(cuts[i], sol=eff_sol)
            # If expected objective improvement is the measure then calculate appropriately
            elif self.expected_improvement:
                # Note: The actual expected objective improvement needs to be multiplied by ||c||. As this is
                # constant across all cuts we simply remove it. Commented out is how you'd obtain ||c||.
                # cols = self.model.getLPColsData()
                # objective_norm = float(np.linalg.norm([col.getObjCoeff() for col in cols]))
                efficacy = self.model.getRowObjParallelism(cuts[i]) * self.model.getCutEfficacy(cuts[i], sol=eff_sol)
            # Default to the standard efficacy calculation
            else:
                efficacy = self.model.getCutEfficacy(cuts[i], sol=eff_sol)
            # We buff the efficacy calculation by the dir_cutoff_dist_weight if no dcd solution exists
            if dcd_sol is None:
                efficacy = efficacy * (self.efficacy_weight + self.dir_cutoff_dist_weight)
            else:
                efficacy = efficacy * self.efficacy_weight

            # Do the directed cutoff distance calculations
            if dcd_sol is not None:
                dir_cutoff_dist = self.dir_cutoff_dist_weight * self.model.getCutLPSolCutoffDistance(cuts[i], dcd_sol)
            else:
                dir_cutoff_dist = 0

            # Now sum the scores of all measurements
            score = obj_parallel + int_support + efficacy + dir_cutoff_dist

            # Add a small random perturbation to the scores to break tiebreaks
            score += random.uniform(0, 1e-6)

            # Keep track of the maximum score
            max_score = max(max_score, score)

            # Store the score of the cut
            scores[i] = score

        return max_score, scores

    def filter_with_parallelism(self, n_cuts, nselectedcuts, cut, cuts, scores, max_parallel, good_max_parallel,
                                good_score):
        """
        Filters the given cut list by any cut_iter in cuts that is too parallel to cut. It does this by moving the
        parallel cut to the back cuts, and decreasing the indices of the list that are scanned over.
        For the rest of the algorithm we then never touch this back portion of cuts.
        Args:
            n_cuts (int): The number of cuts that are still viable cnadidates
            nselectedcuts (int): The number of cuts already selected
            cut (PySCIPOpt Row): The cut that we will add, and are now using to filter the reminaing cuts
            cuts (list): The list of cuts
            scores (list): The list of scores of each cut
            max_parallel (float): The maximum allowed parallelism for non good cuts
            good_max_parallel (float): The maximum allowed parallelism for good cuts
            good_score (float): The benchmark of whether a cut is 'good' and should its allowed parallelism increased

        Returns:
             The now number of viable cuts, the complete list of cuts, and the complete list of scores
        """

        # Go backwards through the still viable cuts.
        for i in range(nselectedcuts + n_cuts - 1, nselectedcuts - 1, -1):
            cut_parallel = self.model.getRowParallelism(cut, cuts[i])
            # The maximum allowed parallelism depends on the whether the cut is 'good'
            allowed_parallel = good_max_parallel if scores[i] >= good_score else max_parallel
            if cut_parallel > allowed_parallel:
                # Throw the cut to the end of the viable cuts and decrease the number of viable cuts
                cuts[nselectedcuts + n_cuts - 1], cuts[i] = cuts[i], cuts[nselectedcuts + n_cuts - 1]
                scores[nselectedcuts + n_cuts - 1], scores[i] = scores[i], scores[nselectedcuts + n_cuts - 1]
                n_cuts -= 1

        return n_cuts, cuts, scores

    def filter_with_density(self, n_cuts, nselectedcuts, cuts, max_density=0.4):
        """
        Filters the given cut list for any cuts that are too dense. That is, contain more non_zeros than
        max_density * self.model.getNVars(). It does this by moving the
        parallel cut to the back cuts, and decreasing the indices of the list that are scanned over.
        For the rest of the algorithm we then never touch this back portion of cuts.

        Args:
            n_cuts (int): The number of cuts in the cuts list that we are actually considering
            nselectedcuts (int): The number of cuts that we have already selected from the list
            cuts (list): The list of cuts
            max_density (float): The threshold for filtering cuts by density. max_density * self.model.getNVars()

        Returns:
            The now number of viable cuts, and the complete list of cuts
        """

        # Return the same array if max density is >= 1
        if max_density >= 1:
            return n_cuts, cuts

        # Go backwards through the still viable cuts.
        max_density = max_density * self.model.getNLPCols()
        for i in range(nselectedcuts + n_cuts - 1, nselectedcuts - 1, -1):
            if max_density < cuts[i].getNNonz():
                # Throw the cut to the end of the viable cuts and decrease the number of viable cuts
                cuts[nselectedcuts + n_cuts - 1], cuts[i] = cuts[i], cuts[nselectedcuts + n_cuts - 1]
                n_cuts -= 1

        return n_cuts, cuts

    def select_best_cut(self, n_cuts, nselectedcuts, cuts, scores):
        """
        Moves the cut with highest score which is still considered viable (not too parallel to previous cuts) to the
        front of the list. Note that 'front' here still has the requirement that all added cuts are still behind it.

        Args:
            n_cuts (int): The number of still viable cuts
            nselectedcuts (int): The number of cuts already selected to be added
            cuts (list): The list of cuts themselves
            scores (list): The list of scores of each cut

        Returns:
            The re-sorted list of cuts, and the re-sorted list of scores
        """

        # Initialise the best index and score
        best_pos = nselectedcuts
        best_score = scores[nselectedcuts]
        for i in range(nselectedcuts + 1, nselectedcuts + n_cuts):
            if scores[i] > best_score:
                best_pos = i
                best_score = scores[i]
        # Move the cut with highest score to the front of the still viable cuts
        cuts[nselectedcuts], cuts[best_pos] = cuts[best_pos], cuts[nselectedcuts]
        scores[nselectedcuts], scores[best_pos] = scores[best_pos], scores[nselectedcuts]
        return cuts, scores

    def get_dcd_eff_alternate_lp_sols(self):
        """
        Main function for getting all alternate solutions used for distance measurements in rating cuts.
        It decides on the solutions that it will retrieve depending on the cutoff option set (see parameters.py).

        Returns:
            The dcd_sol (either None or a pysciopt Solution)
            The eff_sol (either None or a pyscipopt Solution)
            A list of alternate LP optimal solutions (either None or a list of pyscipopt Solutions)
        """
        if (self.min_efficacy or self.average_efficacy) and self.num_lp_solutions > 1:
            alternate_lp_solutions = self.generate_alternate_lp_solutions()
        else:
            alternate_lp_solutions = None

        if self.analytic_dir_cutoff:
            dcd_sol = self.get_analytic_dcd_sol()
        elif self.approximate_analytic_dir_cutoff:
            dcd_sol = self.get_approximate_analytic_dcd_sol()
        elif self.average_multiple_primal_solutions:
            dcd_sol = self.get_dcd_sol_from_average_primal()
        else:
            dcd_sol = self.model.getBestSol() if self.model.getNSols() > 0 else None

        # Check if dcd_sol is None. If so, replace with best solution is one exists
        if dcd_sol is None:
            dcd_sol = self.model.getBestSol() if self.model.getNSols() > 0 else None

        if self.analytic_efficacy:
            eff_sol = self.get_analytic_eff_sol()
        elif self.efficacy_check_projection:
            eff_sol = self.create_storage_for_original_lp_solution()
        else:
            eff_sol = None

        return dcd_sol, eff_sol, alternate_lp_solutions

    def create_storage_for_original_lp_solution(self):
        """
        Function for just copying the current LP optimal solution to storage.

        Returns:
            The current LP optimal solution stored in a new pyscipopt Solution
        """

        # Create an empty solution
        eff_sol = self.model.createSol()

        # Get the variables of the transformed space. Cycle over them and copy the LP solution into storage
        cols = self.model.getLPColsData()
        t_vars = [col.getVar() for col in cols]
        for t_var in t_vars:
            self.model.setSolVal(eff_sol, t_var, t_var.getLPSol())

        return eff_sol

    def get_analytic_eff_sol(self):
        """
        Function for getting the analytic center of the optimal face of the original LP formulation

        Returns:
            The pyscipopt Solution for the analytic center of the optimal face of the original LP (or None if an error)
        """

        self.model.startDive()
        lperror, cutoff = self.solve_dive_barrier_lp(face=True)
        if not lperror and self.model.getLPSolstat() == 1:
            eff_sol = self.model.createSol()
            t_vars = [col.getVar() for col in self.model.getLPColsData()]
            for t_var in t_vars:
                self.model.setSolVal(eff_sol, t_var, t_var.getLPSol())
        else:
            print('Error when solving Barrier LP', flush=True)
            eff_sol = None
        self.model.endDive()

        return eff_sol

    def get_analytic_dcd_sol(self):
        """
        Function for getting the analytic center of the original LP formulation.

        Returns:
            The pyscipopt Solution for the analytic center of the original LP. (Or None if an error)
        """

        # Start dive and solve the analytic center LP
        if self.analytic_dcd_sol is not None:
            self.free_solution(self.analytic_dcd_sol)
        self.model.startDive()
        lperror, cutoff = self.solve_dive_barrier_lp(face=False)
        if not lperror and self.model.getLPSolstat() == 1:
            dcd_sol = self.model.createSol()
            t_vars = [col.getVar() for col in self.model.getLPColsData()]
            for t_var in t_vars:
                self.model.setSolVal(dcd_sol, t_var, t_var.getLPSol())
        else:
            print('Error when solving Barrier LP', flush=True)
            dcd_sol = None
        self.model.endDive()
        self.analytic_dcd_sol = dcd_sol
        return dcd_sol

    def get_approximate_analytic_dcd_sol(self):
        """
        Function for getting the approximate analytic center of the original LP formulation.
        Approximate here means that we reuse older iterations of the algorithm (self.analytic_dcd_sol)
        The solutions are only reused if the remain LP feasible after the previous round of cuts have been applied.
        Warning: This method becomes very dangerous and misleading when sued in the branch and bound tree.
        Warning: The feasible regions shift at each node when branching, so reuse rarely happens

        Returns:
            The pyscipopt Solution used for approximate directed cutoff distance calculations
        """

        # If there is a previous solution that is no longer LP feasible, then remove it
        if self.analytic_dcd_sol is not None:
            if not self.check_lp_feasibility(self.analytic_dcd_sol):
                self.new_approximate_analytic_dcd_sols[0] += 1
                self.free_solution(self.analytic_dcd_sol)
                self.analytic_dcd_sol = None

        # If the previous solution is None then calculate a new one
        if self.analytic_dcd_sol is None:
            dcd_sol = self.get_analytic_dcd_sol()
        else:
            # Otherwise use the LP feasible older solution
            dcd_sol = self.analytic_dcd_sol

        self.new_approximate_analytic_dcd_sols[1] += 1
        return dcd_sol

    def get_dcd_sol_from_average_primal(self):
        """
        Function for aggregating the current primal solutions for use in the directed cutoff measurement.
        The previous iteration's average primal solution was stored in self.average_primal_sol

        Returns:
            A new pyscipopt Solution that is the average of all stored primal solutions
        """

        # Get the number of stored solutions
        n_sols = self.model.getNSols()

        # Free the old average primal
        self.free_solution(self.average_primal_sol)

        # If there are no stored solutions then return None
        if n_sols == 0:
            dcd_sol = None
        else:
            dcd_sol = self.model.createSol()
            cols = self.model.getLPColsData()
            t_vars = [col.getVar() for col in cols]
            primals = self.model.getSols()
            assert n_sols == len(primals)
            # For each variable, iterate over all primal solutions and average their values
            for t_var in t_vars:
                t_var_val = 0
                for primal in primals:
                    t_var_val += primal[t_var]
                self.model.setSolVal(dcd_sol, t_var, t_var_val / n_sols)
            self.average_primal_sol = dcd_sol

        return dcd_sol

    def generate_alternate_lp_solutions(self):
        """
        Function for generating alternate LP optimal solutions. It does this by fixing the feasible region to the
        optimal face of the original LP, and then optimising over a new objective in a DiveLP.

        Returns:
            Either None (Something failed, or there are not alternative LP solutions), or a list of alternate
            LP optimal solutions (pyscipopt Solutions)
        """

        if self.num_lp_solutions <= 1:
            print('Tried to generate LP solutions with a limit of 1! This is just the original LP sol', flush=True)
            return None

        # Get columns and variables of the LP
        cols = self.model.getLPColsData()
        t_vars = [col.getVar() for col in cols]
        rows = self.model.getLPRowsData()

        # Initialise list containing the alternate solutions we will find
        alternate_solutions = []

        # Get the basis status of each column
        basis_status_cols = [col.getBasisStatus() for col in cols]
        # Initialise list containing variables that could be swapped into basis while remaining on optimal face
        zero_red_cost_vars, zero_red_cost_dict = self.get_non_zero_reduced_cost_vars(t_vars, basis_status_cols)

        # If the optimal face is a single point, then simply return None
        if len(zero_red_cost_vars) <= 0:
            return None

        # Add the current LP solution to the LP solution storage
        original_lp_solution = self.model.createSol()
        for t_var in t_vars:
            self.model.setSolVal(original_lp_solution, t_var, t_var.getLPSol())
        alternate_solutions.append(original_lp_solution)

        # Get the old objective in numpy format
        old_obj = np.array([t_var.getObj() for t_var in t_vars])
        old_scip_obj = self.model.getObjective()

        # Enter diving mode so we don't influence the original formulation
        self.model.startDive()
        # Reduce the LP to that of the optimal face
        self.fix_to_optimal_face(t_vars, rows)

        # Now generate the alternate LP solutions
        for sol_i in range(self.num_lp_solutions - 1):

            # Change the objective of the DiveLP such that it points in the opposite to the original objective
            n_cols = self.model.getNLPCols()
            new_obj = np.random.randn(n_cols)
            new_obj -= new_obj.dot(old_obj) * old_obj
            new_obj /= np.linalg.norm(new_obj)
            # Noticed there were numeric troubles. Scale and round the objective values.
            new_obj = np.round(1000 * new_obj, decimals=2)
            for i, t_var in enumerate(t_vars):
                self.model.chgVarObjDive(t_var, float(new_obj[i]))

            # Solve the dive LP
            lperror, cutoff = self.model.solveDiveLP()
            # Check if the solve was successful
            if lperror or self.model.getLPSolstat() != 1:
                print('The dive LP was infeasible when finding alternate LP solutions. Defaulting to efficacy!',
                      flush=True)
                # If the solve failed, then free our generated LP solutions and end the dive process
                for alternate_sol in alternate_solutions:
                    self.model.freeSol(alternate_sol)
                self.model.endDive()
                return None

            # Store the alternate LP solution
            alternate_sol = self.model.createSol()
            for t_var in t_vars:
                self.model.setSolVal(alternate_sol, t_var, t_var.getLPSol())
            alternate_solutions.append(alternate_sol)

        self.model.endDive()

        return alternate_solutions

    def fix_to_optimal_face(self, t_vars, rows):
        """
        Function that fixes the LP to the optimal face. This is used so alternate LP optimal solutions can be found.
        WARNING: This function should only be used in dive mode!!!!
        Args:
            t_vars (list): The list of variables from the transformed space
            rows (list): The list of rows of the LP

        Returns:
            Nothing. It fixes the LP to the optimal face
        """

        # If this variable cannot be swapped into the basis without harming the objective, then fix it.
        for t_var in t_vars:
            t_var_val = t_var.getLPSol()
            if not self.model.isFeasZero(self.model.getVarRedcost(t_var)):
                self.model.chgVarLbDive(t_var, t_var_val)
                self.model.chgVarUbDive(t_var, t_var_val)

        # If this row cannot be swapped into the basis without harming the objective, then fix it
        for row in rows:
            dual_val = self.model.getRowDualSol(row)
            if not self.model.isFeasZero(dual_val):
                # TODO: Find out why chgRow(L)(R)hsDive removes columns to the LPI.
                # TODO: This results in the next round LP (after cuts are applied) to be solved from scratch
                if dual_val > 0 and self.model.isFeasEQ(row.getLhs(), self.model.getRowActivity(row)):
                    self.model.chgRowRhsDive(row, row.getLhs())
                elif dual_val < 0 and self.model.isFeasEQ(row.getRhs(), self.model.getRowActivity(row)):
                    self.model.chgRowLhsDive(row, row.getRhs())

        return

    def get_non_zero_reduced_cost_vars(self, t_vars, basis_status_cols):
        """
        This function is used for retrieving all variables in the transformed space that have zero reduced cost.
        These variables are those that are free on the optimal face

        Args:
            t_vars (list): A list of variables from the transformed space (list of pyscipopt Variables)
            basis_status_cols (list): A list of the basis status of all columns from the LP (ordered as t_vars)

        Returns:
            The list of t_vars with zero reduced cost, and a dictionary mapping from transformed variable names to
            whether the variable has zero reduced cost.
        """

        # Get the objective sense
        obj_sense = self.model.getObjectiveSense()

        # Initialise the data structures
        zero_red_cost_vars = []
        zero_red_cost_dict = {}

        # Cycle over all variables in the transformed space
        for i, t_var in enumerate(t_vars):
            # If the variable has zero reduced cost
            if self.model.isFeasZero(self.model.getVarRedcost(t_var)):
                # Depending on the direction, it matters whether the variable is at its LB or UB
                if obj_sense == 'minimize' and basis_status_cols[i] not in ['basis', 'lower']:
                    zero_red_cost_dict[t_var.name] = True
                    zero_red_cost_vars.append(t_var)
                elif obj_sense == 'maximize' and basis_status_cols[i] not in ['basis', 'upper']:
                    zero_red_cost_vars.append(t_var)
                    zero_red_cost_dict[t_var.name] = True
                else:
                    zero_red_cost_dict[t_var.name] = False
            else:
                zero_red_cost_dict[t_var.name] = False

        return zero_red_cost_vars, zero_red_cost_dict

    def check_projected_lp_solution_on_cut_is_feasible(self, cut, eff_sol, t_vars):
        """
        Function checks if the LP solution projected onto the cut is feasible w.r.t the LP.
        The formula used (where x is the LP optimal point and (a,b) is the cut of the form ax <= b is:
        x - (((<x,a> - b)/||a||) * a)

        Args:
            cut (PySCIPOpt Row): The cut that we are currently scoring
            eff_sol (PySCIPOpt Solution): The LP solution, or the solution that will be used for efficacy calculations
            t_vars (list): The list of transformed variables from the LP

        Returns:
            Whether the projection of the LP point onto the cut lies inside of the LP feasible region.
        """

        # Create the solution in which the projected solution will be stored
        projected_eff_sol = self.model.createSol()

        # Get all the information from the LP and the cut that will be needed for the calculations.
        cut_cols = cut.getCols()
        cut_vals = cut.getVals()
        t_cut_vars = [col.getVar() for col in cut_cols]

        # Set the solution value originally from LP solution. The projection only changed dimensions the cut lives in
        for t_var in t_vars:
            self.model.setSolVal(projected_eff_sol, t_var, eff_sol[t_var])

        # Create the scale factor that we will use. ((<x,a> - b)/||a||)
        scale = 0
        for i, t_var in enumerate(t_cut_vars):
            scale += eff_sol[t_var] * cut_vals[i]
        # Note that: lhs <= activity + cst <= rhs
        lhs = cut.getLhs()
        rhs = cut.getRhs()
        cst = cut.getConstant()
        assert not self.model.isInfinity(abs(cst))
        # Get the rhs / lhs of the row. We manually transform all rows into single rhs rows
        lhss = (lhs - cst) if not self.model.isInfinity(abs(lhs)) else 0
        rhss = (rhs - cst) if not self.model.isInfinity(abs(rhs)) else 0
        if lhss != 0 and rhss != 0:
            # Cannot project onto a ranged row!
            print('SCIP produced a ranged row as a cut!', flush=True)
            return False
        scale -= lhss + rhss
        scale /= cut.getNorm() ** 2

        # Set the new solution value for the variables involved in the cut
        for i, t_var in enumerate(t_cut_vars):
            self.model.setSolVal(projected_eff_sol, t_var, eff_sol[t_var] - (scale * cut_vals[i]))

        # Determine if the projected point is LP feasible
        feasible = self.check_lp_feasibility(projected_eff_sol)

        # Free the solution and return whether it was feasible w.r.t. the LP
        self.free_solution(projected_eff_sol)

        # Increment the logging values
        self.infeasible_lp_projections[0] += 1 if not feasible else 0
        self.infeasible_lp_projections[1] += 1

        return feasible

    def free_solution(self, sol):
        """
        Frees the given solution from storage
        Args:
            sol (PySCIPOpt Solution): Solution used for type of cutoff distance calculations

        Returns:
            Nothing
        """

        if sol is not None:
            self.model.freeSol(sol)

    def add_statistics(self, cuts):
        """
        Function for adding statistics to storage from the round of selected cuts.
        Args:
            cuts (list): List of cuts that will be added to the LP

        Returns:
            Nothing
        """

        # Cycle through the cuts and store the density / objective parallelism of the cuts
        num_vars = self.model.getNLPCols()
        for cut_i, cut in enumerate(cuts):
            density = cut.getNNonz() / num_vars
            self.density_cuts_added[0] += density
            self.density_cuts_added[1] += 1
            if density > self.density_max_cuts_added:
                self.density_max_cuts_added = density
            self.objective_parallelism_cuts_added[0] += self.model.getRowObjParallelism(cut)
            self.objective_parallelism_cuts_added[1] += 1

        # Log the number of cuts applied in the round
        self.cuts_added_per_round[1] += 1
        self.cuts_added_per_round[0] += len(cuts)

        # Do a pairwise comparison for all cuts to get the average orthogonality of applied cuts
        if len(cuts) > 1:
            for cut_i in range(len(cuts) - 1):
                for cut_j in range(cut_i + 1, len(cuts)):
                    self.orthogonality_cuts_added[1] += 1
                    self.orthogonality_cuts_added[0] += 1 - self.model.getRowParallelism(cuts[cut_i], cuts[cut_j])


    def get_statistics(self):
        """
        Turns all statistics into a dictionary. This can then be extracted by methods not attached
        to the cut selector.

        Returns:
            Dictionary of statistics
        """

        statistic_data = {}
        if self.objective_parallelism_good_cuts[1] > 0:
            average = self.objective_parallelism_good_cuts[0] / self.objective_parallelism_good_cuts[1]
            print('Average Objective Parallelism on Good Cuts {}'.format(average))
            statistic_data['objective_parallelism_good_cuts'] = average
        if self.objective_parallelism_cuts_added[1] > 0:
            average = self.objective_parallelism_cuts_added[0] / self.objective_parallelism_cuts_added[1]
            print('Average Objective Parallelism on Cuts Added {}'.format(average))
            statistic_data['objective_parallelism_cuts_added'] = average
        if self.orthogonality_cuts_added[1] > 0:
            average = self.orthogonality_cuts_added[0] / self.orthogonality_cuts_added[1]
            print('Average Orthogonality of Cuts Added {}'.format(average))
            statistic_data['orthogonality_cuts_added'] = average
        if self.density_cuts_added[1] > 0:
            average = self.density_cuts_added[0] / self.density_cuts_added[1]
            print('Average Density of Cuts Added {}'.format(average))
            statistic_data['density_cuts_added'] = average
        if self.cuts_added_per_round[1] > 0:
            average = self.cuts_added_per_round[0] / self.cuts_added_per_round[1]
            print('Average Number of Cuts Added Per Round {}'.format(average))
            statistic_data['cuts_added_per_round'] = average
        if self.density_max_cuts_added > 0:
            maximum = self.density_max_cuts_added
            print('Density of Max Cut {}'.format(maximum))
            statistic_data['max_cut_density'] = maximum
        if self.infeasible_lp_projections[1] > 0:
            infeasible_ratio = self.infeasible_lp_projections[0] / self.infeasible_lp_projections[1]
            print('Ratio of infeasible LP projections: {}'.format(infeasible_ratio))
            statistic_data['infeasible_lp_projection_ratio'] = infeasible_ratio
        if self.new_approximate_analytic_dcd_sols[1] > 1:
            ratio_solves = self.new_approximate_analytic_dcd_sols[0] / (self.new_approximate_analytic_dcd_sols[1] - 1)
            print('Ratio of new Analytic DCD Solves: {}'.format(ratio_solves))
            statistic_data['ratio_new_approximate_analytic_dcd_sol'] = ratio_solves

        statistic_data['cut_selection_time'] = self.time_taken

        return statistic_data

    def print_cut_scores(self, yml_file):
        """
        For the case we want to compare how individual methods would've scored cuts. Print out the individual scores.

        Args:
            yml_file (file): The YML file all cut scores would be dumped to

        Returns:
            Nothing
        """

        with open(yml_file, 'w') as s:
            yaml.dump(self.cut_scores, s)

    def set_cutoff_parameters(self, cutoff):
        """
        Function for resetting the distance measure we use in our cut selector.

        Args:
            cutoff (str): The distance measure of the experiment we want to use. See parameters.py for a full list

        Returns:
            Nothing
        """

        self.analytic_dir_cutoff = False
        self.analytic_efficacy = False
        self.approximate_analytic_dir_cutoff = False
        self.min_efficacy = False
        self.average_efficacy = False
        self.average_multiple_primal_solutions = False
        self.efficacy_check_projection = False
        self.expected_improvement = False
        self.compare_all_methods = False
        self.efficacy_weight = 0.0
        self.dir_cutoff_dist_weight = 0.0

        if cutoff in EFFICACY_CUTOFF_OPTIONS or cutoff == 'compare_all_methods':
            self.efficacy_weight = 1.0
            self.dir_cutoff_dist_weight = 0.0
        else:
            self.efficacy_weight = 0.0
            self.dir_cutoff_dist_weight = 1.0

        if cutoff == 'analytic_directed_cutoff':
            self.analytic_dir_cutoff = True
        elif cutoff == 'analytic_efficacy':
            self.analytic_efficacy = True
        elif cutoff == 'approximate_analytic_directed_cutoff':
            self.approximate_analytic_dir_cutoff = True
        elif cutoff == 'average_efficacy':
            self.average_efficacy = True
        elif cutoff == 'minimum_efficacy':
            self.min_efficacy = True
        elif cutoff == 'efficacy_check_projection':
            self.efficacy_check_projection = True
        elif cutoff == 'average_primal_directed_cutoff':
            self.average_multiple_primal_solutions = True
        elif cutoff == 'expected_improvement':
            self.expected_improvement = True
        else:
            if cutoff not in ['efficacy', 'directed_cutoff']:
                print('Invalid cutoff option given: {}'.format(cutoff), flush=True)
                quit()

    def check_lp_feasibility(self, sol=None, epsilon=1e-6):
        """
        Function for checking LP feasibility
        Args:
            sol (PySCIPOpt Solution): The solution we want to check for lp feasibility. If None, then LP solution
            epsilon (float): The epsilon used for bound checks
        Returns:
            True or False
        """

        # First check the LP rows for feasibility
        lp_rows = self.model.getLPRowsData()
        for row in lp_rows:
            cols = row.getCols()
            vals = row.getVals()
            assert len(vals) == len(cols)
            activity = 0
            for i in range(len(cols)):
                if sol is not None:
                    activity += vals[i] * sol[cols[i].getVar()]
                else:
                    activity += vals[i] * cols[i].getVar().getLPSol()
            activity += row.getConstant()
            if self.model.isLT(activity + epsilon, row.getLhs()) or self.model.isLT(row.getRhs(), activity - epsilon):
                # print(row.name, row.getLhs(), activity, row.getRhs())
                return False

        # Now check the LP cols for feasibility
        lp_cols = self.model.getLPColsData()
        for col in lp_cols:
            t_var = col.getVar()
            lb = t_var.getLbLocal()
            ub = t_var.getUbLocal()
            if sol is not None:
                val = sol[t_var]
            else:
                val = t_var.getLPSol()
            if self.model.isLT(val + epsilon, lb) or self.model.isLT(ub, val - epsilon):
                # print(t_var.name, lb, val, ub)
                return False

        return True

    def solve_dive_barrier_lp(self, face=True):
        """
        Solves the LP of the current dive using barrier method without crossover.
        The result of this solve is the analytic center (when face=False), and the analytic center of the optimal face,
        when face=True.
        Args:
            face (bool): Whether we want the analytic center of the optimal face, or the feasible region itself
        Returns:
            lperror (bool): Was there an error during solving
            cutoff (bool): Has an objective limit been reached
        """

        # Get the current solve parameters
        initial_algorithm = self.model.getParam('lp/initalgorithm')
        resolve_algorithm = self.model.getParam('lp/resolvealgorithm')
        check_dual_feasibility = self.model.getParam('lp/checkdualfeas')
        disable_cutoff = self.model.getParam('lp/disablecutoff')
        solution_polishing = self.model.getParam('lp/solutionpolishing')
        check_stability = self.model.getParam('lp/checkstability')
        check_farkas = self.model.getParam('lp/checkfarkas')
        barrier_convergence_tolerance = self.model.getParam('numerics/barrierconvtol')

        # Set the solve parameters to barrier without crossover
        self.model.setParam('lp/initalgorithm', 'b')
        self.model.setParam('lp/resolvealgorithm', 'b')
        self.model.setParam('lp/checkdualfeas', 0)
        self.model.setParam('lp/disablecutoff', 1)
        self.model.setParam('lp/solutionpolishing', 0)
        self.model.setParam('lp/checkstability', 0)
        self.model.setParam('lp/checkfarkas', 0)
        # self.model.setParam('numerics/barrierconvtol', 1e-9)

        # Change the objective to 0 in the case of face==False (analytic center of the entire polytope)
        # For the case of face==True (analytic center of optimal face), double the objective (to force a resolve)
        t_vars = [col.getVar() for col in self.model.getLPColsData()]
        for t_var in t_vars:
            obj_val = 0 if not face else 2 * t_var.getObj()
            self.model.chgVarObjDive(t_var, obj_val)

        # Solve the DiveLP
        lperror, cutoff = self.model.solveDiveLP()

        # Set the parameters back to their original values
        self.model.setParam('lp/initalgorithm', initial_algorithm)
        self.model.setParam('lp/resolvealgorithm', resolve_algorithm)
        self.model.setParam('lp/checkdualfeas', check_dual_feasibility)
        self.model.setParam('lp/disablecutoff', disable_cutoff)
        self.model.setParam('lp/solutionpolishing', solution_polishing)
        self.model.setParam('lp/checkstability', check_stability)
        self.model.setParam('lp/checkfarkas', check_farkas)
        self.model.setParam('numerics/barrierconvtol', barrier_convergence_tolerance)

        return lperror, cutoff
