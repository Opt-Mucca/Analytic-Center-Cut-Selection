from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING, Branchrule
import logging
import pdb


class RootNodeFeatureExtractor(Branchrule):

    def __init__(self):
        super().__init__()
        self.features = {}
        self.count = 0

    def extract_features(self):
        """
        Function for extracting features from the pyscipopt model. This needs to be called after solving the
        root LP as we need the current LP information.
        We only take features that are readily available after the initial LP solve. (Note that the values
        would change as the solving process progresses)

        Returns:
            Nothing
        """

        # Initialise the features
        self.features['dual_degeneracy'] = 0
        self.features['primal_degeneracy'] = 0
        self.features['lp_solution_fractionality'] = 0
        self.features['ratio_of_equality_constraints'] = 0
        self.features['density_of_constraint_matrix'] = 0
        self.features['max_row_density'] = 0

        # Retrieve the LP column data
        cols = self.model.getLPColsData()
        ncols = self.model.getNLPCols()

        # Initialise the data we will retrieve from the columns
        num_basic_vars = 0
        num_basic_vars_at_var_bounds = 0
        num_non_basic_vars = 0
        num_non_basic_vars_with_zero_reduced_cost = 0
        num_base_zero_vars = 0
        num_int_vars = 0
        num_frac_int_vars = 0

        # Iterate over all the LP columns
        for col in cols:

            # Gets variable this column represents
            var = col.getVar()

            # Calculate the values necessary for dual and primal degeneracy
            basis_stat = col.getBasisStatus()
            if basis_stat == 'basic':
                num_basic_vars += 1
                col_lb = col.getLb()
                col_ub = col.getUb()
                sol = col.getPrimsol()
                if self.model.isEQ(col_lb, sol) or self.model.isEQ(col_ub, sol):
                    num_basic_vars_at_var_bounds += 1
            elif basis_stat == 'lower' or basis_stat == 'upper':
                num_non_basic_vars += 1
                red_cost = self.model.getVarRedcost(var)
                if self.model.isEQ(red_cost, 0):
                    num_non_basic_vars_with_zero_reduced_cost += 1
            else:
                assert basis_stat == 'zero'
                num_base_zero_vars += 1

            # Keep track of the integer variables, and those with fractional LP solution values
            vtype = var.vtype()
            if vtype in ['BINARY', 'INTEGER', 'IMPLINT']:
                num_int_vars += 1
                if not self.model.isFeasIntegral(var.getLPSol()):
                    num_frac_int_vars += 1

        # We now cycle through the constraints
        rows = self.model.getLPRowsData()
        nrows = self.model.getNLPRows()

        # Initialise a dual_sol_val list so we can normalise our results later
        num_equality_rows = 0
        num_row_entries = 0

        # Iterate over the rows
        for row in rows:
            # Check if the LP row is an equality constraint
            if self.model.isEQ(row.getLhs(), row.getRhs()):
                num_equality_rows += 1
            # Log the number of entries
            num_row_entries += row.getNLPNonz()
            # Keep track of the maximum row density
            if row.getNLPNonz() / ncols > self.features['max_row_density']:
                self.features['max_row_density'] = row.getNLPNonz() / ncols

        # Now actually assign all the values
        # First the dual degeneracy
        assert num_non_basic_vars + num_basic_vars + num_base_zero_vars == ncols
        assert num_non_basic_vars_with_zero_reduced_cost <= num_non_basic_vars
        if num_non_basic_vars >= 1:
            self.features['dual_degeneracy'] = num_non_basic_vars_with_zero_reduced_cost / num_non_basic_vars
        if num_basic_vars >= 1:
            self.features['primal_degeneracy'] = num_basic_vars_at_var_bounds / num_basic_vars
        if num_int_vars >= 1:
            self.features['lp_solution_fractionality'] = num_frac_int_vars / num_int_vars
        if nrows >= 1:
            self.features['ratio_of_equality_constraints'] = num_equality_rows / nrows
        if nrows >= 1 and ncols >= 1:
            self.features['density_of_constraint_matrix'] = num_row_entries / (nrows * ncols)

        return

    def branchexeclp(self, allowaddcons):
        # Iterate the number of times this branching rule has been called
        self.count += 1
        # This branching rule should interrupt solve after being called once!
        if self.count >= 2:
            logging.error('Dummy branch rule is called after root node and its first child')
            quit()
        assert allowaddcons

        assert not self.model.inRepropagation()
        assert not self.model.inProbing()

        # Extract the features from the model
        self.extract_features()
        # Interrupt the solving process. We have extracted the features and have no need to solve the model
        self.model.interruptSolve()

        # Make a dummy child. This branch rule should only be used at the root node!
        self.model.createChild(1, 1)
        return {"result": SCIP_RESULT.BRANCHED}
