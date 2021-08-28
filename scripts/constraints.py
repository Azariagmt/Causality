from causalnex.structure.notears import from_pandas, from_pandas_lasso

class Constraints:
    """
    Aids construct manual interference on structural model
    """

    def __init__(self, structural_model:from_pandas_lasso = None):
        self.structural_model = structural_model

    def add_edge(self, cause, effect) -> from_pandas_lasso:
        self.structural_model.add_edge(cause, effect)

    def remove_edge(self, cause,effect):
        self.structural_model.remove_edge(cause, effect)

    def get_model(self)->from_pandas_lasso:
        """Gets constrainted structural model

        Returns:
            from_pandas_lasso: Returns Constraint added structural model
        """
        return self.structural_model

constraint = Constraints()
sm = constraint.structural_model