# Libs

# Internal
from bayesian_optimization.acquisition.acquisition_function import AcquisitionFunction

# Type hinting

class MIPAcquisitionFunction(AcquisitionFunction):
    """Abstract class defining a Acquisition function formulation which is applicable for a
    Mixed Integer Problem (MIP).
    A MIP requires a expression as formulation for the acquisition function

    """

    def __init__(self):
        super().__init__()

    def single_expression(self, model, mu, sigma, incumbent):
        """Expression Formulation of the acquisition function at one single point

        :param model: cplex model
        :param mu: mean prediction
        :param sigma: variance of the prediction
        :param incumbent: previously best evaluation
        :return: Acquisition function formulation expression
        """
        pass

