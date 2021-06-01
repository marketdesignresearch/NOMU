# Libs
import logging
import pickle
import numpy as np
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

# Internals
from bayesian_optimization.acquisition.acquisition_function import AcquisitionFunction
from bayesian_optimization.context.inspector import Inspector
from bayesian_optimization.estimators.estimator import Estimator
from bayesian_optimization.nn_models.nn_model import NNModel

# Type hints
from typing import NoReturn, Union


class Context:
    """Context class that defines the configuration of the BO run.
    By configuring the context it can be defined with estimators, optimizers
    and models to use. It can also be configured which data to store during the process.
    """

    def __init__(
            self,
            callback,
    ):
        self.reset_model_for_each_fit = True
        self.reset_model_optimizer_for_each_fit = True
        self.acq_optimizer = None
        self.acq = None
        self.estimator = None
        self.callback = callback
        self.samples_x = None
        self.samples_y = None
        self.start_samples_x = None
        self.start_samples_y = None
        self.inspector = None
        self.nn_model: Union[NNModel, None] = None
        self.model_optimizer = None
        self._model_optimizer_config = None
        self.plotting_methods = {}
        self.reset_nn_model_for_each_fit = True
        self.bo_step = 0
        self._logger = logging.getLogger("context")
        self._logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self._logger.addHandler(ch)
        self.bo_step = 0
        self.out_path = None

    def set_out_path(self, path):
        self.out_path = path

    def reset_model_optimizer(self) -> NoReturn:
        """reset the optimizer so that is looses it's momentum.
        """
        if self.reset_model_optimizer_for_each_fit:
            #self._logger.info("reset model optimizer with config: " + str(self._model_optimizer_config))
            self.model_optimizer = self.model_optimizer.from_config(self._model_optimizer_config)

    def reset_model(self, model: 'NNModel') -> NoReturn:
        """reset the optimizer so that is looses it's momentum.
        """
        assert model is not None, "nn_model can not be reset since it is not defined"
        if self.reset_model_for_each_fit:
            # self._logger.info("reset nn-model of class: " + str(model.__class__.__name__))
            model._create_model()

    def set_model_optimizer(self, model_optimizer: OptimizerV2) -> NoReturn:
        """define which model optimizer should be used for the neuronal network model.
        Possible optimizers are SDG, Adam or more...
        This will also save the configuration of the optimizer so that a fresh optimizer can be created later
        to use a fresh optimizer without any existing momentum.

        :param model_optimizer: Keras model optimizer
        """
        self.model_optimizer = model_optimizer
        self._model_optimizer_config = self.model_optimizer.get_config()

    def set_estimator(self, estimator: Estimator) -> NoReturn:
        """define which estimator to use for the prediction of the mu and sigma.

        :param estimator: Estimator for mu and sigma
        :return:
        """
        self.estimator = estimator
        self.estimator.set_context(self)

    def set_acq(self, acq: AcquisitionFunction) -> NoReturn:
        """define which acquisition function to use.
        :param acq: Acquisition function for which the arg-max is the preferred point to evaluate next
        """
        self.acq = acq
        self.acq.set_context(self)

    def set_acq_optimizer(self, acq_optimizer) -> NoReturn:
        """define which optimizer should be used to find the optima of the acquisition function
        :param acq_optimizer: Acquisition function optimizer
        """
        self.acq_optimizer = acq_optimizer
        self.acq_optimizer.set_context(self)

    def set_samples_x(self, samples_x: np.array) -> NoReturn:
        """define the sample input values. For 1D- function these represent the values on the x-Axis
        :param samples_x: input values of the samples
        """
        self.samples_x = samples_x
        self.start_samples_x = samples_x

    def set_samples_y(self, samples_y: np.array) -> NoReturn:
        """define the sample target values. For 1D-function these represent the values on the y-Axis
        :param samples_y: target values of the samples
        """
        self.samples_y = samples_y
        self.start_samples_y = samples_y

    def set_samples(self, samples_x: np.array, samples_y: np.array) -> NoReturn:
        """define the samples/evaluations that shall be used for the training/fitting of the estimator.
        :param samples_x: input values of the samples
        :param samples_y: target values of the samples
        """
        self.set_samples_x(samples_x)
        self.set_samples_y(samples_y)

    def set_inspector(self, inspector: Inspector) -> NoReturn:
        """define the inspector. The Inspector allows for additional insight into the different processes and allows
        to run them on additional test data.

        :param inspector:
        :return:
        """
        self.inspector = inspector
        self.inspector.set_context(self)

    def set_network_model(self, nn_model: NNModel) -> NoReturn:
        """define the neural network that should be used in the neural network estimator if such a estimator is used.
        :param nn_model: Neural Network model class
        """
        self.nn_model = nn_model
        self.nn_model.set_context(self)

    def set_bo_step(self, step: int) -> NoReturn:
        """set counter for the BO-steps
        :param step: counter integer
        """
        self.bo_step = step

    def save_as_pickle(self, path) -> NoReturn:
        """saves the whole context as a pickle file
        :param path: path where to store the pickle
        :return:
        """
        pickle.dump(self, open(path, "wb"))

