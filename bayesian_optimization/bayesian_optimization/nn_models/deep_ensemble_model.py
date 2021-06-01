"""
Some code recycled and adopted with permission from https://gitlab.ethz.ch/jakobwe/Pseudo_Uncertainty_Bounds_for_NNs
"""

from typing import *
from typing import NoReturn

import numpy as np
from bayesian_optimization.nn_models.nn_model import NNModel
from bayesian_optimization.utils.utils import config_int_or_none
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomUniform  # , GlorotUniform
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from bayesian_optimization.acq_optimizer.gridsearch import GridSearch


class DeepEnsembleModel(NNModel):
    """Implementation of the Deep Ensembles as a class
    """

    def __init__(
            self,
            layers: List,
            activation: str,
            RSN: bool,
            s: float,
            seed: int,
            l2reg: float,
            softplus_min_var: float,
            model_name: str,
            loss=None,
            no_noise: bool = False,
            dropout_prob: float = 0
    ):
        self.model_name = model_name
        self.softplus_min_var = softplus_min_var
        self._n_hidden_layers = len(layers[0][2: -1])
        self.dropout_prob = dropout_prob
        self.no_noise = no_noise

        super().__init__(layers, activation, RSN, s, seed, l2reg, loss)

        if self.loss is None:
            self.loss = self.gaussian_nll # TODO this is a hard overwrite

    def _create_main_architecture(self, x_input: 'Input') -> 'Model':
        """creates the main architecture of the metwork according
        :param x_input: input layer to the network
        :return: constructed network model
        """
        # first hidden layer
        layer = Dense(
            self.main_layers[1],
            activation=self.activation,
            name=self.model_name + "_hidden_layer_{}".format(1),
            kernel_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(self.seed, 0)
            ),
            bias_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(self.seed, 1)
            ),
            kernel_regularizer=l2(self.l2reg),
            bias_regularizer=l2(self.l2reg)
        )(x_input)
        if self.dropout_prob != 0:
            layer = Dropout(self.dropout_prob)(layer)


        # hidden layers
        for i, n in enumerate(self.main_layers[2: -1]):
            layer = Dense(
                n,
                activation=self.activation,
                name=self.model_name + "_hidden_layer_{}".format(i + 2),
                kernel_initializer=RandomUniform(
                    minval=-self.s,
                    maxval=self.s,
                    seed=self._update_seed(self.seed, 2 * i + 2)
                ),
                bias_initializer=RandomUniform(
                    minval=-self.s,
                    maxval=self.s,
                    seed=self._update_seed(self.seed, 2 * i + 3)
                ),
                kernel_regularizer=l2(self.l2reg),
                bias_regularizer=l2(self.l2reg))(layer)
        if self.dropout_prob != 0:
            layer = Dropout(self.dropout_prob)(layer)

        # output layer with two parameters for each output dimension:
        mu_output = Dense(
            self.main_layers[-1],
            activation='linear',
            name=self.model_name + "_output_layer_mu",
            kernel_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(self.seed, 2 * (self._n_hidden_layers + 1) + 2)
            ),
            bias_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(self.seed, 3 * (self._n_hidden_layers + 1) + 33)
            ),
            kernel_regularizer=l2(self.l2reg),
            bias_regularizer=l2(self.l2reg))(layer)

        if self.no_noise:
            return mu_output

        sigma_output = Dense(
            self.main_layers[-1],
            activation=self._softplus_wrapper(self.softplus_min_var),
            name=self.model_name + "_output_layer_sigma",
            kernel_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(self.seed, 2 * (self._n_hidden_layers + 2) + 2)
            ),
            bias_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(self.seed, 3 * (self._n_hidden_layers + 2) + 3)
            ),
            kernel_regularizer=l2(self.l2reg),
            bias_regularizer=l2(self.l2reg)
        )(layer)

        x_output = concatenate([mu_output, sigma_output])
        return x_output

    def _create_model(self) -> 'Model':
        """responsible for creation of the network
        creates input layer according to set parameters
        :return:
        """
        # input layer
        x_input = Input(shape=(self.main_layers[0],), name=self.model_name + "_input_layer")
        x_output = self._create_main_architecture(x_input)
        self.model = Model(inputs=[x_input], outputs=x_output)
        # self.model.summary()
        return self.model

    def create_copy(self, base_seed, regularizer_factor=1.0):
        """creating a copy of a Network model
        :param base_seed: seed to copy
        :param regularizer_factor: regularizer factor to use
        :return:
        """
        return DeepEnsembleModel(
            layers=self.layers,
            activation=self.activation,
            RSN=self.RSN,
            s=self.s,
            seed=self._update_seed(self.seed, base_seed * (2 * (self._n_hidden_layers + 2) + 3 + 1)),
            l2reg=regularizer_factor*self.l2reg,
            loss=self.loss,
            softplus_min_var=self.softplus_min_var,
            model_name=self.model_name + "_{}".format(base_seed),
            no_noise=self.no_noise
        )

    def create_copy_advanced(self, seed_counter, **kwargs):
        """creating a copy of a Network model
        :param seed_counter: seed to copy
        :param dropout_prob: regularizer factor to use
        :param l2reg: regularizer factor to use
        :return:
        """
        options = {
            "dropout_prob": self.dropout_prob,
            "l2reg": self.l2reg,
            "model_name": self.model_name + "_{}".format(seed_counter),
        }
        options.update(kwargs)
        return DeepEnsembleModel(
            layers=self.layers,
            activation=self.activation,
            RSN=self.RSN,
            s=self.s,
            seed=self._update_seed(self.seed, seed_counter * (2 * (self._n_hidden_layers + 2) + 3 + 1)),
            l2reg=options["l2reg"],
            loss=self.loss,
            softplus_min_var=self.softplus_min_var,
            model_name=options["model_name"],
            no_noise=self.no_noise,
            dropout_prob=options["dropout_prob"]
        )

    def fit(self, x: np.array, y: np.array, *args, **kwargs) -> Model:
        """fit the network model
        :param x: sample input
        :param y: sample output
        :param args: additional arguments for the fitting process to be passed to the Keras fit function
        :param kwargs: additional arguments for the fitting process to be passed to the Keras fit function
        :return: fitted Model
        """
        return super().fit(x, y, *args, **kwargs)

    def predict(self, x_values) -> Tuple[np.array, np.array]:
        """Wrapper method for the prediction method of the model.
        Adds the additional input which flags the input ad real data point.

        :param x_values:
        :return:
        """
        return self.model.predict(x_values)

    def _softplus_wrapper(self, min_var=1e-06):
        """
        Softplus wrapper with added minimum variance for numerical stability, used for deep ensembles
        """

        def custom_softplus(x):
            return K.log(1 + K.exp(x)) + min_var

        return custom_softplus

    def gaussian_nll(self, ytrue, ypred):  # Attention keras does mean automatically in loss function.

        """Adapted and CHANGED from https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e

        Keras implmementation of univariate Gaussian negative loglikelihood loss function.

        Parameters
        ----------
        ytrue: tf.tensor of shape [n_samples, 1]
            ground truth values
        ypreds: tf.tensor of shape [n_samples, 2]
            predicted mu and variance values (e.g. by your neural network)
        Returns
        -------
        neg_log_likelihood: float
            negative loglikelihood summed over samples

        This loss can then be used as a target loss for any keras model, e.g.:
            model.compile(loss=gaussian_nll, optimizer='Adam')

        """

        mean = ypred[:, 0:1]
        variance = ypred[:, 1:]

        mse = 0.5 * K.square(ytrue - mean) / variance
        sigma_trace = 0.5 * K.log(variance)
        constant = 0.5 * np.log(2 * np.pi)

        log_likelihood = mse + sigma_trace + constant  # keep the constant to be able to compare the value

        return log_likelihood  # Attention mean is returned (not sum)

    def squared_loss(self, y_true, y_pred) -> Loss:
        """sqared loss wrapper splitting y_pred into mean and variance
        :param y_true: true output values
        :param y_pred: predicted output values
        :return: loss
        """
        mean = y_pred[:, 0:1]
        variance = y_pred[:, 1:]
        return K.square(y_true - mean) + variance

    def mse_no_noise(self, y_true, y_pred) -> 'Loss':
        """mse loss wrapper making sure no_noise parameter is set
        :param y_true: true output values
        :param y_pred: predicted output values
        :return: loss
        """
        assert self.no_noise, "squared_loss_no_noise loss is only allowed for no_noise=True"
        mse = MeanSquaredError()
        return mse(y_true, y_pred)

    def use_mse_no_noise(self) -> NoReturn:
        """setting no_noise parameter to ensure noiseless case
        :return:
        """
        self.loss = self.mse_no_noise

    @staticmethod
    def read_from_config(config) -> 'DeepEnsembleModel':
        """reads the configuration from a config object (from config file)
        and created DeepEnsembleModel accordingly

        :param config: configuration object
        :return: DeepEnsembleModel instance
        """
        layers = [int(i) for i in config.as_list("layers")]
        deep_ensemble_model = DeepEnsembleModel(
            layers=[layers, layers],
            loss=None,
            seed=config_int_or_none(config["seed"]),
            l2reg=config.as_float("l2reg"),
            activation=config["activation"],
            RSN=config.as_bool("RSN"),
            softplus_min_var=config.as_float("softplus_min_var"),
            model_name="DeepEnsemble",
            s=config.as_float("s"),
            no_noise=config.as_bool("no_noise"),
        )
        deep_ensemble_model.use_mse_no_noise()
        return deep_ensemble_model

