"""
Some code recycled and adopted with permission from https://gitlab.ethz.ch/jakobwe/Pseudo_Uncertainty_Bounds_for_NNs
"""

from typing import *

import numpy as np
from bayesian_optimization.nn_models.nn_model import NNModel
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from bayesian_optimization.utils.utils import config_int_or_none


class DropoutModel(NNModel):
    """Implementation of the MC Dropout Model as a class
    """

    def __init__(
        self,
        layers: List,
        activation: str,
        RSN: bool,
        seed: int,
        l2reg: float,
        loss,
        dropout: float,
        s: float = 0.05
    ):
        self.dropout = dropout
        super().__init__(layers, activation, RSN, s, seed, l2reg, loss)

    def _create_main_architecture(self, x_input: 'Input') -> 'Model':
        """creates the main architecture of the metwork according
        :param x_input: input layer to the network
        :return: constructed network model
        """
        # first hidden layer
        layer = Dense(
            self.main_layers[1],
            activation=self.activation,
            name="hidden_layer_{}".format(1),
            kernel_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self.seed
            ),
            bias_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(self.seed, 1)
            ),
            kernel_regularizer=l2(self.l2reg),
            bias_regularizer=l2(self.l2reg)
        )(x_input)

        if self.dropout != 0:
            layer = Dropout(self.dropout)(layer, training=True)
        # hidden layers

        for i, n in enumerate(self.main_layers[2: -1]):
            layer = Dense(
                n,
                activation=self.activation,
                name="hidden_layer_{}".format(i + 2),
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
                bias_regularizer=l2(self.l2reg)
            )(layer)

            if self.dropout != 0:
                layer = Dropout(self.dropout)(layer, training=True)

        # output layer
        x_output = Dense(
            self.main_layers[-1],
            activation='linear',
            name="output_layer",
            kernel_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(self.seed, -1)
            ),
            bias_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(self.seed, -2)
            ),
            kernel_regularizer=l2(self.l2reg),
            bias_regularizer=l2(self.l2reg)
        )(layer)

        return x_output

    def _create_model(self) -> 'Model':
        """responsible for creation of the network
        creates input layer according to set parameters
        :return:
        """
        # input layer
        x_input = Input(shape=(self.main_layers[0],), name="input_layer")
        x_output = self._create_main_architecture(x_input)
        self.model = Model(inputs=[x_input], outputs=x_output)
        #self.model.summary()
        return self.model

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
        :param x_values:
        :return:
        """
        res = self.model.predict(x_values)

        return res

    @staticmethod
    def read_from_config(config) -> 'DropoutModel':
        """reads the configuration from a config object (from config file)
        and created DeepEnsembleModel accordingly

        :param config: configuration object
        :return: DeepEnsembleModel instance
        """
        layers = [int(i) for i in config.as_list("layers")]
        return DropoutModel(
            layers=[layers],
            activation=config["activation"],
            RSN=config.as_bool("RSN"),
            seed=config_int_or_none(config["seed"]),
            dropout=config.as_float("dropout"),
            loss=config["loss"],
            l2reg=config.as_float("l2reg"),
        )


