"""
Some code recycled and adopted with permission from https://gitlab.ethz.ch/jakobwe/Pseudo_Uncertainty_Bounds_for_NNs
"""
# Libs
from datetime import datetime
# type-hinting
from typing import *

import numpy as np
from bayesian_optimization.context.context import Context
from bayesian_optimization.utils.utils import config_int_or_none
from bayesian_optimization.nn_models.extensions.tf_callbacks import ReturnBestWeights

# Internal
from bayesian_optimization.losses.losses import RLossWrapper, SquaredLossWrapper
from bayesian_optimization.nn_models.extensions.data_generators import AddAugmentedDataGenerator
from bayesian_optimization.nn_models.nn_model import NNModel
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class NOMUModel(NNModel):
    """Class defining the model structure of the 'OurModel'
    """

    def __init__(
            self,
            layers: List,
            lowerbound_x_aug: np.array,
            upperbound_x_aug: np.array,
            n_aug: int,
            c_aug: float,
            mu_sqr: float,
            mu_abs: float,
            mu_exp: float,
            c_exp: float,
            c_2: float,
            loss: Union[Callable, List[Callable], RLossWrapper, None] = None,
            seed: int = None,
            l2reg: float = 0,
            activation: str = "relu",
            RSN: bool = False,
            s: float = 0.05,
            best_weight_callback: bool = True
    ):
        """constructor

        :param layers:  List of List of integer displaying layer structure (dimension) must include two lists.
                        One for the main model and one for the side-model. Each list must include a input layer,
                        an output layer and at least one hidden layer. If only one list in defined then it's structure
                        will be applied to the main and the side model.
                        An example input is  [[1, 2**10, 2**10, 2**10, 1]] and results in:
                            Input layer with shape (1,)
                            Hidden layer with shape (1024,)
                            Hidden layer with shape (1024,)
                            Hidden layer with shape (1024,)
                            Output Layer with shape (1,)
                        An the main and the side model have both this same structure.
        :param loss:
        :param seed:
        :param l2reg:
        :param activation:
        :param RSN:
        :param s:
        :param best_weight_callback:
        """
        self.lowerbound_x_aug = lowerbound_x_aug
        self.upperbound_x_aug = upperbound_x_aug
        self.n_aug = n_aug
        self.c_aug = c_aug
        self.mu_sqr = mu_sqr
        self.mu_abs = mu_abs
        self.mu_exp = mu_exp
        self.c_exp = c_exp
        self.c_2 = c_2
        self.squared_loss_wrapper = None
        self.r_loss_wrapper = None
        self.best_weight_callback = best_weight_callback
        super().__init__(layers, activation, RSN, s, seed, l2reg, loss)

    def get_json(self) -> dict:
        """get a dict with the parameter configuration set for the model
        :return: Json
        """
        return {
            "lb_x_aug": self.lowerbound_x_aug,
            "ub_x_aug": self.upperbound_x_aug,
            "n_aug": self.n_aug,
            "c_aug": self.c_aug,
            "mu_sqr": self.mu_sqr,
            "mu_abs": self.mu_abs,
            "mu_exp": self.mu_exp,
            "c_exp": self.c_exp,
            "c_2": self.c_2,
            "squared_loss_wrapper": self.squared_loss_wrapper,
            "layers": self.layers,
            "activation": self.activation,
            "RSN": self.RSN,
            "s": self.s,
            "seed": self.seed,
            "l2reg": self.l2reg,
        }

    @classmethod
    def create_from_context(cls, context: Context) -> 'NOMUModel':
        """create neu instance according to configuration of context file
        :param context: context of BO-run
        :return: instance of Nomu model
        """
        assert context.nn_model.__class__.__name__ == cls.__class__.__name__, \
            "trying to create instance of " + str(cls.__class__.__name__) + \
            " but context has Model of class " + str(context.nn_model.__class__.__name__)
        return cls(
            layers=context.nn_model.layers,
            lowerbound_x_aug=context.nn_model.lowerbound_x_aug,
            upperbound_x_aug=context.nn_model.upperbound_x_aug,
            n_aug=context.nn_model.n_aug,
            c_aug=context.nn_model.c_aug,
            mu_sqr=context.nn_model.mu_sqr,
            mu_abs=context.nn_model.mu_abs,
            mu_exp=context.nn_model.mu_exp,
            c_exp=context.nn_model.c_exp,
            c_2=context.nn_model.c_2,
            loss=context.nn_model.loss,
            seed=context.nn_model.seed,
            l2reg=context.nn_model.l2reg,
            activation=context.nn_model.activation,
            RSN=context.nn_model.RSN,
            s=context.nn_model.s,
        )

    def _create_main_architecture(self, x_input: 'Input') -> 'Model':
        """creates the model structure for the main met which is responsible for the learning and prediction of
        the y value.

        :param x_input: Input layer
        :return: constructed network model of the main net
        """
        # first hidden layer
        y = Dense(
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
            bias_regularizer=l2(self.l2reg),
            trainable=not self.RSN
        )(x_input)

        # hidden layers
        for i, n in enumerate(self.main_layers[2: -1]):
            y = Dense(
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
                bias_regularizer=l2(self.l2reg),
                trainable=not self.RSN
            )(y)

        # output layer
        y_output = Dense(
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
        )(y)

        return y, y_output

    def _create_side_architecture(self, x_input: 'Input', y: 'Model') -> 'Model':
        """creates the model structure for the side net which is responsible for the learning and prediction of
        the r value.

        :param x_input: input layer
        :param y: partial model from main net
        :return: network model of the side net
        """
        # set new seed
        seed = self._update_seed(self.seed, 2 ** 6)
        # first hidden layer
        r = Dense(
            self.side_layers[1],
            activation=self.activation,
            name="r_hidden_layer_{}".format(1),
            kernel_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=seed
            ),
            bias_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(seed, 1)
            ),
            kernel_regularizer=l2(self.l2reg),
            bias_regularizer=l2(self.l2reg),
            trainable=not self.RSN
        )(x_input)
        # hidden layers
        for i, n in enumerate(self.side_layers[2: -1]):
            r = Dense(
                n,
                activation=self.activation,
                name="r_hidden_layer_{}".format(i + 2),
                kernel_initializer=RandomUniform(
                    minval=-self.s,
                    maxval=self.s,
                    seed=self._update_seed(seed, 2 * i + 2)
                ),
                bias_initializer=RandomUniform(
                    minval=-self.s,
                    maxval=self.s,
                    seed=self._update_seed(seed, 2 * i + 3)
                ),
                kernel_regularizer=l2(self.l2reg),
                bias_regularizer=l2(self.l2reg),
                trainable=not self.RSN
            )(r)
        # concatenate last hidden of y and r
        y_r_concat = concatenate([y, r])
        # output layer
        r_output = Dense(
            self.side_layers[-1],
            activation='linear',
            name="r_output_layer",
            kernel_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(seed, -1)
            ),
            bias_initializer=RandomUniform(
                minval=-self.s,
                maxval=self.s,
                seed=self._update_seed(seed, -2)
            ),
            kernel_regularizer=l2(self.l2reg),
            bias_regularizer=l2(self.l2reg)
        )(y_r_concat)
        return r_output

    def _define_loss(self, n_train: np.array):
        """ constructs the correct loss
        :param n_train: trainings data
        :return:
        """
        self.squared_loss_wrapper = SquaredLossWrapper(
            n_train=n_train,
            n_aug=self.n_aug)
        self.r_loss_wrapper = RLossWrapper(
            mu_sqr=self.mu_sqr,
            mu_abs=self.mu_abs,
            mu_exp=self.mu_exp,
            c_exp=self.c_exp,
            n_train=n_train,
            n_aug=self.n_aug,
            c_2=self.c_2
        )

    def _create_model(self) -> 'Model':
        """responsible for creation of the network
        creates input layer according to set parameters
        :return:
        """
        # input layer
        flag = Input(shape=(1,), name="flag_input")
        x_input = Input(shape=(self.main_layers[0],), name="input_layer")
        y, y_output = self._create_main_architecture(x_input)
        r_output = self._create_side_architecture(x_input, y)
        self.model = Model(inputs=[x_input, flag], outputs=[y_output, r_output])
        self._add_default_loss()
        #self.model.summary()
        return self.model

    def _add_default_loss(self):
        """assignes default losses in case no specific loss is parametrized
        :return:
        """
        if self.r_loss_wrapper is not None and self.squared_loss_wrapper is not None:
            self.loss = {
                "output_layer": self.squared_loss_wrapper.get_loss(self.model.input[1]),
                "r_output_layer": self.r_loss_wrapper.get_loss(self.model.input[1]),
            }

    def fit(self, x: np.array, y: np.array, *args, **kwargs) -> Model:
        """fit the network model
        :param x: sample input
        :param y: sample output
        :param args: additional arguments for the fitting process to be passed to the Keras fit function
        :param kwargs: additional arguments for the fitting process to be passed to the Keras fit function
        :return: fitted Model
        """
        gen = AddAugmentedDataGenerator(
            samples_x=x,
            samples_y=y,
            batch_size=len(x),
            lower_bound_x=self.lowerbound_x_aug,
            upper_bound_x=self.upperbound_x_aug,
            n_aug=self.n_aug,
            c_aug=self.c_aug
        )
        self._define_loss(len(x))
        self._add_default_loss()
        self.context.reset_model(self)
        self.context.reset_model_optimizer()
        callbacks = []
        if self.best_weight_callback:
            callbacks.append(
                ReturnBestWeights(monitor='loss', verbose=0, mode='min', baseline=None)
            )
        self.model.compile(optimizer=self.context.model_optimizer, loss=self.loss, experimental_run_tf_function=False)
        start_time = datetime.now()
        history = self.model.fit(x=gen, callbacks=callbacks, *args, **kwargs)
        time_elapsed = datetime.now() - start_time
        self._inspect(history, time_elapsed)
        return self.model

    def predict(self, x_values) -> Tuple[np.array, np.array]:
        """Wrapper method for the prediction method of the model.
        Adds the additional input which flags the input ad real data point.

        :param x_values:
        :return:
        """
        model_input = [x_values, np.zeros(len(x_values))]
        model_outputs = self.model.predict(model_input)
        return model_outputs[0], model_outputs[1]

    @classmethod
    def read_from_config(cls, config) -> 'NOMUModel':
        """reads the configuration from a config object (from config file)
        and created DeepEnsembleModel accordingly

        :param config: configuration object
        :return: DeepEnsembleModel instance
        """
        lowerbound_x_aug = [float(i) for i in config.as_list("lowerbound_x_aug")]
        upperbound_x_aug = [float(i) for i in config.as_list("upperbound_x_aug")]
        main_layers = [int(i) for i in config.as_list("main_layers")]
        side_layers = [int(i) for i in config.as_list("side_layers")]
        return NOMUModel(
            layers=[main_layers, side_layers],
            loss=None,
            lowerbound_x_aug=lowerbound_x_aug,
            upperbound_x_aug=upperbound_x_aug,
            n_aug=config.as_int("n_aug"),
            c_aug=config.as_float("c_aug"),
            mu_sqr=config.as_float("mu_sqr"),
            mu_abs=config.as_float("mu_abs"),
            mu_exp=config.as_float("mu_exp"),
            c_exp=config.as_float("c_exp"),
            c_2=config_int_or_none(config["c_2"]),
            seed=config_int_or_none(config["seed"]),
            l2reg=config.as_float("l2reg"),
            activation=config["activation"],
            RSN=config.as_bool("RSN"),
        )
