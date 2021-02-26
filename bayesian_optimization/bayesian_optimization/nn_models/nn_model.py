#from __future__ import annotations

from datetime import datetime
# Type hints
from typing import *
from typing import NoReturn

# Libs
from tensorflow.keras.models import Model, model_from_json
from bayesian_optimization.acq_optimizer.gridsearch import GridSearch
from bayesian_optimization.acq_optimizer.direct_optimizer import DirectOptimizer
from bayesian_optimization.acq_optimizer.nn_mip import NNMIP
import json



class NNModel:
    """Basis class for neural netork models"""

    SUPPORTED_ACQ_OPTIMIZER = {
        "grid_search": GridSearch,
        "direct": DirectOptimizer,
        "mip": NNMIP
    }

    def __init__(
            self,
            layers: List,
            activation: str,
            RSN: bool,
            s: float,
            seed: int,
            l2reg: float,
            loss: Union[Callable, List[Callable]]
    ):
        # copy main to side layer if needed
        self.main_layers = layers[0]
        if len(layers) == 1:
            self.side_layers = self.main_layers
        else:
            self.side_layers = layers[1]
        self.layers = layers
        self.activation = activation
        self.RSN = RSN
        self.s = s
        self.seed = seed
        self.l2reg = l2reg
        self.loss = loss
        self.flag = None
        self.model = None
        self._create_model()
        self.context = None

    def set_context(self, context):
        self.context = context

    @classmethod
    def create_from_context(cls, context: 'Context') -> 'NNModel':
        """Creates new instance of the Neural Network model in the given context.
        Creates the same class of NNModels and parametrizes it with the same parameter values.

        :param context: context from which a 'copy' of the NNmodel should be created.
        :return: new NNModel instance
        """
        return context.nn_model.create_from_context(context)

    def _create_model(self) -> Model:
        """responsible for creation of the network
        creates input layer according to set parameters
        :return:
        """
        raise NotImplementedError("Please Implement this method")

    def save(self, path: str) -> NoReturn:
        """saves the network model to a file
        :param path: path where a model should be saved
        :return:
        """
        model_json = self.model.to_json()
        with open("{}/{}.json".format(path, self.model.name), "w") as json_file:
            json_file.write(model_json)
        meta = {"seed": self.seed}
        with open("{}/{}_meta.json".format(path, self.model.name), "w") as meta_file:
            json.dump(meta, meta_file)
        # serialize weights to HDF5
        self.model.save_weights("{}/{}.h5".format(path, self.model.name))
        print("Saved model to disk")

    def load(self, path: str) -> 'NNModel':
        """load a network model from a file
        :param path: path where a model should loaded from
        :return: loaded Model
        """
        print("{}.json".format(path))
        json_file = open("{}.json".format(path), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("{}.h5".format(path))
        self.model = loaded_model
        with open("{}_meta.json".format(path), "r") as meta_file:
            meta = json.load(meta_file)
        self.seed = meta["seed"]
        print("Loaded model from disk")
        return self

    def compile(self) -> NoReturn:
        """compile the model
        :return:
        """
        self.model.compile(optimizer=self.context.model_optimizer, loss=self.loss, experimental_run_tf_function=False)

    def fit(self, x, y, *args, **kwargs) -> Model:
        """fit the given samples (x,y) with the underlying Keras Model.
        This is only a wrapper to the fit-method of the Keras model
        :param x: input values of the samples
        :param y: target values of the samples
        :param args: additional parameters directly passed to the Keras-Model fit method
        :param kwargs: additional parameters directly passed to the Keras-Model fit method
        :return: return the Keras model (now 'trained')
        """
        self.context.reset_model(self)
        self.context.reset_model_optimizer()
        self.model.compile(optimizer=self.context.model_optimizer, loss=self.loss, experimental_run_tf_function=False)
        start_time = datetime.now()
        history = self.model.fit(x=x, y=y, *args, **kwargs)
        time_elapsed = datetime.now() - start_time
        self._inspect(history, time_elapsed)
        return self.model

    @staticmethod
    def _update_seed(seed: Union[int, None], add: int) -> Union[int, None]:
        """update the given seed. Adds the given value specified under 'add' to the given seed
        If the given seen is None this method will return None as well
        :param seed: input see
        :param add: value to be added to the seed
        :return: new seed
        """
        return [None if seed is None else seed + add][0]

    def predict(self, x_values):
        raise NotImplementedError("Please Implement this method")

    def _inspect(self, history, time_elapsed) -> NoReturn:
        """allows for writing inspection data into the inspector in the context.

        :param history: history of the model fit
        :param time_elapsed: time needed for the model fitting
        """
        if self.context.inspector and self.context.inspector.inspect_nn_model:
            inspection_data = {
                "model": self.__class__.__name__,
                "history": history,
                "time_elapsed": time_elapsed,
            }
            self.context.inspector.add_estimation(inspection_data)
