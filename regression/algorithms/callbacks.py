# -*- coding: utf-8 -*-
"""

This file contains callbacks training networks for uncertainty bounds.

"""

# Libs
from tensorflow.keras.callbacks import Callback
import warnings
import numpy as np
from typing import NoReturn, Any, Dict, Optional

# %%
# customized callbacks shown if verbose==2
class PredictionHistory(Callback):

    """
    Customized Callback Class for printing y&r predictions during training of NOMU.

    ...

    Attributes
    ----------
    generator:
        Instance of DataGenerator

    Methods
    -------
    on_epoch_begin()
        Prints y and r prediction at the beginning of each epoch for the current data defined by self.generator.


    """

    def __init__(self, generator: Any) -> NoReturn:

        """Constructor of the class PredictionHistory.

        Arguments
        ----------
        generator :
            Instance of DataGenerator. Defines current batch of training data.

        """

        super(Callback, self).__init__()
        self.generator = generator

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> NoReturn:

        """Prints y and r prediction at the beginning of each epoch.

        Arguments
        ----------
        epoch :
            Epoch number.
        logs :
            The logs dict contains the loss value, and all the metrics at the end of a batch or epoch (default in tensorflow.keras).
        """

        if self.params["verbose"] == 2:
            y_pred, r_pred = self.model.predict(self.generator)
            print(
                "\nPrediction at epoch: {}: \n\n y_pred:\n{}\n\n r_pred:\n{}\n\n ".format(
                    epoch, y_pred.flatten(), r_pred.flatten()
                )
            )


# %% customized callbacks shown if verbose==2
class PredictionHistory_DE(Callback):

    """
    Customized Callback Class for printing y&r predictions during training of
    Deep Ensembles.

    ...

    Attributes
    ----------
    x_true :
        Training input points.
    y_true :
        Training output points.

    Methods
    -------
    on_epoch_end()
        Prints at the end of each 100s epoch the loss.
    on_epoch_begin()
        Prints at the beginning of each 100s epoch the prediction.

    """

    def __init__(self, x_true: np.array, y_true: np.array) -> NoReturn:

        super(Callback, self).__init__()
        self.x_true = x_true
        self.y_true = y_true

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> NoReturn:

        """Prints at the end of each 100s epoch the loss of the deep ensemble.

        Arguments
        ----------
        epoch :
            Epoch number.
        logs :
            The logs dict contains the loss value, and all the metrics at the end of a batch or epoch (default in tensorflow.keras).
        """

        if self.params["verbose"] == 0:
            if epoch % 100 == 0:
                print("Loss for epoch {} is {}".format(epoch, logs["loss"]))

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> NoReturn:

        """Prints at the beginning of each 100s epoch the prediction of the deep ensemble.

        Arguments
        ----------
        epoch :
            Epoch number.
        logs :
            The logs dict contains the loss value, and all the metrics at the end of a batch or epoch (default in tensorflow.keras).
        """

        if self.params["verbose"] == 2:
            if epoch % 100 == 0:
                pred = self.model.predict(self.x_true)
                mu_pred = pred[:, 0]
                var_pred = pred[:, 1]
                print(
                    "\nPrediction at epoch: {}: \n\n mu_pred:\n{}\n\n var_pred:\n{}\n\n ".format(
                        epoch, mu_pred, var_pred
                    )
                )


# %%
class ReturnBestWeights(Callback):
    """Returns the best weights during the whole training process.

    Arguments
    ----------
    monitor :
        quantity to be monitored.
    verbose :
        verbosity mode.
    mode :
        one of {auto, min, max}. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `max`
        mode it will stop when the quantity
        monitored has stopped increasing; in `auto`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline :
        Baseline value for the monitored quantity to reach.
        Training will stop if the model doesn't show improvement
        over the baseline.
    """

    def __init__(self, monitor="loss", verbose=0, mode="auto", baseline=None):

        super(ReturnBestWeights, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.verbose = verbose
        self.wait = 0
        self.best_epoch = 0
        self.best_weights = None

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                "EarlyStopping mode %s is unknown, " "fallback to auto mode." % mode,
                RuntimeWarning,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self, logs=None):
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        # update best weights during training
        if self.monitor_op(current, self.best):
            self.best = current
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        # set best weights always at the end of training
        self.model.set_weights(self.best_weights)
        if self.verbose > 0:
            print(
                "\nRestoring model weights from the end of the best epoch: epoch %d"
                % (self.best_epoch)
            )

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s"
                % (self.monitor, ",".join(list(logs.keys()))),
                RuntimeWarning,
            )
        return monitor_value
