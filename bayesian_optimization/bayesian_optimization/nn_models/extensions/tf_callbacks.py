from tensorflow.keras.callbacks import Callback
import warnings
import numpy as np


class ReturnBestWeights(Callback):
    __author__ = 'anonymous'
    __copyright__ = 'Copyright 2020, Pseudo Uncertainty Bounds for Neural Networks'
    __license__ = 'AGPL-3.0'
    __version__ = '0.1.0'
    __maintainer__ = 'anonymous'
    __email__ = 'anonymous'
    __status__ = 'Dev'
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

    def __init__(self,
                 monitor='loss',
                 verbose=0,
                 mode='auto',
                 baseline=None
                 ):

        super(ReturnBestWeights, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.verbose = verbose
        self.wait = 0
        self.best_epoch = 0
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
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
            print('\nRestoring model weights from the end of the best epoch: epoch %d' %(self.best_epoch))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value
