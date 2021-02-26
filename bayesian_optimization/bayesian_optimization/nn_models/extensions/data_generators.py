import numpy as np
from tensorflow.keras.utils import Sequence


class AddAugmentedDataGenerator(Sequence):
    def __init__(
            self,
            samples_x,
            samples_y,
            batch_size,
            lower_bound_x,
            upper_bound_x,
            n_aug,
            c_aug,
            shuffle=True
    ):
        self.samples_x = samples_x
        self.samples_y = samples_y
        assert len(self.samples_x) == len(self.samples_y), "samples_x and samples_y must have same dimension"
        self.n_train = len(self.samples_x)
        self.indexes = np.arange(self.n_train)
        self.lower_bound_x = lower_bound_x
        self.upper_bound_x = upper_bound_x
        assert len(self.lower_bound_x) == len(self.upper_bound_x), "lower and upper must have same dimensions"
        self.input_dim = len(self.lower_bound_x)
        self.batch_size = batch_size
        self.n_aug = n_aug
        self.c_aug = c_aug
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(self.n_train / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        x_aug_input = np.random.uniform(low=self.lower_bound_x, high=self.upper_bound_x, size=(self.n_aug, self.input_dim))
        x_aug_flag = np.ones((self.n_aug, 1))
        x_aug = np.concatenate((x_aug_input, x_aug_flag), axis=-1)

        x_train_input = self.samples_x[indexes]
        x_train_flag = np.zeros((self.n_train, 1))
        x_train = np.concatenate((x_train_input, x_train_flag), axis=-1)
        x = np.concatenate((x_train, x_aug))

        y_aug_value = np.zeros((self.n_aug, 1))  # dummy value, not relevant
        y_aug_sigma = self.c_aug * np.ones((self.n_aug, 1))
        y_aug = np.concatenate((y_aug_value, y_aug_sigma), axis=-1)

        y_train_value = self.samples_y[indexes]
        y_train_sigma = np.zeros((self.n_train, 1))
        y_train = np.concatenate((y_train_value, y_train_sigma), axis=-1)

        y = np.concatenate((y_train, y_aug))

        return [x[:, :-1], x[:, -1:]], [y[:, :-1], y[:, -1:]]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)




























class CallBackDataGeneratorStaticAugmented(Sequence):
    def __init__(
            self,
            batch_size,
            lower_bound_x,
            upper_bound_x,
            n_train,
            n_aug,
            c_aug,
            callback,
            shuffle=True
    ):
        self.batch_size = batch_size
        self.lower_bound_x = lower_bound_x
        self.upper_bound_x = upper_bound_x
        self.n_train = n_train
        self.n_aug = n_aug
        self.c_aug = c_aug
        self.callback = callback
        assert len(self.lower_bound_x) == len(self.upper_bound_x), "lower and upper must have same dimensions"
        self.input_dim = len(self.lower_bound_x)
        self.shuffle = shuffle
        self.indexes = np.arange(n_train)
        self.samples_x = None
        self.samples_y = None
        self.aug_x = None
        self.aug_y = None
        self.aug_target = None
        self._sample_data()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.n_train / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        inputs = np.concatenate((self.samples_x[indexes][:, 0], self.aug_x[:, 0]))
        flags = np.concatenate((np.zeros(len(indexes)), np.zeros(self.n_aug)))
        targets = np.concatenate((self.samples_y[indexes], self.aug_target))
        # print("---------", [inputs, flags], [targets, self.c_aug * np.ones((self.n_aug+self.n_train,))])

        samples_x = np.random.uniform(
            low=self.lower_bound_x,
            high=self.upper_bound_x,
            size=(self.n_train, self.input_dim)
        )
        x_aug = np.random.uniform(low=self.lower_bound_x, high=self.upper_bound_x, size=(self.n_aug, self.input_dim))
        samples_y = np.array([self.callback(x) for x in self.samples_x])
        y_aug = self.c_aug * np.ones((self.n_aug, 1))

        x_train = np.concatenate((samples_x[indexes], np.zeros((self.n_train, 1))), axis=-1)
        x_aug = np.concatenate((x_aug, np.ones((self.n_aug, 1))), axis=-1)
        x = np.concatenate((x_train, x_aug))

        y = np.concatenate((np.reshape(samples_y[indexes], (self.n_train, 1)), y_aug))
        print("x_train",x)

        print("x_train",[x[:, :-1], x[:, -1:]])

        return [x[:, :-1], x[:, -1:]], [y, y]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _sample_data(self):
        self.samples_x = np.random.uniform(
            low=self.lower_bound_x,
            high=self.upper_bound_x,
            size=(self.n_train, self.input_dim)
        )
        print("ww", self.samples_x)
        self.samples_y = np.array([self.callback(x) for x in self.samples_x])

        resolution = int(self.n_aug**(1/self.input_dim))
        x_grid = np.meshgrid(*[np.linspace(lower, upper, resolution) for lower, upper in zip(self.lower_bound_x, self.upper_bound_x)])
        self.aug_x = np.concatenate(
            [np.expand_dims(x, axis=-1) for x in x_grid],
            axis=-1).reshape((resolution**self.input_dim, self.input_dim))
        self.aug_target = np.array([self.callback(x) for x in self.aug_x])


class DataGenerator(Sequence):
    def __init__(self, batch_size, x, y, n_train, n_aug, MCaug=False,
                 x_min_aug=-1 - 0.1, x_max_aug=1 + 0.1,
                 r_loss=False, shuffle=True):
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.n_train = n_train
        self.n_aug = n_aug
        self.shuffle = shuffle
        self.indexes = np.arange(self.n_train)
        self.on_epoch_end()  # shuffle when instantiating
        self.r_loss = r_loss
        self.MCaug = MCaug  # sample new augmented data points with every batch?
        self.x_min_aug = x_min_aug  # set range to sample augmented data from
        self.x_max_aug = x_max_aug
        self.din = x.shape[1] - 1  # input dimension (-1 because we get flagged x input)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):  # number of batches per epoch
        return int(np.floor(self.n_train / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        # batch indices of training batch and all augmented indices
        indexes_aug = np.concatenate((indexes,
                                      np.arange(self.n_train,
                                                self.n_train + self.n_aug)))  # take in every batch all artifical training points

        if self.MCaug:
            # create new augmented random x-data
            x_aug = np.random.uniform(low=self.x_min_aug,
                                      high=self.x_max_aug,
                                      size=(self.n_aug, self.din))
            x_aug = np.concatenate((x_aug, np.ones((self.n_aug, 1))), axis=-1)
            x = np.concatenate((self.x[indexes, :], x_aug))

        else:
            x = self.x[indexes_aug, :]

        y = self.y[
            indexes_aug]  # select targets. (augmented targets are constant and can hence be taken from initial augmented set)

        # version for y/r architecture:
        if self.r_loss:
            return [x[:, :-1], x[:, -1]], [y, y]

        return [x[:, :-1], x[:, -1]], y
