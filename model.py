from keras.callbacks import TensorBoard
import tensorflow as tf

from keras.models import Sequential, load_model  # , clone_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam

from constants import *
from game import FiarGame
import time

from datetime import datetime


class ModelLog():
    def __init__(self):
        pass

    def add_model_info(self, model):
        self.model_name = model.model_name
        self.model_starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.model_timestamp = model.timestamp

    def add_constants(self):
        self.constants = {
            "MIN_REWARD": MIN_REWARD,
            "EPISODES": EPISODES,
            "REWARD_WINNING": FiarGame.REWARD_WINNING,
            "REWARD_LOSING": FiarGame.REWARD_LOSING,
            "REWARD_TIE": FiarGame.REWARD_TIE,
            "REWARD_INVALID_MOVE": FiarGame.REWARD_INVALID_MOVE,
            "DISCOUNT": DISCOUNT,
            "REPLAY_MEMORY_SIZE": REPLAY_MEMORY_SIZE,
            "MIN_REPLAY_MEMORY_SIZE": MIN_REPLAY_MEMORY_SIZE,
            "MINIBATCH_SIZE": MINIBATCH_SIZE,
            "UPDATE_TARGET_EVERY": UPDATE_TARGET_EVERY,
            "MEMORY_FRACTION": MEMORY_FRACTION,
            "epsilon": epsilon,
            "EPSILON_DECAY": EPSILON_DECAY,
            "MIN_EPSILON": MIN_EPSILON,
            "AGGREGATE_STATS_EVERY": AGGREGATE_STATS_EVERY,
            "SHOW_PREVIEW": SHOW_PREVIEW
        }
        # "LOAD_MODEL": LOAD_MODEL

    def write_to_file(self, path):
        f = open(path, "a+")
        f.write(f"{self.model_starttime} \nmodel name = {self.model_name}-{self.model_timestamp}\n")
        for key, constant in self.constants.items():
            f.write(f"{key} = {constant}\n")
        f.write("\n")
        f.close()


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class load_a_model:
    def __init__(self, path):
        self.model = load_model(path)
        self.target_model = load_model(path)
        old_modelname = path.split('_')
        self.model_name = f'PreLoadedModel_{old_modelname[0]}_{old_modelname[1]}'
        self.timestamp = int(time.time())


class model_1:
    def __init__(self, input_shape, output_num):
        self.model = self.create_model(input_shape, output_num)
        self.target_model = self.create_model(input_shape, output_num)
        self.model_name = 'model1_conv2x128'
        self.timestamp = int(time.time())

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(128, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same'))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(output_num, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model_2:
    def __init__(self, input_shape, output_num):
        self.model = self.create_model(input_shape, output_num)
        self.target_model = self.create_model(input_shape, output_num)
        self.model_name = 'model2_dense1x64'
        self.timestamp = int(time.time())

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Dense(64, input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        # model.add(Dense(64))

        model.add(Dense(output_num, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model_3:
    def __init__(self, input_shape, output_num):
        self.model = self.create_model(input_shape, output_num)
        self.target_model = self.create_model(input_shape, output_num)
        self.model_name = 'model3_dense2x64'
        self.timestamp = int(time.time())

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Dense(64, input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Dense(64, input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        # model.add(Dense(64))

        model.add(Dense(output_num, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model