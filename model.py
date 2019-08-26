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
    def __init__(self, file):
        self.logfilepath = file

    def add_player_info(self, p1, p2):
        self.player1_class = p1.player_class
        self.player2_class = p2.player_class

        self.player1_name = p1.name
        self.player2_name = p2.name

        try:
            self.model1_name = p1.model.model_name
            self.model1_class = p1.model.model_class
            self.model1_timestamp = p1.model.timestamp
        except AttributeError:
            self.model1_name = 'n/a'
            self.model1_class = 'n/a'
            self.model1_timestamp = 'n/a'

        try:
            self.model2_name = p2.model.model_name
            self.model2_class = p2.model.model_class
            self.model2_timestamp = p2.model.timestamp
        except AttributeError:
            self.model2_name = 'n/a'
            self.model2_class = 'n/a'
            self.model2_timestamp = 'n/a'

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

    def write_parameters_to_file(self):
        f = open(self.logfilepath, "a+")

        f.write(f"=================================================\n\n")
        f.write(f"loaded models at = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # player/model info
        f.write(f"player1\n")
        f.write(f" class = {self.player1_class}\n")
        f.write(f" name = {self.player1_name}\n")
        f.write(f" model class = {self.model1_class}\n")
        f.write(f" model name = {self.model1_name}-{self.model1_timestamp}\n")

        f.write(f"player2\n")
        f.write(f" class = {self.player2_class}\n")
        f.write(f" name = {self.player2_name}\n")
        f.write(f" model class = {self.model2_class}\n")
        f.write(f" model name = {self.model2_name}-{self.model2_timestamp}\n\n")

        # constantsinfo
        for key, constant in self.constants.items():
            f.write(f"{key} = {constant}\n")
        f.write("\n")
        f.close()

    def log_text_to_file(self, text):
        f = open(self.logfilepath, "a+")
        f.write(f"{text}\n")
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
        path = path[7:]  # removes subdir (models/)
        old_modelname = path.split('_')
        self.model_name = f'PreLoadedModel_{old_modelname[0]}_{old_modelname[1]}_{old_modelname[2]}_{old_modelname[3]}'
        self.timestamp = int(time.time())


class model_base:
    """
    Base class for all models
    """
    def __init__(self, input_shape, output_num):
        self.model = self.create_model(input_shape, output_num)
        self.target_model = self.create_model(input_shape, output_num)
        self.model.model_name = None
        self.model.timestamp = int(time.time())
        self.model.model_class = self.__class__.__name__

    def create_model(self, input_shape, output_num):
        pass


class model_1(model_base):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
        #self.model = self.create_model(input_shape, output_num)
        #self.target_model = self.create_model(input_shape, output_num)
        self.model.model_name = 'model1_conv2x128'
        #self.model.timestamp = int(time.time())
        #self.model.model_class = self.__class__.__name__

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


class model_2(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.model = self.create_model(input_shape, output_num)
        #self.target_model = self.create_model(input_shape, output_num)
        self.model.model_name = 'model2_dense1x64'
        #self.model.timestamp = int(time.time())
        #self.model.model_class = self.__class__.__name__

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Dense(64, input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        # model.add(Dense(64))

        model.add(Dense(output_num, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model_3(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.model = self.create_model(input_shape, output_num)
        #self.target_model = self.create_model(input_shape, output_num)
        self.model.model_name = 'model3_dense2x64'
        #self.model.timestamp = int(time.time())
        #self.model.model_class = self.__class__.__name__

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


class model_4(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.model = self.create_model(input_shape, output_num)
        #self.target_model = self.create_model(input_shape, output_num)
        self.model.model_name = 'model4_dense2x128(softmax)'
        #self.model.timestamp = int(time.time())
        #self.model.model_class = self.__class__.__name__

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Dense(128, input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Dense(128, input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        # model.add(Dense(64))
        #model.add(Dense(128, input_shape=input_shape))
        #model.add(Activation('relu'))

        model.add(Dense(output_num, activation='softmax'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
