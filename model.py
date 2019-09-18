from keras.callbacks import TensorBoard
import tensorflow as tf

from keras.models import Model as FuncModel, Sequential, load_model  # , clone_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam, SGD

from constants import *
from game import FiarGame
import time
from datetime import datetime

import csv


class ModelLog():
    def __init__(self, file):
        self.logfilepath = file

    def add_player_info(self, p1, p2):
        self.player1_class = p1.player_class
        self.player2_class = p2.player_class

        self.player1_name = p1.name
        self.player2_name = p2.name

        self.timestamp = int(time.time())
        self.timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            self.model1_name = p1.model.model_name
            self.model1_class = p1.model.model_class
            self.model1_timestamp = p1.model.timestamp
            self.model1_fullname = f'{self.model1_class}_{self.model1_name}_startstamp{self.model1_timestamp}'
        except AttributeError:
            self.model1_name = 'n/a'
            self.model1_class = 'n/a'
            self.model1_timestamp = 'n/a'
            self.model1_fullname = 'n/a'

        try:
            self.model2_name = p2.model.model_name
            self.model2_class = p2.model.model_class
            self.model2_timestamp = p2.model.timestamp
            self.model2_fullname = f'{self.model2_class}_{self.model2_name}_startstamp{self.model2_timestamp}'
        except AttributeError:
            self.model2_name = 'n/a'
            self.model2_class = 'n/a'
            self.model2_timestamp = 'n/a'
            self.model2_fullname = 'n/a'

        try:
            self.model1_used_path = p1.model.model_used_path
        except AttributeError:
            self.model1_used_path = 'n/a'
        try:
            self.model2_used_path = p2.model.model_used_path
        except AttributeError:
            self.model2_used_path = 'n/a'

    def add_constants(self):
        self.constants = {
            "MIN_REWARD": MIN_REWARD,
            "EPISODES": EPISODES,
            "REWARD_WINNING": FiarGame.REWARD_WINNING,
            "REWARD_LOSING": FiarGame.REWARD_LOSING,
            "REWARD_TIE": FiarGame.REWARD_TIE,
            "REWARD_INVALID_MOVE": FiarGame.REWARD_INVALID_MOVE,
            "REWARD_STEP": FiarGame.REWARD_STEP,
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
        f.write(f"loaded models at = {self.timenow}\n")
        f.write(f" timestamp = {self.timestamp}\n")

        # player/model info
        f.write(f"player1\n")
        f.write(f" class = {self.player1_class}\n")
        f.write(f" name = {self.player1_name}\n")
        f.write(f" model class = {self.model1_class}\n")
        f.write(f" model name = {self.model1_name}\n")
        f.write(f" model startstamp = {self.model1_timestamp}\n")
        f.write(f" model fullname = {self.model1_fullname}\n")
        f.write(f" model used path = {self.model1_used_path}\n")

        f.write(f"player2\n")
        f.write(f" class = {self.player2_class}\n")
        f.write(f" name = {self.player2_name}\n")
        f.write(f" model class = {self.model2_class}\n")
        f.write(f" model name = {self.model2_name}\n")
        f.write(f" model startstamp = {self.model2_timestamp}\n")
        f.write(f" model fullname = {self.model2_fullname}\n")
        f.write(f" model used path = {self.model2_used_path}\n")

        # constantsinfo
        for key, constant in self.constants.items():
            f.write(f"{key} = {constant}\n")
        f.write("\n")

        # closing file
        f.close()

    def log_text_to_file(self, text):
        f = open(self.logfilepath, "a+")
        f.write(f"{text}\n")
        f.close()

    def write_to_csv(self):
        with open('parameters.csv', mode='a+', newline='') as parameters_file:
            parameter_writer = csv.writer(parameters_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            parameter_writer.writerow([self.timestamp, self.timenow, 1, self.player1_class, self.player1_name, self.model1_class, self.model1_name, self.model1_timestamp, self.model1_fullname, self.model1_used_path])
            parameter_writer.writerow([self.timestamp, self.timenow, 2, self.player2_class, self.player2_name, self.model2_class, self.model2_name, self.model2_timestamp, self.model2_fullname, self.model2_used_path])

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
        self.model.model_name = f'PreTrainedModel-{old_modelname[0]}-{old_modelname[1]}-{old_modelname[2]}-{old_modelname[3]}'
        self.model.timestamp = int(time.time())
        self.model.model_class = self.__class__.__name__
        self.model.model_used_path = path


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
        time.sleep(2)  # needed for batch training otherwise with 2 same models there is a possibility that they will be instanciated at the same time, which causes tensorboard to append the logfile  onto each other.

    def create_model(self, input_shape, output_num):
        pass


class model1(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'conv3x128'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(128, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model1b(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdense'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model1c(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdenseSMALL3x3'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(12, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(24, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(48, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model1d(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdenseSMALL4x4'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(24, (4, 4), padding='same', activation='relu'))
        model.add(Conv2D(48, (4, 4), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(48, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model2(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense1x64'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model3(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense2x64'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model4(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax)'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Dense(128, input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Dense(128, input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        # model.add(Dense(64))
        #model.add(Dense(128, input_shape=input_shape))
        #model.add(Activation('relu'))

        model.add(Dense(output_num, activation='softmax'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model4a(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax)(flattenLAST,input_shape bug gone lr=0.001)'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Dense(128, input_shape=input_shape, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model4b(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax)(flattenfirst,input_shape bug gone lr=0.001)'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


class model4c(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax)(flattenfirst,input_shape bug gone, lr=0.01)'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.01), metrics=['accuracy'])
        return model


class model4catcross(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax+CatCrossEntr)'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Dense(128, input_shape=input_shape, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
        return model


class model4catcross2(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax+CatCrossEntr)2'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
        return model


class model4catcross3(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax+CatCrossEntr)3 lr=0.001'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
        return model


class model5(model_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dense4x128'

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
