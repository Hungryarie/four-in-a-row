from keras.callbacks import TensorBoard
import tensorflow as tf

from keras import Sequential
from keras.models import load_model, Model as FuncModel
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers import Input, Add, Subtract, Lambda, concatenate, add  # functional API specific
import keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop

from constants import *
from game import FiarGame
import time
from datetime import datetime

import csv


class ModelLog():
    def __init__(self, file, log_flag=True):
        self.logfilepath = file
        self.log_flag = log_flag

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
            self.model1_opt_name = p1.model.optimizer_name
            self.model1_lr = p1.model._lr
            self.model1_loss = p1.model.loss
            self.model1_acc = p1.model.metrics
        except AttributeError:
            self.model1_name = 'n/a'
            self.model1_class = 'n/a'
            self.model1_timestamp = 'n/a'
            self.model1_fullname = 'n/a'
            self.model1_opt_name = 'n/a'
            self.model1_lr = 'n/a'
            self.model1_loss = 'n/a'
            self.model1_acc = 'n/a'

        try:
            self.model2_name = p2.model.model_name
            self.model2_class = p2.model.model_class
            self.model2_timestamp = p2.model.timestamp
            self.model2_fullname = f'{self.model2_class}_{self.model2_name}_startstamp{self.model2_timestamp}'
            self.model2_opt_name = p2.model.optimizer_name
            self.model2_lr = p2.model._lr
            self.model2_loss = p2.model.loss
            self.model2_acc = p2.model.metrics
        except AttributeError:
            self.model2_name = 'n/a'
            self.model2_class = 'n/a'
            self.model2_timestamp = 'n/a'
            self.model2_fullname = 'n/a'
            self.model2_opt_name = 'n/a'
            self.model2_lr = 'n/a'
            self.model2_loss = 'n/a'
            self.model2_acc = 'n/a'

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
        if self.log_flag is False:
            return  # do not write to file

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
        f.write(f" model loss func = {self.model1_loss}\n")
        f.write(f" model optimizer = {self.model1_opt_name} lr={self.model1_lr}\n")
        f.write(f" model startstamp = {self.model1_timestamp}\n")
        f.write(f" model fullname = {self.model1_fullname}\n")
        f.write(f" model used path = {self.model1_used_path}\n")

        f.write(f"player2\n")
        f.write(f" class = {self.player2_class}\n")
        f.write(f" name = {self.player2_name}\n")
        f.write(f" model class = {self.model2_class}\n")
        f.write(f" model name = {self.model2_name}\n")
        f.write(f" model loss func = {self.model2_loss}\n")
        f.write(f" model optimizer = {self.model2_opt_name} lr={self.model2_lr}\n")
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
        if self.log_flag is False:
            return  # do not write to file

        f = open(self.logfilepath, "a+")
        f.write(f"{text}\n")
        f.close()

    def write_to_csv(self):
        if self.log_flag is False:
            return  # do not write to file

        with open('parameters.csv', mode='a+', newline='') as parameters_file:
            parameter_writer = csv.writer(parameters_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            parameter_writer.writerow([self.timestamp, self.timenow, 1, self.player1_class, self.player1_name, self.model1_class, self.model1_name, self.model1_loss, self.model1_opt_name, self.model1_lr, self.model1_timestamp, self.model1_fullname, self.model1_used_path])
            parameter_writer.writerow([self.timestamp, self.timenow, 2, self.player2_class, self.player2_name, self.model2_class, self.model2_name, self.model2_loss, self.model2_opt_name, self.model2_lr, self.model2_timestamp, self.model2_fullname, self.model2_used_path])
        parameters_file.close()


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
        # temporary fix.. see issue #6
        time.sleep(2)  # needed for batch training otherwise with 2 same models there is a possibility that they will be instanciated at the same time, which causes tensorboard to append the logfile  onto each other.


class model_base:
    """
    Base class for all models
    """
    def __init__(self, input_shape, output_num, par_loss, par_opt, par_metrics, *args, **kwargs):  #par_loss="mse", par_opt=Adam(lr=0.001), par_metrics="accuracy", 
        # K.set_floatx('float64')

        # parameters
        self.loss = par_loss
        self.opt = par_opt
        self.metrics = par_metrics

        # create models
        self.model = self.create_model(input_shape, output_num)
        self.target_model = self.create_model(input_shape, output_num)
        self.compile_model(self.model)
        self.compile_model(self.target_model)

        self.model.model_name = None
        self.model.timestamp = int(time.time())
        self.model.model_class = self.__class__.__name__

        # for stats and logging
        self.model._lr = K.eval(self.model.optimizer.lr)
        self.model._lr = format(self.model._lr, '.00000g')
        self.model.optimizer_name = self.model.optimizer.__class__.__name__
        #print(self.model.optimizer.get_config())

        # temporary fix.. see issue #6
        time.sleep(2)  # needed for batch training otherwise with 2 same models there is a possibility that they will be instanciated at the same time, which causes tensorboard to append the logfile  onto each other.

    def create_model(self, input_shape, output_num):
        pass

    def compile_model(self, model):
        model.compile(loss=self.loss, optimizer=self.opt, metrics=[self.metrics])

    def append_hyperpar_to_name(self):
        clip_string = ""
        if hasattr(self.model.optimizer, 'clipnorm'):
            clip_string += f"^clipnorm={self.model.optimizer.clipnorm}"
        if hasattr(self.model.optimizer, 'clipvalue'):
            clip_string += f"^clipvalue={self.model.optimizer.clipvalue}"

        self.model.model_name += f'({self.loss}^{self.model.optimizer_name}^lr={self.model._lr}{clip_string})'

    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class model1(model_base):
    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = 'conv3x128'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(128, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Dense(output_num, activation='softmax'))

        # model.compile happens in baseclass method compile_model()
        return model


class model1b(model_base):
    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdense'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        # model.compile happens in baseclass method compile_model()
        return model


class model1c(model_base):
    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdenseSMALL3x3'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(12, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(24, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(48, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        # model.compile happens in baseclass method compile_model()
        return model


class model1d(model_base):
    """model version made with the Sequential API
    Functional API counterpart is 'func_model1()"""
    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdenseSMALL4x4(seq)'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(24, (4, 4), padding='same', activation='relu'))
        model.add(Conv2D(48, (4, 4), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(48, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_num, activation='softmax'))

        # model.compile happens in baseclass method compile_model()
        return model


class func_model1(model_base):
    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdenseSMALL4x4(func)'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu')(inputs)
        x = Conv2D(24, (4, 4), padding='same', activation='relu')(x)
        x = Conv2D(48, (4, 4), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(48, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        predictions = Dense(output_num, activation='softmax')(x)

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=predictions)
        #model.compile(optimizer='rmsprop',
        #            loss='categorical_crossentropy',
        #            metrics=['accuracy'])
        # model.compile happens in baseclass method compile_model()
        return model


class func_model_duel1b(model_base):
    """dueling model (using functional Keras API)
    without extra dense layer after model split.

    Good hyperparameters:
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9) => against trained model
    -par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001) => old func_model_duel1c, against trained model

    Medium hyperparameters:
    

    Bad hyperparameters:
    -par_loss='mse', par_opt=Adam(lr=0.001) = > old func_model_duel1
    -par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001) => #old func_model_duel1c, against random model
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9, clipnorm=1.0, clipvalue=0.5) not great model
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9) # returns NaNs at episode 130, against random model
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dueling_3xconv+2xdenseSMALL4x4'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu')(inputs)
        x = Conv2D(24, (4, 4), padding='same', activation='relu')(x)
        x = Conv2D(48, (4, 4), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(48, activation='relu')(x)
        x = Dense(32, activation='relu')(x)

        value = Dense(1, activation='linear')(x)
        advantage = Dense(output_num, activation='linear')(x)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)
        advantage = Subtract()([advantage, mean])
        predictions = Add()([value, advantage])

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=predictions)
        # model.compile happens in baseclass method compile_model()
        return model


class func_model_duel1b1(model_base):
    """dueling model (using functional Keras API)
    with extra dense layer after model split.

    Good hyperparameters:
    - not yet found

    Bad hyperparameters:
    -par_loss='mse', par_opt=Adam(lr=0.001) => not great
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9) => creates around 85 eps NaN
    -par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001) => not great
    - par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001, clipnorm=1.0, clipvalue=0.5) => not great
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = f'dueling_3xconv+2xdenseSMALL4x4+extra dense Functmethod1'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu')(inputs)
        x = Conv2D(24, (4, 4), padding='same', activation='relu')(x)
        x = Conv2D(48, (4, 4), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(48, activation='relu')(x)

        value = Dense(32, activation='relu')(x)
        value = Dense(1, activation='linear')(value)

        advantage = Dense(32, activation='relu')(x)
        advantage = Dense(output_num, activation='linear')(advantage)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)
        advantage = Subtract()([advantage, mean])

        predictions = Add()([value, advantage])

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=predictions)
        # model.compile happens in baseclass method compile_model()
        return model


class func_model_duel1b2(model_base):
    """creates around 900 eps NaN

    Good hyperparameters:
    -

    Bad hyperparameters:
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9) => creates around 900 eps NaN
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dueling_3xconv+2xdenseSMALL4x4+extra dense Functmethod2'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu')(inputs)
        x = Conv2D(24, (4, 4), padding='same', activation='relu')(x)
        x = Conv2D(48, (4, 4), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(48, activation='relu')(x)

        # network separate state value and advantages
        value_fc = Dense(32, activation='relu')(x)
        value = Dense(1, activation='linear')(value_fc)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(output_num,))(value)

        advantage_fc = Dense(32, activation='relu')(x)
        advantage = Dense(output_num, activation='linear')(advantage_fc)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                           output_shape=(output_num,))(advantage)

        q_value = add([value, advantage])

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=q_value)
        # model.compile happens in baseclass method compile_model()
        return model


class model2(model_base):
    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = 'dense1x64'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        
        model.add(Dense(output_num, activation='softmax'))
        # model.compile happens in baseclass method compile_model()
        return model


class model3(model_base):
    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = 'dense2x64'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dense(64, activation='relu'))
        
        model.add(Dense(output_num, activation='softmax'))
        # model.compile happens in baseclass method compile_model()
        return model


class model4a(model_base):
    """
    -par_loss='mse', par_opt=Adam(lr=0.001) # defaul model4a
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.01, momentum=0.9) # old model4catcross
    """
    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax)(flattenLAST)'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Dense(128, input_shape=input_shape, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors

        model.add(Dense(output_num, activation='softmax'))
        # model.compile happens in baseclass method compile_model()
        return model


class model4b(model_base):
    """
    -par_loss='mse', par_opt=Adam(lr=0.001) # default model4b
    -par_loss='mse', par_opt=Adam(lr=0.01) # old model4c
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.01, momentum=0.9) # old model4catcross2
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9) # old model4catcross3
    """

    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax)(flattenfirst)'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))

        model.add(Dense(output_num, activation='softmax'))
        # model.compile happens in baseclass method compile_model()
        return model


class model5(model_base):
    """sequential api: dense4x128"""
    def __init__(self, **kwargs):
        #defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')

        super().__init__(**kwargs)
        self.model.model_name = 'dense4x64'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        #input_shape=input_shape
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(output_num, activation='softmax'))
        # model.compile happens in baseclass method compile_model()
        return model

class func_model5(model_base):
    """model (using functional Keras API)
    dense4x128

    Good hyperparameters:
    -

    Medium hyperparameters:
    -

    Bad hyperparameters:
    -
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'func_dense4x64'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        x = Flatten()(inputs) # converts the 3D feature maps to 1D feature vectors
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        predictions = Dense(output_num, activation='softmax')(x)
        model = FuncModel(inputs=inputs, outputs=predictions)
        # model.compile happens in baseclass method compile_model()
        return model


class func_model5_duel1(model_base):
    """dueling model (using functional Keras API)
    dense4x128

    Good hyperparameters:
    -

    Medium hyperparameters:
    -par_loss='mse', par_opt=Adam(lr=0.001)

    Bad hyperparameters:
    -
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.model_name = 'dueling_dense4x64'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        x = Flatten()(inputs) # converts the 3D feature maps to 1D feature vectors
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        value = Dense(1, activation='linear')(x)
        advantage = Dense(output_num, activation='linear')(x)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)
        advantage = Subtract()([advantage, mean])
        predictions = Add()([value, advantage])

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=predictions)
        # model.compile happens in baseclass method compile_model()
        return model
