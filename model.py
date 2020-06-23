import tensorflow as tf
# from keras.callbacks import TensorBoard

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model, Model as FuncModel
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, LeakyReLU
from tensorflow.keras.layers import Input, Add, Subtract, Lambda, concatenate, add  # functional API specific
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import initializers

# from constants import *
# from game import FiarGame
import time
# from datetime import datetime
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
# import matplotlib.animation as animation
# import random


class empty_model:
    def __init__(self):
        pass


class load_a_model:
    def __init__(self, path):
        try:
            self.model = load_model(path)
            self.target_model = load_model(path)
        except Exception:
            raise NameError(f"could not open ({path}). Doesn't exist?")

        path = path[7:]  # removes subdir (models/)
        old_modelname = path.split('_')
        # when old_modelname has a to short array length, append it with 'na'
        if len(old_modelname) <= 3:
            for i in range(len(old_modelname), 3 + 1):
                old_modelname.append('na')
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
    def __init__(self, input_shape, output_num, par_loss, par_opt, par_metrics, par_final_act, *args, **kwargs):
        # K.set_floatx('float64')

        # parameters
        self.loss = par_loss
        self.opt = par_opt
        self.metrics = par_metrics
        self.fin_activation = par_final_act
        self.layer_multiplier = kwargs.pop('par_layer_multiplier', 1)

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
        self.model.fin_activation = self.fin_activation  # final activation
        # print(self.model.optimizer.get_config())

        # self.model.in_shape = input_shape
        # self.model.out_num = output_num

        self.model.hyper_dict = {}
        self.model.hyper_dict['model_name'] = None
        self.model.hyper_dict['timestamp'] = int(time.time())
        self.model.hyper_dict['model_class'] = self.__class__.__name__
        self.model.hyper_dict['learning_rate'] = format(K.eval(self.model.optimizer.lr), '.00000g')
        self.model.hyper_dict['optimizer_name'] = self.model.optimizer.__class__.__name__
        self.model.hyper_dict['final_activation'] = self.fin_activation
        self.model.hyper_dict['input_shape'] = input_shape
        self.model.hyper_dict['output_num'] = output_num
        self.model.hyper_dict['layer_multiplier'] = self.layer_multiplier

        # temporary fix.. see issue #6
        time.sleep(2)  # needed for batch training otherwise with 2 same models there is a possibility that they will be instanciated at the same time, which causes tensorboard to append the logfile  onto each other.

    # def he_normal(self):
    #    return initializers.he_normal(seed=None)

    def create_model(self, input_shape, output_num):
        pass

    def compile_model(self, model):
        # if self.loss == 'huber_loss':
        #    self.loss = tf.losses.huber_loss #self.huber_loss
        # elif self.loss == 'huber_loss_mean':
        #    self.loss = self.huber_loss_mean

        model.compile(loss=self.loss, optimizer=self.opt, metrics=[self.metrics])

    def append_hyperpar_to_name(self):
        clip_string = ""
        if hasattr(self.model.optimizer, 'clipnorm'):
            clip_string += f"^clipnorm={self.model.optimizer.clipnorm}"
        if hasattr(self.model.optimizer, 'clipvalue'):
            clip_string += f"^clipvalue={self.model.optimizer.clipvalue}"

        if callable(self.loss):
            loss_name = self.loss.__name__
        else:
            loss_name = self.loss

        self.model.model_name += f'({loss_name}^{self.model.optimizer_name}^lr={self.model._lr}{clip_string}^{self.fin_activation})'
        self.model.hyper_dict['model_name'] += f'({loss_name}^{self.model.optimizer_name}^lr={self.model._lr}{clip_string}^{self.fin_activation})'
    """
    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        # not easy to load model with custom loss function.
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def huber_loss_mean(self, y_true, y_pred, clip_delta=1.0):
        # not easy to load model with custom loss function.
        return K.mean(self.huber_loss(y_true, y_pred, clip_delta))
    """

######################################################################################
#
#   # ##  ##     ###    ####    ######  #        ####
#   #   #   #   #   #   #    #  #       #       #
#   #   #   #   #   #   #    #  ####    #        ##
#   #       #   #   #   #    #  #       #           #
#   #       #    ###    ####    #####   ######  ###
#
######################################################################################


class model1(model_base):
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = 'conv3x128'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(128, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Dense(output_num, activation=self.fin_activation))

        # model.compile happens in baseclass method compile_model()
        return model


class model1b(model_base):
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdense'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_num, activation=self.fin_activation))

        # model.compile happens in baseclass method compile_model()
        return model


class model1c(model_base):
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdenseSMALL3x3'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(12, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(24, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(48, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_num, activation=self.fin_activation))

        # model.compile happens in baseclass method compile_model()
        return model


class model1d(model_base):
    """model version made with the Sequential API
    Functional API counterpart is 'func_model1()"""
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdenseSMALL4x4(seq)'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu'))
        model.add(Conv2D(24, (4, 4), padding='same', activation='relu'))
        model.add(Conv2D(48, (4, 4), padding='same', activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(48, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_num, activation=self.fin_activation))

        # model.compile happens in baseclass method compile_model()
        return model


class func_model1_small_conv(model_base):
    """
    Good hyperparameters:
    -

    Medium hyperparameters:
    -

    Bad hyperparameters:
    -
    """
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = '1xconv+2xdenseSMALL5x5(func)-HEnormal'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        multipl = self.layer_multiplier

        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Conv2D(128 * multipl, (5, 5), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x = Flatten()(x)
        x = Dense(48 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(24 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        predictions = Dense(output_num, activation=self.fin_activation, kernel_initializer='he_normal')(x)

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=predictions)
        return model


class func_model1(model_base):
    """
    Good hyperparameters:
    - par_loss='mse', par_opt=Adam(lr=0.001) against drunk and against itself

    Medium hyperparameters:
    -

    Bad hyperparameters:
    - par_loss='logcosh', par_opt=Adam(lr=0.01), par_metrics='accuracy', par_final_act='linear' loss explosion at 600 steps
    - par_loss='logcosh', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear' loss explosion at 10k steps
    - par_loss='categorical_crossentropy', par_opt=Adam(lr=0.01), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'
    - par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'
    """
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+1xdenseSMALL3x3(func)-HEnormal(TANH)'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        multipl = self.layer_multiplier

        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Conv2D(12 * multipl, (3, 3), input_shape=input_shape, data_format="channels_last", padding='same', activation='tanh', kernel_initializer='he_normal')(inputs)
        x = Conv2D(24 * multipl, (3, 3), padding='valid', activation='tanh', kernel_initializer='he_normal')(x)
        x = Conv2D(48 * multipl, (3, 3), padding='valid', activation='tanh', kernel_initializer='he_normal')(x)
        x = Flatten()(x)
        x = Dense(48 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        # x = Dense(32 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        predictions = Dense(output_num, activation=self.fin_activation, kernel_initializer='he_normal')(x)

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=predictions)
        # model.compile(optimizer='rmsprop',
        #            loss='categorical_crossentropy',
        #            metrics=['accuracy'])
        # model.compile happens in baseclass method compile_model()
        return model


class ACmodel1(model_base):
    """
    Good hyperparameters:
    - par_loss='mse', par_opt=Adam(lr=0.001) against drunk and against itself

    Medium hyperparameters:
    -

    Bad hyperparameters:
    - par_loss='logcosh', par_opt=Adam(lr=0.01), par_metrics='accuracy', par_final_act='linear' loss explosion at 600 steps
    - par_loss='logcosh', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear' loss explosion at 10k steps
    - par_loss='categorical_crossentropy', par_opt=Adam(lr=0.01), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'
    - par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'
    """
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+3xdenseSMALL3x3(twohead)-HEnormal(ReLu) RESIDUAL'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        multipl = self.layer_multiplier

        inputs = Input(shape=input_shape)

        x = Conv2D(24 * multipl, (3, 3), input_shape=input_shape, use_bias=False, data_format="channels_last", padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        block_1_output = Activation('relu')(x)
        x = Conv2D(24 * multipl, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(block_1_output)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, block_1_output])

        x = Conv2D(48 * multipl, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        block_2_output = Activation('relu')(x)
        x = Conv2D(48 * multipl, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(block_2_output)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, block_2_output])

        x = Conv2D(96 * multipl, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        block_3_output = Activation('relu')(x)
        x = Conv2D(96 * multipl, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(block_3_output)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, block_3_output])


        x = Flatten()(x)
        act = Dense(128 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        act = Dense(64 * multipl, activation='relu', kernel_initializer='he_normal')(act)
        act_predictions = Dense(output_num, activation=self.fin_activation, kernel_initializer='he_normal')(act)

        crit = Dense(128 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        crit = Dense(64 * multipl, activation='relu', kernel_initializer='he_normal')(crit)
        crit_predictions = Dense(1, activation='linear', kernel_initializer='he_normal')(crit)

        model = FuncModel(inputs=inputs, outputs=[act_predictions, crit_predictions])

        return model


class func_model_duel1b(model_base):
    """dueling model (using functional Keras API)
    without extra dense layer after model split.

    Good hyperparameters:
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9) => against trained model
    -par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001) => old func_model_duel1c, against trained model

    Medium hyperparameters:
    -

    Bad hyperparameters:
    -par_loss='mse', par_opt=Adam(lr=0.001) = > old func_model_duel1
    -par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001) => #old func_model_duel1c, against random model
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9, clipnorm=1.0, clipvalue=0.5) not great model
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9) # returns NaNs at episode 130, against random model
    """

    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'linear')

        super().__init__(**kwargs)
        self.model.model_name = 'dueling_3xconv+2xdenseSMALL4x4'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x = Conv2D(24, (4, 4), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(48, (4, 4), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Flatten()(x)
        x = Dense(48, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)

        value = Dense(1, activation=self.fin_activation)(x)
        advantage = Dense(output_num, activation=self.fin_activation)(x)
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
        # defaults keyword arguments
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'linear')

        super().__init__(**kwargs)
        self.model.model_name = f'dueling_3xconv+2xdenseSMALL4x4+extra dense Functmethod1-HEnormal'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x = Conv2D(24, (4, 4), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(48, (4, 4), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(48, activation='relu', kernel_initializer='he_normal')(x)

        value = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
        value = Dense(1, activation=self.fin_activation, kernel_initializer='he_normal')(value)

        advantage = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
        advantage = Dense(output_num, activation=self.fin_activation, kernel_initializer='he_normal')(advantage)
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
    - par_loss='logcosh', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear' loss explosion at 600 steps
    - par_loss='categorical_crossentropy', par_opt=SGD(lr=0.001, momentum=0.9) => creates around 900 eps NaN
    """
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'linear')

        super().__init__(**kwargs)
        self.model.model_name = 'dueling_3xconv+2xdenseSMALL4x4+extra dense Functmethod2-HEnormal'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Conv2D(12, (4, 4), input_shape=input_shape, data_format="channels_last", padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x = Conv2D(24, (4, 4), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(48, (4, 4), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(48, activation='relu', kernel_initializer='he_normal')(x)

        # network separate state value and advantages
        value_fc = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
        value = Dense(1, activation=self.fin_activation)(value_fc)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(output_num,))(value)

        advantage_fc = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
        advantage = Dense(output_num, activation=self.fin_activation, kernel_initializer='he_normal')(advantage_fc)
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
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = 'dense1x64, HE_normal'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))

        model.add(Dense(output_num, activation=self.fin_activation, kernel_initializer='he_normal'))
        # model.compile happens in baseclass method compile_model()
        return model


class func_model2(model_base):
    """
    Good hyperparameters:
    -

    Medium hyperparameters:
    -

    Bad hyperparameters:
    -
    """
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = '1xdense(func)-HEnormal'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        multipl = self.layer_multiplier

        # This returns a tensor
        inputs = Input(shape=input_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        # This returns a tensor
        inputs = Input(shape=input_shape)
        x = Flatten()(inputs)
        x = Dense(24 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        predictions = Dense(output_num, activation=self.fin_activation, kernel_initializer='he_normal')(x)

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=predictions)
        return model


class model3(model_base):
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = 'dense2x64, HE_normal'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))

        model.add(Dense(output_num, activation=self.fin_activation, kernel_initializer='he_normal'))
        # model.compile happens in baseclass method compile_model()
        return model


class model4a(model_base):
    """
    -par_loss='mse', par_opt=Adam(lr=0.001) # defaul model4a
    -par_loss='categorical_crossentropy', par_opt=SGD(lr=0.01, momentum=0.9) # old model4catcross
    """
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax)(flattenLAST)'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Dense(128, input_shape=input_shape, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors

        model.add(Dense(output_num, activation=self.fin_activation))
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
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = 'dense2x128(softmax)(flattenfirst)'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))

        model.add(Dense(output_num, activation=self.fin_activation))
        # model.compile happens in baseclass method compile_model()
        return model


class model5(model_base):
    """sequential api: dense4x128"""
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = 'dense4x64'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        # input_shape=input_shape
        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(output_num, activation=self.fin_activation))
        # model.compile happens in baseclass method compile_model()
        return model


class func_model5(model_base):
    """model (using functional Keras API)
    dense4x64

    Good hyperparameters:
    -par_loss='huber_loss_mean', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'

    Medium hyperparameters:
    -par_loss='huber_loss', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'

    Bad hyperparameters:
    -par_loss='mse', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'
    """

    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = 'func_dense4x64-HEnormal'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        multipl = self.layer_multiplier

        # This returns a tensor
        inputs = Input(shape=input_shape)

        x = Flatten()(inputs)  # converts the 3D feature maps to 1D feature vectors
        x = Dense(64 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(64 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(64 * multipl, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(64 * multipl, activation='relu', kernel_initializer='he_normal')(x)

        predictions = Dense(output_num, activation=self.fin_activation)(x)
        model = FuncModel(inputs=inputs, outputs=predictions)
        # model.compile happens in baseclass method compile_model()
        return model


class func_model5_duel1(model_base):
    """dueling model (using functional Keras API)
    dense4x128

    Good hyperparameters:
    -

    Medium hyperparameters:
    -par_loss='huber_loss', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'
    -par_loss='huber_loss_mean', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'

    Bad hyperparameters:
    -par_loss='mse', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'
    """

    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'linear')

        super().__init__(**kwargs)
        self.model.model_name = 'dueling_dense4x64-HEnormal'
        self.model.hyper_dict['model_name'] = self.model.model_name
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        x = Flatten()(inputs)  # converts the 3D feature maps to 1D feature vectors
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)

        value = Dense(1, activation=self.fin_activation)(x)
        advantage = Dense(output_num, activation=self.fin_activation)(x)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)
        advantage = Subtract()([advantage, mean])
        predictions = Add()([value, advantage])

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=predictions)
        # model.compile happens in baseclass method compile_model()
        return model
