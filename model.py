from keras.callbacks import TensorBoard
import tensorflow as tf

from keras import Sequential
from keras.models import load_model, Model as FuncModel
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers import Input, Add, Subtract, Lambda, concatenate, add  # functional API specific
import keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras import initializers

from constants import *
from game import FiarGame
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.animation as animation
import random

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
            self.model1_fin_activation = p1.model.fin_activation
        except AttributeError:
            self.model1_name = 'n/a'
            self.model1_class = 'n/a'
            self.model1_timestamp = 'n/a'
            self.model1_fullname = 'n/a'
            self.model1_opt_name = 'n/a'
            self.model1_lr = 'n/a'
            self.model1_loss = 'n/a'
            self.model1_acc = 'n/a'
            self.model1_fin_activation = 'n/a'

        try:
            self.model2_name = p2.model.model_name
            self.model2_class = p2.model.model_class
            self.model2_timestamp = p2.model.timestamp
            self.model2_fullname = f'{self.model2_class}_{self.model2_name}_startstamp{self.model2_timestamp}'
            self.model2_opt_name = p2.model.optimizer_name
            self.model2_lr = p2.model._lr
            self.model2_loss = p2.model.loss
            self.model2_acc = p2.model.metrics
            self.model2_fin_activation = p2.model.fin_activation
        except AttributeError:
            self.model2_name = 'n/a'
            self.model2_class = 'n/a'
            self.model2_timestamp = 'n/a'
            self.model2_fullname = 'n/a'
            self.model2_opt_name = 'n/a'
            self.model2_lr = 'n/a'
            self.model2_loss = 'n/a'
            self.model2_acc = 'n/a'
            self.model2_fin_activation = 'n/a'

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

        f = open(f'{self.logfilepath}.log', "a+")

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
        f.write(f" model final activation = {self.model1_fin_activation}\n")
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
        f.write(f" model final activation = {self.model2_fin_activation}\n")
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

        f = open(f'{self.logfilepath}.log', "a+")
        f.write(f"{text}\n")
        f.close()

    def write_to_csv(self):
        if self.log_flag is False:
            return  # do not write to file

        with open(f'{self.logfilepath}.csv', mode='a+', newline='') as parameters_file:
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


class AnalyseModel:
    def __init__(self):
        """
        Analyse the intermediate activation outputs per given layer.
        inspired by:
        https://github.com/gabrielpierobon/cnnshapes"""

        self.activation_fig_name = "activation at turn"
        self.state_fig_name = "state at turn"

    def update_model(self, model, analyze_layers=[1, 2, 3]):
        """Updates the model with new weights (for instance during training) """

        #layer_outputs = [layer.output for layer in model.layers[analyze_layers[0]:analyze_layers[1]]] # Extracts the outputs of the top x layers
        layer_outputs = [model.layers[layer_num].output for layer_num in analyze_layers]               # Extracts the outputs of the top x layers
        final_layer = model.layers[len(model.layers)-1]
        layer_outputs.append(final_layer.output)

        self.activation_model = FuncModel(inputs=model.input, outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
        self.activation_model.name = "analyze_activation_model"

        # get layernames
        self.activationlayer_names = []
        for layer in layer_outputs:
            self.activationlayer_names.append(layer.name)

        # other
        self.model_name = model.model_name
        self.n_features = []  # number of features per layer

    def visual_debug_train(self, state, turns, prefix="", print_num=False, save_to_file=False):
        """wrapper for visual debuging during training"""
        self.get_activations(state=state)
        #self.render_state(state=state, turns=turns)
        self.numberfy_activations(state=state, print_num=print_num)
        self.visualize_activations(state=state, turns=turns, save_to_file=False)

        if save_to_file:
            self.save_img(plot=plt, turns=turns, prefix=f'{prefix} activation at turn')

    def visual_debug_play(self, state, turns, print_num, save_to_file=True):
        """wrapper for visual debuging during playing"""
        self.get_activations(state=state)
        self.render_state(state=state, turns=turns)
        self.numberfy_activations(state=state, print_num=print_num)
        self.visualize_activations(state=state, turns=turns, save_to_file=save_to_file)

    def reset_figs(self):
        """resets the Pyplot Figures """

        self.plt = plt
        # setup for plotting
        self.state_fig = plt.figure()
        self.state_fig.suptitle('States')  # =actually the main title
        self.state_ims = []

        self.activation_fig = plt.figure()
        self.activation_fig.suptitle(f'Activations for: {self.model_name}')  # =actually the main title
        self.activation_ims = []

    def render_state(self, state, turns):
        plt.ioff()                          # turn off interactive plotting mode
        state2 = np.array(state[:,:,0])
        #state -= state.mean()              # Post-processes the feature to make it visually palatable
        #state /= state.std()
        #plt.matshow(state2)
        state2 *= 64
        state2 += 128

        plt.figure(self.state_fig.number)           # activate statefig
        plt.title(f"state at turn: {turns}")

        im = plt.imshow(state2, aspect='equal', cmap='autumn')

        self.state_ims.append([im])

        self.save_img(plot=plt, turns=turns, prefix='state at turn')

    def get_activations(self, state):
        """get activation layers for a given state"""

        state = np.array([state]) #/ 1                              # convert shape from (6,7,1) => (1,6,7,1) to be put into the model
        self.activations = self.activation_model.predict(state)     # get the output for each activation layer

        return self.activations

    def numberfy_activations(self, state, print_num=False):
        IMAGES_PER_ROW = 10

        activations = self.activations  # get_activations(state)
        layer_names = self.activationlayer_names        # Names of the layers, so you can have them as part of your plot

        n_features = []
        self.activation_mean_output = []
        for idx, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):      # Displays the feature maps
            n_features.append(layer_activation.shape[-1])                             # Number of features in the feature map
            n_cols = (n_features[idx] // IMAGES_PER_ROW) + 1                               # Tiles the activation channels in this matrix
            self.activation_mean_output.append([])

            if print_num:
                print(f'activation layer: {idx}, name: {layer_names[idx]}, total features: {n_features[idx]}')
            for col in range(n_cols):                                           # Tiles each filter into a big horizontal grid
                feature_output = {}
                for row in range(IMAGES_PER_ROW):

                    node = col * IMAGES_PER_ROW + row
                    if node >= n_features[idx]:
                        break

                    if len(layer_activation.shape) == 4:  # eg. Convolution layer
                        channel_image = layer_activation[0,
                                                        :, :,
                                                        node]
                        channel_image = np.mean(channel_image)
                    elif len(layer_activation.shape) == 2:  # eg. dense layer
                        channel_image = layer_activation[0, node]

                    self.activation_mean_output[idx].append(channel_image)
                    channel_image = round(channel_image, ndigits=2)
                    #channel_image *= 64
                    #channel_image += 128
                    feature_output[f'node {node}'] = channel_image
                if print_num:
                    print(f' {feature_output}')

        self.n_features = n_features

    def visualize_activations(self, state, turns, save_to_file):
        IMAGES_PER_ROW = 10
        BORDER_WIDTH = 2    # must be even!

        activations = self.activations  # get_activations(state)
        layer_names = self.activationlayer_names        # Names of the layers, so you can have them as part of your plot

        display_grid = []
        non_conv_layers_amount = 0

        for idx, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):      # Displays the feature maps
            #print(f'activation layer: {idx}')
            n_features = layer_activation.shape[-1]                             # Number of features in the feature map
            #size = layer_activation.shape[1]                                    # The feature map has shape (1, size, size, n_features).
            if len(layer_activation.shape) == 4:  # eg. Convolution layer
                size_h = layer_activation.shape[1]
                size_w = layer_activation.shape[2]
                n_cols = (n_features // IMAGES_PER_ROW) + 1                               # Tiles the activation channels in this matrix

                #display_grid = np.zeros((size * n_cols, IMAGES_PER_ROW * size))
                display_grid.append(np.zeros(((size_h + BORDER_WIDTH) * n_cols, IMAGES_PER_ROW * (size_w + BORDER_WIDTH))))
                for col in range(n_cols):                                           # Tiles each filter into a big horizontal grid
                    for row in range(IMAGES_PER_ROW):
                        node = col * IMAGES_PER_ROW + row
                        if node >= n_features:
                            break

                        channel_image = layer_activation[0,
                                                        :, :,
                                                        node]
                        channel_image -= channel_image.mean()                       # Post-processes the feature to make it visually palatable
                        channel_image /= (channel_image.std() + 0.00000001)
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[idx][int(BORDER_WIDTH/2) + (BORDER_WIDTH * col) + col * size_h : int(BORDER_WIDTH/2) + (BORDER_WIDTH * col) + ((col + 1) * size_h),                 # Displays the grid
                                        int(BORDER_WIDTH/2) + (BORDER_WIDTH * row) + row * size_w : int(BORDER_WIDTH/2) + (BORDER_WIDTH * row) + ((row + 1) * size_w)] = channel_image

                scale = 1. / size_w
                scale *= 0.7
            elif len(layer_activation.shape) == 2:  # eg. dense layer
                non_conv_layers_amount += 1

            """
            plt.figure(figsize=(scale * display_grid[idx].shape[1],
                                scale * display_grid[idx].shape[0]))
            plt.title(layer_name)
            plt.grid(True)
            plt.imshow(display_grid[idx], aspect='auto', cmap='viridis')
            """

        # find max x&yshape
        shape_x = []
        shape_y = []
        for grid in display_grid:
            shape_x.append(grid.shape[1])
            shape_y.append(grid.shape[0])

        # activate the statefig figure
        plt.figure(self.activation_fig.number)

        # set shape of plot
        COL = 3  # nr of column in figure
        self.activation_fig.set_figheight(scale * max(shape_y) * len(activations)) 
        self.activation_fig.set_figwidth(scale * 0.8 * max(shape_x) * COL)

        # make subplots in one list
        axs_act = []
        axs_line = []
        axs_st = []

        for idx in range(len(activations)):
            plot_id = idx * COL + 1
            ax = plt.subplot(len(activations), COL, plot_id)
            axs_act.append(ax)

            ax = plt.subplot(len(activations), COL, plot_id + 1)
            axs_line.append(ax)

            ax = plt.subplot(len(activations), COL, plot_id + 2)
            axs_st.append(ax)

        ims = []
        # iterate over subplots
        for idx, ax in enumerate(axs_act):
            try:
                #ax.plot(x, y)
                im = ax.imshow(display_grid[idx], aspect='equal', cmap='viridis')
                ims.append(im)
                ax.set_title(f'{layer_names[idx]}')

                #set x&y limits
                ax.set_xlim(-1, max(shape_x))
                ax.set_ylim(max(shape_y), -1)

                # Change major ticks to show every x and y of the input array (creating borders).
                ax.xaxis.set_major_locator(MultipleLocator(size_w + BORDER_WIDTH))
                ax.yaxis.set_major_locator(MultipleLocator(size_h + BORDER_WIDTH))

                # Change minor ticks to show every 1
                ax.xaxis.set_minor_locator(AutoMinorLocator(size_w + BORDER_WIDTH))
                ax.yaxis.set_minor_locator(AutoMinorLocator(size_h + BORDER_WIDTH))

                ax.grid(True)  # show gridlines
            except:
                # no conv layer, for example a dense layer
                pass

        for idx, ax in enumerate(axs_line):
            #ax.plot(x, y)
            x = np.array(self.activation_mean_output[idx])
            im, = ax.plot(x, label=turns)  # note the comma! Do no want a list, but an object. (otherwise can't create an animation)
            ims.append(im)
            ax.set_title(f'{layer_names[idx]}')

            #set x&y limits
            #ax.set_xlim(-1, max(self.n_features))
            MINIMUM_Y_LIM = 0.3
            y_lim = max(MINIMUM_Y_LIM, max(self.activation_mean_output[idx]))
            ax.set_xlim(0, self.n_features[idx])
            ax.set_ylim(0, y_lim)

            # Change major ticks to show every #.
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(y_lim / 5))

            # Change minor ticks to show every 1
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(y_lim / 5))
            ax.legend()
            ax.grid(True)  # show gridlines

        state2 = np.array(state[:, :, 0])
        for idx, ax in enumerate(axs_st):
            #ax.plot(x, y)
            im = ax.imshow(state2, aspect='equal', cmap='autumn')
            ims.append(im)
            ax.set_title(f'state at turn {turns}')

            # set x&y limits
            ax.set_xlim(-1, size_w)
            ax.set_ylim(size_h, -1)

            # Change major ticks to show every x and y of the input array (creating borders).
            ax.xaxis.set_major_locator(MultipleLocator(size_w + BORDER_WIDTH))
            ax.yaxis.set_major_locator(MultipleLocator(size_h + BORDER_WIDTH))

            # Change minor ticks to show every 1
            ax.xaxis.set_minor_locator(AutoMinorLocator(size_w + BORDER_WIDTH))
            ax.yaxis.set_minor_locator(AutoMinorLocator(size_h + BORDER_WIDTH))

            ax.grid(False)  # show gridlines
            break

        #im = self.state_ims[len(self.state_ims)-1]
        #ims.append(im)
        if save_to_file:
            self.save_img(plot=plt, turns=turns, prefix='activation at turn')
        #plt.show()  # show plot

        #plt.close(fig)
        #self.activation_fig = fig
        #im = ax.imshow(display_grid[0], aspect='auto', cmap='viridis')
        #self.activation_ims.append(axs)
        self.activation_ims.append(ims)

    def save_img(self, plot, prefix, turns):
        plot.savefig(f'output/{prefix}[{turns}].png')

    def render_vid(self):

        #
        plt.rcParams['animation.ffmpeg_path'] = '..\\ffmpeg-20190930-6ca3d34-win64-static\\bin\\ffmpeg.exe'
        FFwriter = animation.FFMpegWriter()

        ani = animation.ArtistAnimation(self.state_fig, self.state_ims, interval=1500, blit=True,
                                        repeat_delay=5000)
        ani.save('output/state.mp4', writer=FFwriter)
        # ani.save('dynamic_images.mpg')

        ani = animation.ArtistAnimation(self.activation_fig, self.activation_ims, interval=1500, blit=True,
                                        repeat_delay=5000, )
        ani.save('output/activations.mp4', writer=FFwriter)

######################################################################################
#
#   # ##  ##     ###    ####    ######  #        ####
#   #   #   #   #   #   #    #  #       #       #
#   #   #   #   #   #   #    #  ####    #        ##
#   #       #   #   #   #    #  #       #           #
#   #       #    ###    ####    #####   ######  ###
#
######################################################################################


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
    def __init__(self, input_shape, output_num, par_loss, par_opt, par_metrics, par_final_act, *args, **kwargs):
        # K.set_floatx('float64')

        # parameters
        self.loss = par_loss
        self.opt = par_opt
        self.metrics = par_metrics
        self.fin_activation = par_final_act

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
        self.model.fin_activation = self.fin_activation
        # print(self.model.optimizer.get_config())

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


class model1(model_base):
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

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


class func_model1(model_base):
    """
    Good hyperparameters:
    - par_loss='mse', par_opt=Adam(lr=0.001) against drunk and against itself

    Medium hyperparameters:
    -

    Bad hyperparameters:
    -par_loss='categorical_crossentropy', par_opt=Adam(lr=0.01), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'
    -par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'
    """
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = '3xconv+2xdenseSMALL4x4(func)'
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
        predictions = Dense(output_num, activation=self.fin_activation, kernel_initializer='he_normal')(x)

        # This creates a model that includes
        #  the Input layer and the stacked output layers
        model = FuncModel(inputs=inputs, outputs=predictions)
        # model.compile(optimizer='rmsprop',
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
        value = Dense(1, activation=self.fin_activation)(value)

        advantage = Dense(32, activation='relu')(x)
        advantage = Dense(output_num, activation=self.fin_activation)(advantage)
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
        # defaults keyword arguments
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'linear')

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
        value = Dense(1, activation=self.fin_activation)(value_fc)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(output_num,))(value)

        advantage_fc = Dense(32, activation='relu')(x)
        advantage = Dense(output_num, activation=self.fin_activation)(advantage_fc)
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
        self.model.model_name = 'dense1x64'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, input_shape=input_shape, activation='relu'))

        model.add(Dense(output_num, activation=self.fin_activation))
        # model.compile happens in baseclass method compile_model()
        return model


class model3(model_base):
    def __init__(self, **kwargs):
        # defaults keyword arguments
        kwargs['par_loss'] = kwargs.pop('par_loss', 'mse')
        kwargs['par_opt'] = kwargs.pop('par_opt', Adam(lr=0.001))
        kwargs['par_metrics'] = kwargs.pop('par_metrics', 'accuracy')
        kwargs['par_final_act'] = kwargs.pop('par_final_act', 'softmax')

        super().__init__(**kwargs)
        self.model.model_name = 'dense2x64'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        model = Sequential()

        model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(output_num, activation=self.fin_activation))
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
        self.model.model_name = 'func_dense4x64'
        self.append_hyperpar_to_name()

    def create_model(self, input_shape, output_num):
        # This returns a tensor
        inputs = Input(shape=input_shape)

        x = Flatten()(inputs)  # converts the 3D feature maps to 1D feature vectors
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)

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
        self.model.model_name = 'dueling_dense4x64'
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
