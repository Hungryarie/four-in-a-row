import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoLocator, AutoMinorLocator, MultipleLocator, LogLocator)
import matplotlib.animation as animation
from tensorflow.keras.models import load_model, Model as FuncModel
import time
import os


class AnalyseModel:
    def __init__(self):
        """
        Analyse the intermediate activation outputs per given layer.
        inspired by:
        https://github.com/gabrielpierobon/cnnshapes"""

        self.activation_fig_name = "activation at turn"
        self.state_fig_name = "state at turn"

        self.state_memory_list = []

    def update_model(self, model, analyze_layers=[1, 2, 3, 4, 5, 6]):
        """Updates the model with new weights (for instance during training) """

        layer_outputs = [model.layers[layer_num].output for layer_num in analyze_layers if layer_num <= len(model.layers) - 1]               # Extracts the outputs of the top x layers

        # remove flatten layer (if present)
        for idx, layers in enumerate(layer_outputs):
            if layers.name.startswith('flatten'):
                del layer_outputs[idx]
                break

        # add final layer (if not present)
        final_layer = model.layers[len(model.layers) - 1]
        if final_layer.output.name != layer_outputs[-1].name:
            layer_outputs.append(final_layer.output)

        self.activation_model = FuncModel(inputs=model.input, outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
        self.activation_model.name2 = "analyze_activation_model"

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
        # self.render_state(state=state, turns=turns)
        self.numberfy_activations(state=state, print_num=print_num)
        self.visualize_activations(state=state, turns=turns, save_to_file=False)

        if save_to_file:
            self.save_img(plot=plt, turns=turns, prefix=f'{prefix} activation at turn')

    def visual_debug_play(self, state, turns, print_num, save_to_file=True):
        """wrapper for visual debuging during playing"""
        self.get_activations(state=state)
        self.render_state(state=state, turns=turns)
        self.numberfy_activations(state=state, print_num=print_num)
        if save_to_file:
            self.visualize_activations(state=state, turns=turns, save_to_file=save_to_file)

    def reset_figs(self):
        """resets the Pyplot Figures """

        plt.close('all')

        self.plt = plt
        # setup for plotting
        self.state_fig = plt.figure()
        self.state_fig.suptitle('States')  # =actually the main title
        self.state_ims = []

        self.activation_fig = plt.figure()
        self.activation_fig.suptitle(f'Activations for: {self.model_name}')  # =actually the main title
        self.activation_ims = []

    def render_state(self, state, turns):
        "Renders and saves an image of the given state"
        plt.ioff()                          # turn off interactive plotting mode
        state2 = np.array(state[:, :, 0])
        # state -= state.mean()              # Post-processes the feature to make it visually palatable
        # state /= state.std()
        # plt.matshow(state2)
        state2 *= 64
        state2 += 128

        plt.figure(self.state_fig.number)           # activate statefig
        plt.title(f"state at turn: {turns}")

        im = plt.imshow(state2, aspect='equal', cmap='autumn')

        self.state_ims.append([im.copy()])

        self.save_img(plot=plt, turns=turns, prefix='state at turn')

    def get_activations(self, state):
        """get activation layers for a given state"""

        state = np.array([state])  # / 1                              # convert shape from (6,7,1) => (1,6,7,1) to be put into the model
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
                    # channel_image *= 64
                    # channel_image += 128
                    feature_output[f'node {node}'] = channel_image
                if print_num:
                    print(f' {feature_output}')

        self.n_features = n_features

    def visualize_activations(self, state, turns, save_to_file):
        IMAGES_PER_ROW = 10
        BORDER_WIDTH = 2    # must be even!
        scale = 2.5   # default
        size_h = 10  # default
        size_w = 10  # default
        activations = self.activations  # get_activations(state)
        layer_names = self.activationlayer_names        # Names of the layers, so you can have them as part of your plot

        self.state_memory_list.append(np.copy(state))

        display_grid = []
        non_conv_layers_amount = 0

        for idx, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):      # Displays the feature maps
            # print(f'activation layer: {idx}')
            n_features = layer_activation.shape[-1]                             # Number of features in the feature map
            # size = layer_activation.shape[1]                                    # The feature map has shape (1, size, size, n_features).
            if len(layer_activation.shape) == 4:  # eg. Convolution layer
                size_h = layer_activation.shape[1]
                size_w = layer_activation.shape[2]
                n_cols = (n_features // IMAGES_PER_ROW) + 1                               # Tiles the activation channels in this matrix

                # display_grid = np.zeros((size * n_cols, IMAGES_PER_ROW * size))
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
        shape_x = [2]
        shape_y = [2]
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
        axs_act = []    # for activations visuals
        axs_line = []   # for average activation output
        axs_st = []     # for current state

        # append empty subplots to list
        for idx in range(len(activations)):
            plot_id = idx * COL + 1
            # ax = plt.subplot(len(activations), COL, plot_id)
            # axs_act.append(ax)
            axs_act.append(self.activation_fig.add_subplot(len(activations), COL, plot_id))

            # ax = plt.subplot(len(activations), COL, plot_id + 1)
            # axs_line.append(ax)
            axs_line.append(self.activation_fig.add_subplot(len(activations), COL, plot_id + 1))

            # ax = plt.subplot(len(activations), COL, plot_id + 2)
            # axs_st.append(ax)
            axs_st.append(self.activation_fig.add_subplot(len(activations), COL, plot_id + 2))

        ims = []
        # iterate over the subplots
        # activations subplot
        for idx, ax in enumerate(axs_act):
            try:
                # ax.plot(x, y)
                im = ax.imshow(display_grid[idx], aspect='equal', cmap='seismic')  # cmap='viridis'
                ims.append(im)
                ax.set_title(f'{layer_names[idx]}')

                # set x&y limits
                ax.set_xlim(-1, max(shape_x))
                ax.set_ylim(max(shape_y), -1)

                # Change major ticks to show every x and y of the input array (creating borders).
                ax.xaxis.set_major_locator(MultipleLocator(size_w + BORDER_WIDTH))
                ax.yaxis.set_major_locator(MultipleLocator(size_h + BORDER_WIDTH))

                # Change minor ticks to show every 1
                ax.xaxis.set_minor_locator(AutoMinorLocator(size_w + BORDER_WIDTH))
                ax.yaxis.set_minor_locator(AutoMinorLocator(size_h + BORDER_WIDTH))

                ax.grid(True)  # show gridlines
            except Exception:
                # no conv layer, for example a dense layer
                pass

        # average activation outputs subplot
        for idx, ax in enumerate(axs_line):
            # ax.plot(x, y)
            x = np.array(self.activation_mean_output[idx])
            if x.size > 1:
                im, = ax.plot(x, label=turns)  # note the comma! Do no want a list, but an object. (otherwise can't create an animation)
            else:
                im, = ax.plot(x, label=turns, marker='o')
            ims.append(im)
            ax.set_title(f'{layer_names[idx]}')

            # set x&y limits
            # ax.set_xlim(-1, max(self.n_features))
            MINIMUM_Y_LIM = 0.3
            y_lim = max(MINIMUM_Y_LIM, max(self.activation_mean_output[idx]))
            ax.set_xlim(-1, self.n_features[idx] + 1)
            ax.set_ylim(min(self.activation_mean_output[idx]) - 0.1, y_lim)

            # Change major ticks to show every #.
            ax.xaxis.set_major_locator(MultipleLocator(1))
            # ax.yaxis.set_major_locator(MultipleLocator(y_lim / 5))
            ax.yaxis.set_major_locator(AutoLocator())

            # Change minor ticks to show every 1
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            # ax.yaxis.set_minor_locator(AutoMinorLocator(y_lim / 5))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.legend()
            ax.grid(True)  # show gridlines

        # state subplot
        for idx, ax in enumerate(axs_st):
            # ax.plot(x, y)
            i = len(self.state_memory_list) - 1 - idx
            if i < 0:
                break
            state = self.state_memory_list[i]
            state2 = np.array(state[:, :, 0])
            # print(state2)

            ax.set_title(f'state at turn {turns-idx}')

            # set x&y limits
            ax.set_xlim(-1, 8)
            ax.set_ylim(7, -1)

            # Change major ticks to show every x and y of the input array (creating borders).
            ax.xaxis.set_major_locator(MultipleLocator(size_w + BORDER_WIDTH))
            ax.yaxis.set_major_locator(MultipleLocator(size_h + BORDER_WIDTH))

            # Change minor ticks to show every 1
            ax.xaxis.set_minor_locator(AutoMinorLocator(size_w + BORDER_WIDTH))
            ax.yaxis.set_minor_locator(AutoMinorLocator(size_h + BORDER_WIDTH))

            ax.grid(False)  # show gridlines

            im = ax.imshow(state2, aspect='equal', cmap='autumn')
            ims.append(im)
            # break  # only show one plot

        # im = self.state_ims[len(self.state_ims)-1]
        # ims.append(im)
        if save_to_file:
            self.save_img(plot=plt, turns=turns, prefix='activation at turn')
        # plt.show()  # show plot

        # plt.close(fig)
        # self.activation_fig = fig
        # im = ax.imshow(display_grid[0], aspect='auto', cmap='viridis')
        # self.activation_ims.append(axs)
        self.activation_ims.append(ims)

    def save_img(self, plot, prefix, turns):
        path = os.getcwd()
        path = os.path.join(path, 'output', f'{time.time()}-{prefix}[{turns}].png')
        plot.savefig(path)

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
