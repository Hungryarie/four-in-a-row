# borrowed from: https://github.com/shakedzy/tic_tac_toe/blob/master/players.py
import random
import numpy as np
from abc import abstractmethod
import tensorflow.keras.optimizers as ko
from tensorflow.keras.utils import plot_model
#
from collections import deque
from model import ModifiedTensorBoard
import time
from constants import *


class Player:
    """
    Base class for all player types
    """
    name = None
    player_id = None
    color = None

    def __init__(self):
        self.player_class = self.__class__.__name__

    def shutdown(self):
        pass

    def add_to_memory(self, add_this):
        pass

    def save(self, filename):
        pass

    @abstractmethod
    def select_cell(self, board, state, actionspace, **kwargs):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass


class Human(Player):
    """
    This player type allow a human player to play the game
    """
    def select_cell(self, board, state, actionspace, **kwargs):
        cell = input("Select column to fill: ")
        return cell

    def learn(self, **kwargs):
        pass


class Drunk(Player):
    """
    Drunk player always selects a random valid move
    """
    def select_cell(self, board, state, actionspace, **kwargs):
        # return random.randint(0,np.size(board,1)-1)
        #return random.randint(min(actionspace), max(actionspace))
        return np.random.choice(actionspace)

    def learn(self, **kwargs):
        pass


class DDQNPlayer(Player):
    def __init__(self, model, *args):
        super().__init__(*args)
        # Main model: gets trained every step => .fit()
        self.model = model.model  # self.create_model(input_shape=(10,10,3),output_num=9)

        # Target network: this is what will get predict against every step => .predict()
        self.target_model = model.target_model  # self.create_model(input_shape=(10, 10, 3), output_num=9)
        self.target_model.set_weights(self.model.get_weights())

        self.setup = False

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def setup_for_training(self):
        """
        Setup the modified tensorboard
        Set the target update counter to zero.
        """
        if self.setup is False:
            # setup custom tensorboard object
            self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.model_name}-{int(time.time())}")

            # Used to count when to update target network with main network's weights
            self.target_update_counter = 0

            self.setup = True

    def update_replay_memory(self, transition):
        """
        Adds step's data to a memory replay array"""
        # (observation space, action, reward, new observation space, done)
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        """
        Trains main network every step during episode
        """
        self.setup_for_training()

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        # 2do: normilize function instead of /255
        current_states = np.array([transition[0] for transition in minibatch]) / 2  # 55  # transition:(observation space, action, reward, new observation space, done)
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        # 2do: normilize function instead of /255
        new_current_states = np.array([transition[3] for transition in minibatch]) / 2  # 55  # transition:(observation space, action, reward, new observation space, done)
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward  # no further max_future_q possible, because done=True

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q  # replace the current_q for the self.model.predict() action by the new_q from the self.target_model.predict() action q-value

            # And append to our training data
            X.append(current_state)  # features = state
            y.append(current_qs)    # labels = actions

        # Fit on all samples as one batch, log only on terminal state
        # normilize function instead of /255 do: /2
        self.model.fit(np.array(X) / 2, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0  # reset

    def get_prob_action(self, state):
        # 2do:
        pass
        prop_out_list = self.get_qs(state)
        # random.random()

    def get_qs(self, state):
        # Queries main network for Q values given current observation space (environment state)
        # So this is just doing a .predict(). We do the reshape because TensorFlow wants that exact explicit way to shape. The -1 just means a variable amount of this data will/could be fed through.
        # divided by 255 is to normalize is.
        # normilize function instead of /255 do: /2
        qs = self.model.predict(np.array(state).reshape(-1, *state.shape) / 2)[0]
        return qs

    def select_cell(self, board, state, actionspace, **kwargs):
        qs = self.get_qs(board)
        # overrule model probabilties according to the (modified) actionspace
        for key, prob in enumerate(qs):
            if key not in actionspace:
                qs[key] = -999  # set to low probability
        action = np.argmax(qs)
        return action


class DQNPlayer(Player):

    def __init__(self, model):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )
        # #input_shape = (6,)
        # #self.model.build(input_shape)
        # summarize layers
        # #print(model.summary())
        # plot graph
        plot_model(model, to_file='modela2ca.png')

    def train(self, env, batch_sz=32, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rews) - 1, ep_rews[-2]))

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
        return ep_rews

    def select_cell(self, board, state, actionspace, **kwargs):

        action, _ = self.model.action_value(state[None, :])
        return action

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            # print (f"action:{action}")
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value'] * kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy'] * entropy_loss


class QPlayer(Player):
    """
    A reinforcement learning agent, based on Double Deep Q Network model
    This class holds two Q-Networks: `qnn` is the learning network, `q_target` is the semi-constant network
    """
    def __init__(self, session, hidden_layers_size, gamma, learning_batch_size, batches_to_q_target_switch, tau, memory_size,
                 maximize_entropy=False, var_scope_name=None):
        """
        :param session: a tf.Session instance
        :param hidden_layers_size: an array of integers, specifying the number of layers of the network and their size
        :param gamma: the Q-Learning discount factor
        :param learning_batch_size: training batch size
        :param batches_to_q_target_switch: after how many batches (trainings) should the Q-network be copied to Q-Target
        :param tau: a number between 0 and 1, determining how to combine the network and Q-Target when copying is performed
        :param memory_size: size of the memory buffer used to keep the training set
        :param maximize_entropy: boolean, should the network try to maximize entropy over direct future rewards
        :param var_scope_name: the variable scope to use for the player
        """
        layers_size = [item for sublist in [[9], hidden_layers_size, [9]] for item in sublist]
        self.session = session
        self.model = DeepQNetworkModel(session=self.session, layers_size=layers_size,
                                       memory=ExperienceReplayMemory(memory_size), default_batch_size=learning_batch_size,
                                       gamma=gamma, double_dqn=True,
                                       learning_procedures_to_q_target_switch=batches_to_q_target_switch,
                                       tau=tau, maximize_entropy=maximize_entropy, var_scope_name=var_scope_name)
        self.session.run(tf.global_variables_initializer())
        super(QPlayer, self).__init__()

    def select_cell(self, board, **kwargs):
        return self.model.act(board, epsilon=kwargs['epsilon'])

    def learn(self, **kwargs):
        return self.model.learn(learning_rate=kwargs['learning_rate'])

    def add_to_memory(self, add_this):
        state = self.player_id * add_this['state']
        next_state = self.player_id * add_this['next_state']
        self.model.add_to_memory(state=state, action=add_this['action'], reward=add_this['reward'],
                                 next_state=next_state, is_terminal_state=add_this['game_over'])

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.session, filename)

    def restore(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.session, filename)

    def shutdown(self):
        try:
            self.session.close()
        except Exception as e:
            logging.warning('Failed to close session', e)
