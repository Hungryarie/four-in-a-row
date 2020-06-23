# build up upon: https://github.com/shakedzy/tic_tac_toe/blob/master/players.py
import random
import numpy as np
from abc import abstractmethod
import tensorflow.keras.optimizers as ko
from tensorflow.keras.utils import plot_model
from collections import deque
import time
import os

# own file imports
# from constants import *
import game
import warnings
import logging
from model import empty_model
from train import ModifiedTensorBoard


class Player:
    """
    Base class for all player types
    """
    # name = None
    # player_id = None
    # color = None

    def __init__(self):
        self.player_class = self.__class__.__name__

        self.setup = False

        self.enriched_features = False

        # self.last_policy = np.zeros(7)
        self.last_argmax = 0
        self.count_argmax_shift = 0
        self.policy_counter = 0

    # def shutdown(self):
    #    pass

    # def add_to_memory(self, add_this):
    #    pass

    # def save(self, filename):
    #    pass

    def preprocess(self, state, actionspace, **kwargs):
        pass

    @abstractmethod
    def select_cell(self, state, actionspace, **kwargs):
        pass

    # @abstractmethod
    # def learn(self, **kwargs):
    #    pass
    def train_model(self, *args, **kwargs):  # state, action, reward, next_state, done
        """pass: no train_model method implemented"""
        pass

    def inverse_state(self, state):
        """swap the player_value in the playingfield/state"""
        inverse = np.array(state) * -1
        return inverse

    def setup_for_training(self, description=None, dir='logs'):
        """
        Only run this once per trainingsession.
        Not needed for evaluation of the model+agent (it will create an unnecessary tensorboard logfile).
        -Setup the modified tensorboard.
        -Set the target update counter to zero.
        """
        if self.setup is False:
            if description is not None:
                description = f" ({description})"
            else:
                description = ""
            # setup custom tensorboard object
            log_dir = os.path.normpath(os.path.join(os.getcwd(), dir, f"{self.model.model_class}-{self.model.model_name}-{self.model.timestamp}{description}"))
            self.tensorboard = ModifiedTensorBoard(log_dir=f"{log_dir}")

            # Used to count when to update target network with main network's weights
            self.target_update_counter = 0

            # flag for NaN as model outputs
            self.got_NaNs = False

            self.setup = True

    def get_prob_action(self, state, actionspace, tau=0.5, clip=(-1000, 300.)):
        """Boltzmann equation"""
        # tau=0.1
        assert len(actionspace) >= 1
        q_values = self.get_qs(state)
        tau = 0.0000000001 if tau == 0 else tau
        nb_actions = q_values.shape[0]

        clipped_val = np.clip(q_values, clip[0], clip[1])
        clipped_val = clipped_val - np.max(clipped_val)  # normalize trick
        exp = clipped_val / tau
        exp = np.clip(exp, -500, 500)  # otherwise inf when passing in too big numbers
        exp_values = np.exp(exp)

        # set probability to zero for all blocked actions
        no_action = np.arange(nb_actions).astype(int)
        no_action = np.delete(no_action, actionspace)
        exp_values[no_action] = 0

        # calculate probability
        try:
            probs = exp_values / np.sum(exp_values)
            # probs2 = np.true_divide(exp_values, np.sum(exp_values))
            action = np.random.choice(range(nb_actions), p=probs)

            self.set_policy_info(q_values)
            self.set_probability_info(probs)
            self.set_argmax_info(q_values)
            # self.print_probability_info(probs, action)

            print(f"tau:{tau}")
            for idx in range(nb_actions):
                str_action = " "
                if idx == np.argmax(probs):
                    str_action = f"{str_action}--> prob. argmax:{idx} "
                if action == idx:
                    str_action = f"{str_action}--> chosen action:{idx}"

                print(f"qs:{q_values[idx]:7.3f} -> "
                      f"clipped:{clipped_val[idx]:7.3f} -> "
                      f"exp:{exp[idx]:7.3f} -> "
                      f"exp_values: {exp_values[idx]:7.3f} -> "
                      f"probs:{probs[idx]:7.3f}"
                      f"{str_action}")

        except RuntimeWarning:
            logging.warning(f"runtime warning. chose random")
            action = np.random.choice(actionspace)
        except ValueError:
            logging.warning(f"tau:{tau} resulted in INF probabilities. chose random")
            action = np.random.choice(actionspace)

        return action

    def set_policy_info(self, policy):
        self.last_policy = policy
        self.certainty_indicator = max(policy) - min(policy)

    def set_probability_info(self, probabilities):
        self.last_probabilities = probabilities

    def set_argmax_info(self, policy):
        self.prev_argmax = self.last_argmax
        self.last_argmax = np.argmax(policy)
        self.policy_counter += 1
        if self.prev_argmax != self.last_argmax:
            self.count_argmax_shift += 1

    def get_policy_info(self):
        self.changiness = self.count_argmax_shift / self.policy_counter

        # reset
        self.policy_counter = 0
        self.count_argmax_shift = 0

        return self.changiness

    def print_probability_info(self, probs, action):
        print(f"raw q: {np.round(self.last_policy, 3)} -> argmax: {np.argmax(self.last_policy)}")
        print(f"probs: {np.round(probs, 3)} -> action: {action}")


class Human(Player):
    """
    This player type allow a human player to play the game
    """
    def select_cell(self, state, actionspace, **kwargs):
        cell = input("Select column to fill: ")
        return cell

    def get_prob_action(self, *args, **kwargs):
        logging.error("Human class has no probability action. Return random.")
        return np.random.choice(actionspace)


class Drunk(Player):
    """
    Drunk player always selects a random valid move
    """
    def select_cell(self, state, actionspace, **kwargs):
        # return random.randint(0,np.size(board,1)-1)
        # return random.randint(min(actionspace), max(actionspace))
        return np.random.choice(actionspace)

    def get_prob_action(self, *args, **kwargs):
        logging.error("Drunk class has no probability action. Return random")
        action = self.select_cell(*args, **kwargs)
        return action


class Stick(Player):
    """
    Stick player always selects the same move, until column is full.
    """
    def __init__(self):
        super().__init__()
        self.reset_column()

    def reset_column(self, actionspace=[0, 1, 2, 3, 4, 5, 6]):
        self.column = np.random.choice(actionspace)
        self.last_policy = np.zeros(7)
        self.last_policy[self.column] = 1
        self.last_probabilities = np.copy(self.last_policy)

    def preprocess(self, state, actionspace, **kwargs):
        pass

    def select_cell(self, state, actionspace, **kwargs):
        # return random.randint(0,np.size(board,1)-1)
        # return random.randint(min(actionspace), max(actionspace))
        if self.column not in actionspace:
            self.reset_column(actionspace)
        return self.column

    def get_prob_action(self, *args, **kwargs):
        # logging.error("Stick class has no probability action")
        action = self.select_cell(*args, **kwargs)
        return action

    def get_action(self, state, actionspace, **kwargs):
        # logging.error("Stick class has no probability action")
        action = self.select_cell(*args, **kwargs)
        return action


class Selfplay(Player):
    def __init__(self, player, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = player

        self.enriched_features = self.player.enriched_features

    def select_cell(self, state, actionspace, **kwargs):
        # state[:, :, 0] = self.inverse_state(state[:, :, 0])  # inverse field
        # if self.player.enriched_features:
        #    state[:, :, 1] = self.inverse_state(state[:, :, 1])  # inverse players turn aswell.
        self.last_policy = np.copy(self.get_qs(state))       # for logging purposes (tensorboard)
        self.last_probabilities = self.last_policy           # for logging purposes (tensorboard)
        action = self.player.select_cell(state, actionspace, **kwargs)
        # state[:, :, 0] = self.inverse_state(state[:, :, 0])  # reverse field back to original game status
        # if self.player.enriched_features:
        #    state[:, :, 1] = self.inverse_state(state[:, :, 1])  # reverse players turn aswell.

        return action

    def train_model(self, *args, **kwargs):
        # pass througt to player train_model method
        self.player.train_model(*args, **kwargs)

    def get_qs(self, *args, **kwargs):
        # pass througt to player get_gs method
        policy = self.player.get_qs(*args, **kwargs)
        return policy


class A2CAgent(Player):
    def __init__(self, actor_model, critic_model, discount, *args, twohead_model=None, enriched_features=True):
        """
        A2C(Advantage Actor-Critic) agent\n
        https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
        """
        super().__init__(*args)
        self.enriched_features = enriched_features

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = discount  # 0.99

        # create model for policy network
        self.actor = None
        self.critic = None
        self.twohead = None
        if actor_model is not None:
            self.actor = actor_model.model
            self.action_size = self.actor.hyper_dict['output_num']      # get size of state and action
            # model metadata
            self.model = self.actor     # for usage in setup_for_training() - > needs proper fix!
        if critic_model is not None:
            self.critic = critic_model.model
            self.value_size = self.critic.hyper_dict['output_num']      # get size of state and action
        if twohead_model is not None:
            self.twohead = twohead_model.model
            self.action_size = self.twohead.hyper_dict['output_num']  # get size of state and action
            self.value_size = 1   # get size of state and action
            # model metadata
            self.model = self.twohead     # for usage in setup_for_training() - > needs proper fix!

    def get_action(self, state, actionspace, **kwargs):
        """ using the output of policy network, pick action stochastically"""
        policy = self.get_qs(state)

        self.set_policy_info(policy)

        choise = np.random.choice(range(len(policy)), p=policy)
        if choise not in actionspace:
            choise = np.random.choice(actionspace)

        self.set_probability_info(policy)
        return choise

    def get_qs(self, state):
        state = state[np.newaxis, :, :]  # add one dimention in order to work (6,7,4) => (1,6,7,4)

        if self.actor is not None:
            policy = self.actor.predict(state).flatten()
        if self.twohead is not None:
            policy = self.twohead.predict(state)[0].flatten()   # index 0 is the policy, index 1 is the state-value
        return policy

    def select_cell(self, state, actionspace, **kwargs):

        qs = self.get_qs(state)

        self.last_policy = qs           # for logging purposes (tensorboard)
        self.last_probabilities = qs    # for logging purposes (tensorboard)

        # overrule model probabilties according to the (modified) actionspace
        for key, prob in enumerate(qs):
            if key not in actionspace:
                qs[key] = -99999  # set to low probability
        action = np.argmax(qs)
        return action

    def discounted_rewards(self, rewards):
        """Compute the discounted rewards over an episode
        (https://github.com/germain-hug/Deep-RL-Keras/blob/master/A2C/a2c.py)
        """
        discounted_r, cumul_r = np.zeros_like(rewards), 0
        for t in reversed(range(0, len(rewards))):
            cumul_r = rewards[t] + cumul_r * self.discount_factor
            discounted_r[t] = cumul_r
        return discounted_r

    def discounted_rewards_norm(self, rewards):
        """Compute the discounted rewards over an episode\n
        with normalisation
        https://www.youtube.com/watch?v=IS0V8z8HXrM (16 minute)"""

        discounted_r = np.zeros_like(rewards)
        for t in range(len(rewards)):
            cumul_r = 0
            discount = 1
            for k in range(t, len(rewards)):
                cumul_r += rewards[k] * discount
                discount *= self.discount_factor
            discounted_r[t] = cumul_r

        # scale the rewards: reinforcement baseline (algoritm) -> 
        mean = np.mean(discounted_r)
        std = np.std(discounted_r) if np.std(discounted_r) > 0 else 1
        discounted_r = (discounted_r - mean) / std
        return discounted_r

    # update policy network every episode
    def train_model_old(self, state, action, reward, next_state, done):
        # add one dimention in order to work (6,7,4) => (1,6,7,4)
        state = state[np.newaxis, :, :]
        next_state = next_state[np.newaxis, :, :]

        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))
        # advantages = self.actor.predict(state)

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0, callbacks=[self.tensorboard] if done else None)
        self.critic.fit(state, target, epochs=1, verbose=0, callbacks=[self.tensorboard] if done else None)

    # update policy network every episode
    def train_model(self, states, actions, rewards):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        #discounted_rewards = self.discounted_rewards(rewards)
        discounted_rewards = self.discounted_rewards_norm(rewards)
        states = np.array(states)
        if self.critic is not None:
            state_values = self.critic.predict(states)
        elif self.twohead is not None:
            predictions = self.twohead.predict(states)
            state_values = predictions[1]
            policy_values = predictions[0]
        #advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        advantages = discounted_rewards - np.squeeze(state_values)
        #advantages_onehot = np.zeros((len(state_values), self.action_size))
        advantages_onehot = np.copy(policy_values)
        for idx, adv in enumerate(advantages_onehot):
            action = actions[idx]
            adv[action] = advantages[idx]

        # Networks optimization
        if self.actor is not None:
            self.actor.fit(states, advantages_onehot, epochs=1, verbose=0, callbacks=[self.tensorboard])
            self.critic.fit(states, discounted_rewards, epochs=1, verbose=0, callbacks=[self.tensorboard])
        elif self.twohead is not None:
            self.twohead.fit(states, [advantages_onehot, discounted_rewards], epochs=1, verbose=0, callbacks=[self.tensorboard])


class DDQNPlayer(Player):
    def __init__(self, model, *args, enriched_features=False):
        """
        Double deep Q Network agent
        """
        super().__init__(*args)
        # Main model: gets trained every step => .fit()
        self.model = model.model  # self.create_model(input_shape=(10,10,3),output_num=9)

        # Target network: this is what will get predict against every step => .predict()
        self.target_model = model.target_model  # self.create_model(input_shape=(10, 10, 3), output_num=9)
        self.target_model.set_weights(self.model.get_weights())

        self.setup = False

        self.enriched_features = enriched_features

        # check on which player-id the model was trained
        if self.model.model_class == 'load_a_model':
            self.model_used_path = self.model.model_used_path.split('_')
            self.model_trained_on_player_id = self.find_model_player_id(self.model_used_path[2][10:], self.model.model_class)
        else:
            self.model_trained_on_player_id = None

        # An array with last n steps for training
        REPLAY_MEMORY_SIZE = 1000
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        AGGREGATE_STATS_EVERY = 50
        self.max_q_list = deque(maxlen=AGGREGATE_STATS_EVERY)
        self.delta_q_list = deque(maxlen=AGGREGATE_STATS_EVERY)

    def setup_for_training(self, description=None, dir='logs'):
        """
        @@@@@@@@@@ also found in the baseclass!!!!
        Only run this once per trainingsession.
        Not needed for using the model+agent (it will create an unnecessary tensorboard logfile).
        -Setup the modified tensorboard.
        -Set the target update counter to zero.
        """
        warnings.warn("deprecated. Is being replaced by parent class", DeprecationWarning)

        if self.setup is False:
            if description is not None:
                description = f" ({description})"
            else:
                description = ""
            # setup custom tensorboard object
            self.tensorboard = ModifiedTensorBoard(log_dir=f"{dir}/{self.model.model_class}-{self.model.model_name}-{self.model.timestamp}{description}")

            # Used to count when to update target network with main network's weights
            self.target_update_counter = 0

            # flag for NaN as model outputs
            self.got_NaNs = False

            self.setup = True

    def find_model_player_id(self, model_startstamp, model_class):
        """
        find out on which player-id the model was trained. returns on default Player 1.
        2Do: Use the log from parameters.csv and return player_id
        """
        # 2DO: find model player-id from parameters.csv and return player_id
        player_id = 1
        return player_id

    def update_replay_memory(self, transition):
        """
        Adds step's data to a memory replay array"""
        # (observation space, action, reward, new observation space, done)
        self.replay_memory.append(transition)

    def analyse_replay_memory(self):
        output = [self.replay_memory[i] for i in range(0, len(self.replay_memory))]
        rewardss = [i[2] for i in output]
        count_REWARD_WINNING = rewardss.count(game.FiarGame.REWARD_WINNING)
        count_REWARD_LOSING = rewardss.count(game.FiarGame.REWARD_LOSING)
        count_REWARD_TIE = rewardss.count(game.FiarGame.REWARD_TIE)
        count_REWARD_INVALID_MOVE = rewardss.count(game.FiarGame.REWARD_INVALID_MOVE)
        count_REWARD_STEP = rewardss.count(game.FiarGame.REWARD_STEP)

        counts = {}
        counts['win'] = count_REWARD_WINNING
        counts['lose'] = count_REWARD_LOSING
        counts['tie'] = count_REWARD_TIE
        counts['invalid move'] = count_REWARD_INVALID_MOVE
        counts['step'] = count_REWARD_STEP

        return counts

    def train_model(self, state, action, reward, next_state, done):

        self.update_replay_memory((state, action, reward, next_state, done))
        self.train(done, None)

    def train(self, terminal_state, step):
        """
        Trains main network every step during episode
        """
        # self.setup_for_training()  # already done

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        # 2do: normilize function instead of /255
        current_states = np.array([transition[0] for transition in minibatch]) / 1  # 55  # transition:(observation space, action, reward, new observation space, done)
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        # 2do: normilize function instead of /255
        new_current_states = np.array([transition[3] for transition in minibatch]) / 1  # 55  # transition:(observation space, action, reward, new observation space, done)
        future_qs_list = self.target_model.predict(new_current_states)

        if np.any(np.isnan(current_qs_list)) or np.any(np.isnan(future_qs_list)):
            self.got_NaNs = True

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                mean_future_q = np.max(future_qs_list[index])
                if reward >= 0:
                    new_q = reward + DISCOUNT * max_future_q
                elif reward < 0:
                    new_q = reward + DISCOUNT * max_future_q

                # track average maximum Q value for stats
                self.max_q_list.append(float(max_future_q))
            else:
                new_q = reward  # no further max_future_q possible, because done=True

            # Update Q value for given state
            current_qs = current_qs_list[index]
            self.delta_q_list.append(float(current_qs[action] - new_q))  # for logging and stats
            current_qs[action] = new_q  # replace the current_q for the self.model.predict() action by the new_q from the self.target_model.predict() action q-value

            # And append to our training data
            X.append(current_state)  # features = state
            y.append(current_qs)    # labels = action values

        # Fit on all samples as one batch, log only on terminal state
        # normilize function instead of /255 do: /2
        self.model.fit(np.array(X) / 1, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0  # reset

    def train2(self, terminal_state, step):
        """
        Trains main network every step during episode
        """
        # Start training only if certain number of samples is already saved
        batch_size = min(len(self.replay_memory), MINIBATCH_SIZE)
        if len(self.replay_memory) < max(MIN_REPLAY_MEMORY_SIZE, batch_size):
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        # 2do: normilize function instead of /255
        current_states = np.array([transition[0] for transition in minibatch]) / 1  # 55  # transition:(observation space, action, reward, new observation space, done)
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        # 2do: normilize function instead of /255
        new_current_states = np.array([transition[3] for transition in minibatch]) / 1  # 55  # transition:(observation space, action, reward, new observation space, done)
        future_action_list = self.model.predict(new_current_states)  # selection of action is from model
        future_qs_list = self.target_model.predict(new_current_states)

        if np.any(np.isnan(current_qs_list)) or np.any(np.isnan(future_action_list)) or np.any(np.isnan(future_qs_list)):
            self.got_NaNs = True
            print("NaN as output")
            print(f"min current state: {np.min(current_states)}")
            print(f"max current state: {np.max(current_states)}")
            print(f"min new current state: {np.min(new_current_states)}")
            print(f"max new current state: {np.max(new_current_states)}")

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                future_action = np.argmax(future_action_list[index])  # get (new-state) future action from model
                max_future_q = future_qs_list[index][future_action]   # but get the new-state actionvalue from target model
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward  # no further max_future_q possible, because done=True

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q  # replace the current_q for the self.model.predict() action by the new_q from the self.target_model.predict() action q-value

            # And append to our training data
            X.append(current_state)  # features = state
            y.append(current_qs)    # labels = action values

        # Fit on all samples as one batch, log only on terminal state
        # normilize function instead of /255 do: /2
        self.model.fit(np.array(X) / 1, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0  # reset

    def get_prob_action(self, state, actionspace, tau=0.5, clip=(-1000, 300.)):
        """Boltzmann equation
        @@@@@@@@@@ also found in the baseclass!!!!
        swap the player_value in the playingfield/state"""
        warnings.warn("deprecated. Is being replaced by parent class", DeprecationWarning)
        raise DeprecationWarning
        assert len(actionspace) >= 1
        q_values = self.get_qs(state)
        tau = 0.0000000001 if tau == 0 else tau
        nb_actions = q_values.shape[0]

        clipped_val = np.clip(q_values, clip[0], clip[1])
        exp = clipped_val / tau
        exp = np.clip(exp, -500, 500)  # otherwise inf when passing in too big numbers
        exp_values = np.exp(exp)

        # set probability to zero for all blocked actions
        no_action = np.arange(nb_actions).astype(int)
        no_action = np.delete(no_action, actionspace)
        exp_values[no_action] = 0

        # calculate probability
        try:
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(nb_actions), p=probs)
        except RuntimeWarning:
            action = np.random.choice(actionspace)
        except ValueError:
            action = np.random.choice(actionspace)

        return action

    def inverse_state(self, state):
        """
        @@@@@@@@@@ also found in the baseclass!!!!
        swap the player_value in the playingfield/state"""
        warnings.warn("deprecated. Is being replaced by parent class", DeprecationWarning)
        raise DeprecationWarning

        inverse = np.array(state) * -1
        return inverse

    def get_qs(self, state):
        # Queries main network for Q values given current observation space (environment state)
        # So this is just doing a .predict(). We do the reshape because TensorFlow wants that exact explicit way to shape. The -1 just means a variable amount of this data will/could be fed through.
        # divided by 255 is to normalize is.
        # normilize function instead of /255 do: /2

        # inverse state when using a loaded model AND current player-id isn't equal to on which player-id the loaded model was binded to.
        if self.model.model_class == 'load_a_model' and self.player_id != self.model_trained_on_player_id:
            # call function to swap the player_value's in the state
            state = self.inverse_state(state)

        qs = self.model.predict(np.array(state).reshape(-1, *state.shape) / 1)[0]
        return qs

    def select_cell(self, state, actionspace, **kwargs):
        qs = self.get_qs(state)
        # overrule model probabilties according to the (modified) actionspace
        for key, prob in enumerate(qs):
            if key not in actionspace:
                qs[key] = -99999  # set to low probability
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
