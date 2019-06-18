# borrowed from: https://github.com/shakedzy/tic_tac_toe/blob/master/players.py
import random
import numpy as np
from abc import abstractmethod

class Player:
    """
    Base class for all player types
    """
    name = None
    player_id = None
    color = None

    def __init__(self):
        pass

    def shutdown(self):
        pass

    def add_to_memory(self, add_this):
        pass
    
    def save(self, filename):
        pass

    @abstractmethod
    def select_cell(self, board, **kwargs):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass


class Human(Player):
    """
    This player type allow a human player to play the game
    """
    name = "Ar"
    def select_cell(self, board, **kwargs):
        cell = input("Select column to fill: ")
        return cell

    def learn(self, **kwargs):
        pass

class Drunk(Player):
    """
    Drunk player always selects a random valid move
    """
    def select_cell(self, board, **kwargs):
        return random.randint(0,np.size(board,1)-1)

    def learn(self, **kwargs):
        pass


class DQNPlayer(Player):
    pass


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
        layers_size = [item for sublist in [[9],hidden_layers_size,[9]] for item in sublist]
        self.session = session
        self.model = DeepQNetworkModel(session=self.session, layers_size=layers_size,
                                       memory=ExperienceReplayMemory(memory_size),default_batch_size=learning_batch_size,
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