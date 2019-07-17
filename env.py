# trying to implement:
# https://github.com/jaromiru/AI-blog/blob/master/CartPole-DQN.py
# https://github.com/keon/deep-q-learning/blob/master/ddqn.py

import numpy as np
import random
import matplotlib.pyplot as plt
import players
from game import FiarGame


class enviroment(FiarGame):
    def __init__(self, *args):
        # FiarGame.__init__(self, *args)
        super().__init__(*args)
        # self.observation_space = self.GetObservationSpace()
        self.observation_space_n = self.GetObservationSize()
        self.observation_max = 2
        self.observation_min = 0
        self.action_space = self.GetActionSpace()
        self.action_space_n = self.GetActionSize()

    def reset(self):
        """
        resets the game and
        returns the observationspace / state, reward, done, info
        """
        super().reset()
        self.action_space = self.GetActionSpace()

        # return self.GetState() #, 0, False, None
        return self.playingField

    def render(self):
        """
        reders the game
        """
        self.ShowField2()

    def env_info(self):
        print(f"player 1:{self.player1.name}, coin:{self.player1.color}, type:{type(self.player1)}")
        print(f"player 2:{self.player2.name}, coin:{self.player2.color}, type:{type(self.player2)}")

    def sample(self):
        """
        returns a random action from the actionspace.
        """
        return random.randint(0, self.action_space_n - 1)

    def reward(self):
        reward = self.REWARD_STEP
        if self.winner == 0 and self.done is True:
            reward = self.REWARD_TIE
        if self.winner == 1:
            reward = self.REWARD_WINNING
        if self.winner == 2:
            reward = self.REWARD_LOSING
        if self._invalid_move_played:
            reward = self.REWARD_INVALID_MOVE

        return reward

    def step(self, action):
        """
        returns
        observation, reward, done, info
        """
        self.addCoin(action, self.active_player.player_id)
        self.CheckGameEnd()

        # return self.GetState(), self.reward(), self.done, self.info()
        return self.playingField, self.reward(), self.done, self.info()

    def block_invalid_moves(self, x=10):
        """
        after x attempts block the invalid moves in the action_space. Preventing the model from getting in a loop.
        """
        if isinstance(self._invalid_move_action, int) and self._invalid_move_count > x and self._invalid_move_played is True:
            try:
                self.action_space.remove(self._invalid_move_action)
                print(f"block action:{self._invalid_move_action} from trying")
            except:
                print('trying to remove item from list that does not excist.')

    def info(self):
        dicti = {"active_player": self.active_player.name,
                 "moves_played": self.turns}
        return dicti

    def test(self, render=False):
        observation, ep_reward, done = self.reset(), False, 0

        while not done:
            # action = self.sample()
            # action = self.active_player.select_cell(self.playingField)

            self.block_invalid_moves()

            action = self.active_player.select_cell(board=self.playingField, state=self.GetState(), actionspace=self.action_space)
            observation, reward, done, info = self.step(action)

            if not self._invalid_move_played:
                self.setNextPlayer()
            ep_reward += reward

            if render:
                print(f'actionspace: {self.action_space}')
                self.render()
                print('\n')
                print(f"> Turn: {self.active_player.name} ({self.active_player.color})")
                
                
                # print(observation)
                # print (f"reward: {reward}")

            if done:
                pass
                # self.render()
                # print(self.Winnerinfo())
                # print(f"Episode finished after {self.turns} timesteps")
                # print("\n")
        return ep_reward
