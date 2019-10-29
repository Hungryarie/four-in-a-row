# trying to implement:
# https://github.com/jaromiru/AI-blog/blob/master/CartPole-DQN.py
# https://github.com/keon/deep-q-learning/blob/master/ddqn.py

import numpy as np
import random
import matplotlib.pyplot as plt
import players
from game import FiarGame
from model import AnalyseModel
import logging


class enviroment(FiarGame):
    def __init__(self, *args):
        # FiarGame.__init__(self, *args)
        super().__init__(*args)
        # self.observation_space = self.GetObservationSpace()
        self.observation_space_n = self.GetObservationSize()
        self.observation_max = self.player1.player_id  # equals 1
        self.observation_min = self.player2.player_id  # equals -1
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
        renders the game
        """
        self.ShowField2()

    def env_info(self):
        print(f"player 1:{self.player1.name}, coin:{self.player1.color}, type:{type(self.player1)}")
        print(f"player 2:{self.player2.name}, coin:{self.player2.color}, type:{type(self.player2)}")

    def sample(self):
        """
        returns a random action from the actionspace.
        """
        # return random.randint(0, self.action_space_n - 1)
        return np.random.choice(self.action_space)

    def reward(self, reward_clipping=False):
        # reward = 0
        # reward_p1 = 0
        # reward_p2 = 0
        reward = self.REWARD_STEP
        if self.active_player.player_id == self.player1.player_id:
            reward_p1 = self.REWARD_STEP
            reward_p2 = self.REWARD_STEP  # 0
        if self.active_player.player_id == self.player2.player_id:
            reward_p2 = self.REWARD_STEP
            reward_p1 = self.REWARD_STEP  # 0

        if self.winner == 0 and self.done is True:
            reward = self.REWARD_TIE
            reward_p1 = self.REWARD_TIE
            reward_p2 = self.REWARD_TIE
        if self.winner == self.player1.player_id:
            reward = self.REWARD_WINNING
            reward_p1 = self.REWARD_WINNING
            reward_p2 = self.REWARD_LOSING
        if self.winner == self.player2.player_id:
            reward = self.REWARD_LOSING
            reward_p1 = self.REWARD_LOSING
            reward_p2 = self.REWARD_WINNING
        if self._invalid_move_played:
            reward = self.REWARD_INVALID_MOVE
            if self.active_player.player_id == self.player1.player_id:
                reward_p1 = self.REWARD_INVALID_MOVE
            if self.active_player.player_id == self.player2.player_id:
                reward_p2 = self.REWARD_INVALID_MOVE

        if reward_clipping:
            reward = self.reward_clipping(reward)
            reward_p1 = self.reward_clipping(reward_p1)
            reward_p2 = self.reward_clipping(reward_p2)

        return [reward, reward_p1, reward_p2]

    def reward_clipping(self, reward):
        if reward < 0:
            reward = -1
        if reward > 0:
            reward = 1
        return reward

    def step(self, action):
        """
        returns
        observation, reward, done, info
        """
        self.addCoin(action, self.active_player.player_id)
        self.CheckGameEnd()

        # return self.GetState(), self.reward(), self.done, self.info()
        return self.playingField, self.reward(reward_clipping=False), self.done, self.info()

    def block_invalid_moves(self, x=10):
        """
        after x attempts block the invalid moves in the action_space. Preventing the model from getting in a loop.
        """
        if isinstance(self._invalid_move_action, int) and self._invalid_move_count > x and self._invalid_move_played is True:
            try:
                self.action_space.remove(self._invalid_move_action)
                logging.info(f"block action:{self._invalid_move_action} from trying")
            except Exception:
                logging.error(f'trying to remove item ({self._invalid_move_action}) from list ({self.action_space}) that does not excist.')

    def info(self):
        dicti = {"active_player": self.active_player.name,
                 "moves_played": self.turns}
        return dicti

    def test(self, render=False, visualize_layers=False):
        """
        test out 1 game.
        default the render=False.
        outputs: episode reward, winner-id
        """
        ep_reward = 0
        ep_reward_p1 = 0
        ep_reward_p2 = 0
        observation, ep_reward, done = self.reset(), False, 0

        if visualize_layers:
            analyse_model = AnalyseModel()  # make analyse model of each layer
            analyse_model.update_model(self.player1.model)
            analyse_model.reset_figs()

            # visualize first turn:
            analyse_model.visual_debug_play(state=observation, turns=self.turns, print_num=True)

        while not done:
            self.block_invalid_moves(x=1)

            if render:
                print(f"> Turn: {self.active_player.name} ({self.active_player.color})")
                print(f' > Actionspace: {self.action_space}')

            action = self.active_player.select_cell(board=self.playingField, state=self.GetState(), actionspace=self.action_space)
            observation, reward, done, info = self.step(action)

            if not self._invalid_move_played:
                self.setNextPlayer()
            ep_reward += reward[0]  # reward default to player 1
            ep_reward_p1 += reward[1]   # reward for player 1
            ep_reward_p2 += reward[2]   # reward for player 2

            if render:
                print(f'chosen action: {action}')
                self.render()
                # self.playingField = self.active_player.inverse_state(self.playingField)
                # print('>> swap player colors:')
                # self.render()
                # self.playingField = self.active_player.inverse_state(self.playingField)
                print('\n')

                # print(observation)
                # print (f"reward: {reward}")

            if visualize_layers:
                analyse_model.visual_debug_play(state=observation, turns=self.turns, print_num=True)

            if done:
                if render:
                    # self.render()
                    print(self.Winnerinfo())
                    print(f"Episode finished after {self.turns} timesteps")
                    print("\n")
                if visualize_layers:
                    analyse_model.render_vid()

        return [ep_reward, ep_reward_p1, ep_reward_p2], self.winner
