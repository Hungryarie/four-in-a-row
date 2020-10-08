# ref:
# https://github.com/jaromiru/AI-blog/blob/master/CartPole-DQN.py
# https://github.com/keon/deep-q-learning/blob/master/ddqn.py


import random
import logging
import numpy as np
import matplotlib.pyplot as plt

import players
from game import FiarGame
from analyse import AnalyseModel


class environment(FiarGame):
    def __init__(self, *args, reward_dict):

        super().__init__()  # basic parameters (playingfield/featurnmap init)

        self.reward_dict = reward_dict

    def reset(self, start_id=None):
        """
        resets the game and
        returns the observationspace / state, reward, done, info

        inputs: start_id (optional). Give the player_id who will start. Otherwise random.
        """
        super().reset()
        #self.action_space = self.GetActionSpace()

        if start_id is not None:
            # overrule the random player start and forces the given player to start
            self.current_player = start_id  # pick a player to start
            self.current_player_value = self.getPlayerById(self.current_player).value
            self._feature_space_active_player()

        return self.get_state()

    def add_players(self, p1, p2):
        super().add_players(p1, p2)

        # self.observation_space_n = self.GetObservationSize()
        self.observation_max = self.player1.value  # equals 1
        self.observation_min = self.player2.value  # equals -1
        # self.action_space = self.GetActionSpace()
        # self.action_space_n = self.GetActionSize()

    def get_feature_size(self, enriched=True):
        if enriched:
            return self.featuremap.shape
        else:
            return self.playingField.shape

    def render(self):
        """
        renders the game
        """
        self.ShowField2()

    def env_info(self):
        print("Player info:")
        print(f" - player 1:{self.player1.name}, coin:{self.player1.color}, type:{type(self.player1)}")
        print(f" - player 2:{self.player2.name}, coin:{self.player2.color}, type:{type(self.player2)}")

    def sample(self):
        """
        returns a random action from the actionspace.
        """
        # return random.randint(0, self.action_space_n - 1)
        return np.random.choice(self.active_player.actionspace)

    def get_selfplay_action(self):
        """get the selfplay action by reversing the playingfield and using the policy from the agent who is training"""
        warnings.warn("deprecated. Is being replaced by player.SelfPlay class", DeprecationWarning)
        raise DeprecationWarning

        # print("###start###")
        # self.print_feature_space()
        self.featuremap[:, :, 0] = self.active_player.inverse_state(self.featuremap[:, :, 0])  # inverse field
        # self.enrich_feature_space()
        # print("------")
        self.featuremap[:, :, 1] = self.active_player.inverse_state(self.featuremap[:, :, 1])  # inverse players turn aswell.
        # self.print_feature_space()
        # print("###end###")
        # state = self.featuremap[np.newaxis, :, :]
        selfplay_action = self.inactive_player.select_cell(state=self.featuremap, actionspace=self.inactive_player.actionspace)
        self.featuremap[:, :, 0] = self.active_player.inverse_state(self.featuremap[:, :, 0])  # reverse field back to original game status
        self.featuremap[:, :, 1] = self.active_player.inverse_state(self.featuremap[:, :, 1])  # reverse players turn aswell.

        return selfplay_action

    def reward(self, reward_clipping=False):
        # reward = 0
        # reward_p1 = 0
        # reward_p2 = 0
        reward = self.reward_dict['step']                   # REWARD_STEP
        if self.active_player.player_id == self.player1.player_id:
            reward_p1 = self.reward_dict['step']            # REWARD_STEP
            reward_p2 = self.reward_dict['step']            # REWARD_STEP  # 0
        if self.active_player.player_id == self.player2.player_id:
            reward_p2 = self.reward_dict['step']            # REWARD_STEP
            reward_p1 = self.reward_dict['step']            # REWARD_STEP  # 0

        if self.winner == 0 and self.done is True:
            reward = self.reward_dict['tie']                # self.REWARD_TIE
            reward_p1 = self.reward_dict['tie']             # self.REWARD_TIE
            reward_p2 = self.reward_dict['tie']             # self.REWARD_TIE
        if self.winner == self.player1.value:           # player_id:
            reward = self.reward_dict['win']                # self.REWARD_WINNING
            reward_p1 = self.reward_dict['win']             # self.REWARD_WINNING
            reward_p2 = self.reward_dict['lose']            # self.REWARD_LOSING
        if self.winner == self.player2.value:           # player_id:
            reward = self.reward_dict['lose']               # self.REWARD_LOSING
            reward_p1 = self.reward_dict['lose']            # self.REWARD_LOSING
            reward_p2 = self.reward_dict['win']             # self.REWARD_WINNING
        if self._invalid_move_played:
            reward = self.reward_dict['invalid']            # self.REWARD_INVALID_MOVE
            if self.active_player.player_id == self.player1.player_id:
                reward_p1 = self.reward_dict['invalid']     # self.REWARD_INVALID_MOVE
            if self.active_player.player_id == self.player2.player_id:
                reward_p2 = self.reward_dict['invalid']     # self.REWARD_INVALID_MOVE

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

    def get_state(self):
        # print(f"featuremap size: {self.get_feature_size(self.active_player.enriched_features)}")
        # if self.enriched_features:
        if self.active_player.enriched_features:
            self.enrich_feature_space()  # enrich always (needed for training)
            return self.featuremap
        else:
            return self.playingField

    def step(self, action):
        """
        returns
        observation, reward, done, info
        """
        self.addCoin(action, coin_value=self.active_player.value)
        self.CheckGameEnd()

        state = self.get_state()

        return state, self.reward(reward_clipping=False), self.done, self.info()

    def block_invalid_moves(self, x=10):
        """
        after x attempts block the invalid moves in the action_space. Preventing the model from getting in a loop.
        """
        if isinstance(self._invalid_move_action, int) and (self._invalid_move_count > x or self.current_player == 2) and self._invalid_move_played is True:
            try:
                #self.action_space.remove(self._invalid_move_action)
                self.active_player.actionspace.remove(self._invalid_move_action)
                logging.info(f"block action:{self._invalid_move_action} from trying (by player: {self.active_player})")
            except Exception:
                logging.error(f'trying to remove item ({self._invalid_move_action}) from list ({self.active_player.actionspace}) that does not excist. (by player: {self.active_player})')

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
                print(f"> Turn: {self.active_player.name} (p{self.active_player.player_id}) (coin:{self.active_player.color})")
                print(f' > Actionspace: {self.active_player.actionspace}')

            state = self.get_state()

            action = self.active_player.select_cell(state=state, actionspace=self.active_player.actionspace)
            observation, reward, done, info = self.step(action)

            if not self._invalid_move_played and not done:
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

                # self.print_feature_space()

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
