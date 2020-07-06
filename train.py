import logging
import random
import os
import time

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from tensorflow.keras.callbacks import TensorBoard

# own file imports
from logandstats import Stats, ModelLog
from analyse import AnalyseModel


class TrainAgent:
    def __init__(self, environment, parameters, debug_flag=False):

        # training parameters, constants and settings
        self.para = parameters

        self.env = environment

        # step_dict is used to train on the two previous played actions (one from the opponent and one from the active player)
        self.step_dict = {}
        self.step_dict['start_state'] = [None, [], []]
        self.step_dict['mid_state'] = [None, [], []]
        self.step_dict['action'] = [None, [], []]
        self.step_dict['reward'] = [None, [], []]

        # debug
        self.debug = debug_flag
        if self.debug:
            log_flag = True
            log_filename = 'parameters(debug)'
            self.tensorboard_dir = 'logs(debug)'
            logging.basicConfig(level=logging.DEBUG)
        else:
            log_flag = True
            log_filename = 'parameters'
            self.tensorboard_dir = 'logs'
            logging.basicConfig(level=logging.WARNING)

        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')

        # for stats
        self.setup_logging(log_filename, log_flag)

        # set new counter class
        self.count_stats = Stats()

    def setup_logging(self, log_filename, log_flag):
        self.log = ModelLog(log_filename, log_flag)
        self.log.add_player_info(self.env.player1)
        self.log.add_player_info(self.env.player2)

        for model_name, model in self.env.player1.models_dic.items():
            if type(model) == tf.python.keras.engine.training.Model or type(model) == tf.python.keras.engine.training_v1.Model:
                self.log.add_modelinfo(1, model, model_name)

        """
        if hasattr(self.env.player1, 'actor'):
            if self.env.player1.actor is not None:
                self.log.add_modelinfo(1, self.env.player1.actor, "actor")
                self.log.add_modelinfo(1, self.env.player1.critic, "critic")
        if hasattr(self.env.player1, 'twohead'):
            if self.env.player1.twohead is not None:
                self.log.add_modelinfo(1, self.env.player1.twohead, "twohead")
        if hasattr(self.env.player1, 'model'):
            self.log.add_modelinfo(1, self.env.player1.model, "model")
        if hasattr(self.env.player1, 'policy_model'):
            self.log.add_modelinfo(1, self.env.player1.policy_model, "policy_model")
        if hasattr(self.env.player1, 'predict_model'):
            self.log.add_modelinfo(1, self.env.player1.predict_model, "predict_model")"""

        self.log.add_constants(self.para)
        self.log.write_to_csv()
        self.log.write_parameters_to_file()

        """if hasattr(self.env.player1, 'actor'):
            if self.env.player1.actor is not None:
                self.log.plot_model(self.env.player1.actor, 'actor')
                self.log.print_model_summary(self.env.player1.actor, 'actor')
        if hasattr(self.env.player1, 'critic'):
            if self.env.player1.critic is not None:
                self.log.plot_model(self.env.player1.critic, 'critic')
                self.log.print_model_summary(self.env.player1.critic, 'critic')
        if hasattr(self.env.player1, 'twohead'):
            if self.env.player1.twohead is not None:
                self.log.plot_model(self.env.player1.twohead, 'twohead')
                self.log.print_model_summary(self.env.player1.twohead, 'twohead')
        if hasattr(self.env.player1, 'model'):
            self.log.plot_model(self.env.player1.model, 'model')
            self.log.print_model_summary(self.env.player1.model, 'model')
        if hasattr(self.env.player1, 'policy_model'):
            self.log.plot_model(self.env.player1.policy_model, 'policy_model')
            self.log.print_model_summary(self.env.player1.policy_model, 'policy_model')
        if hasattr(self.env.player1, 'predict_model'):
            self.log.plot_model(self.env.player1.predict_model, 'predict_model')
            self.log.print_model_summary(self.env.player1.predict_model, 'predict_model')"""

        for model_name, model in self.env.player1.models_dic.items():
            if type(model) == tf.python.keras.engine.training.Model or type(model) == tf.python.keras.engine.training_v1.Model:
                self.log.plot_model(model, model_name)
                self.log.print_model_summary(model, model_name)

    def setup_training(self, train_description):
        # setup tensorboard for training
        # p2modclass = str(self.log.model2_class)
        p2modclass = self.log.get_info(2, "model_class")
        p2modclass = p2modclass.replace('/', '')
        description = f"({train_description}) vs p2={self.env.player2.player_class}@{p2modclass}"
        self.env.player1.setup_for_training(
            description=description, dir=self.tensorboard_dir)

        self.log.log_text_to_file(f"Training description: {description}\n")

        if self.debug:
            self.analyse_model = AnalyseModel()  # make analyse model of each layer
            self.analyse_model.set_analyze_layers([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
            self.analyse_model.update_model(self.env.player1.model)
            self.analyse_model.reset_figs()

    def _preset_episode(self, start_id):
        self.step_dict['start_state'] = [None, [], []]
        self.step_dict['mid_state'] = [None, [], []]
        self.step_dict['action'] = [None, [], []]
        self.step_dict['reward'] = [None, [], []]
        state = self.env.reset(start_id=start_id)
        self.step_dict['mid_state'][self.env.inactive_player.player_id].append(state)

        self.count_stats.episode_reward = 0

        # Update tensorboard step every episode
        self.env.player1.tensorboard.step = self.count_stats.episode

    def _episode_review(self, per_x_episode, done=False):
        """analyse and render every x episodes"""
        if not self.env._invalid_move_played and self.para.SHOW_PREVIEW and self.count_stats.episode % per_x_episode == 0:
            self.env.render()
            print('\n')
            if done:
                print(self.env.Winnerinfo())                        # show winner info
                self.print_reward(self.count_stats.episode,
                                  self.count_stats.episode_reward)  # print episode reward
                save_to_file = True
            else:
                save_to_file = False

            if self.debug:
                # analyse model graphically
                if self.env.current_player == 1 or done:
                    self.analyse_model.update_model(self.env.player1.model)
                    self.analyse_model.visual_debug_train(state=self.env.featuremap, turns=self.env.turns,
                                                          save_to_file=save_to_file, print_num=False,
                                                          prefix=f'actor @ episode {self.count_stats.episode}')
                if done:
                    if hasattr(self.env.player1, 'critic'):
                        if self.env.player1.critic is not None:
                            self.analyse_model.reset_figs()
                            self.analyse_model.update_model(self.env.player1.critic)
                            self.analyse_model.visual_debug_train(state=self.env.featuremap, turns=self.env.turns,
                                                                  save_to_file=True, print_num=False,
                                                                  prefix=f'critic @ episode {self.count_stats.episode}')
                    self.analyse_model.reset_figs()

    def _in_episode_stat_update(self):
        # for stats
        if self.env.active_player.player_id == 1:
            self.env.player1.tensorboard.update_hist(p1_policyCentainty=self.env.player1.last_policy)
            self.env.player1.tensorboard.update_hist(p1_probabilities=self.env.player1.last_probabilities)
        else:
            self.env.player1.tensorboard.update_hist(p2_policyCentainty=self.env.player2.last_policy)
            self.env.player1.tensorboard.update_hist(p2_probabilities=self.env.player2.last_probabilities)

        if self.env._invalid_move_played and self.env.current_player == 1:
            self.count_stats.invalidmove_count += 1  # tensorboard stats

        if self.env.prev_invalid_move_reset and self.env.current_player == 1:
            # only collect the succesive invalidmove count
            self.count_stats.invalidmove_count_succesive += self.env.prev_invalid_move_count

    def run_training(self, start_id=None):
        self.log.log_text_to_file(
            f"start training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Iterate over episodes
        for self.count_stats.episode in tqdm(range(1, self.para.EPISODES + 1), ascii=True, unit=' episodes'):

            # Restarting episode - preset / reset
            self._preset_episode(start_id=start_id)
            done = False

            # check if training needs to proceed or if the model got corrupted
            if self.env.player1.got_NaNs:
                self.log.log_text_to_file(
                    f"NaN as model output at episode {self.count_stats.episode}")
                self.log.log_text_to_file(
                    f"end training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                break  # Nan as model output. useless to continue training

            while not done:

                self.env.block_invalid_moves(x=self.para.MAX_INVALID_MOVES)  # high number to being able to actualy train upon

                # select action, step, get reward (and train)
                reward_p1, reward_p2, done = self.action_and_train(train_on_step=True)

                # for tensorboard logging etc
                self._in_episode_stat_update()
                self.count_stats.episode_reward += reward_p1

                # analyse and render every x episodes
                self._episode_review(per_x_episode=self.para.AGGREGATE_STATS_EVERY)

                # change player after a valid move
                if not self.env._invalid_move_played and not done:
                    self.env.setNextPlayer()

            # episode is ended
            # TRAININGPART
            # self.visualize_state_action_rewards(self.step_dict['start_state'][1], self.step_dict['action'][1], self.step_dict['reward'][1])
            # self.visualize_state_action_rewards(self.step_dict['start_state'][2], self.step_dict['action'][2], self.step_dict['reward'][2])
            """states_adj, actions_adj, rewards_adj = self.correct_states_actions_rewards(self.step_dict['start_state'][1],
                                                                                       self.step_dict['action'][1],
                                                                                       self.step_dict['reward'][1])"""
            """states_adj2, actions_adj2, rewards_adj2 = self.correct_states_actions_rewards(self.step_dict['start_state'][2],
                                                                                          self.step_dict['action'][2],
                                                                                          self.step_dict['reward'][2])
            states_adj = states_adj + states_adj2
            actions_adj = actions_adj + actions_adj2
            rewards_adj = rewards_adj + rewards_adj2"""
            """self.env.player1.train_model(states_adj, actions_adj, rewards_adj)"""

            # analyse / review every x episodes
            self._episode_review(per_x_episode=self.para.AGGREGATE_STATS_EVERY, done=True)

            # collect stats for tensorboard after every episode
            self.collect_stats()

            # print(f"certainty: {np.round(self.env.player1.certainty_indicator, 3)}")
            # print(f"changiness: {np.round(self.env.player1.get_policy_info(), 4)}")

            # update tensorboard every x episode
            self.update_tensorboard(per_x_episode=self.para.AGGREGATE_STATS_EVERY)
            # update tensorboard histogrammen
            self.env.player1.tensorboard.update_hist(p1_column=self.step_dict['action'][1], p2_column=self.step_dict['action'][2])

            # save an image every x episode
            self.save_img(per_x_episode=self.para.AGGREGATE_STATS_EVERY)

            # save model every x episodes
            if self.count_stats.episode % 100 == 0:
                self.save_model(player=self.env.player1)

            # Decay epsilon
            if self.count_stats.epsilon > self.para.MIN_EPSILON:
                self.count_stats.epsilon *= self.para.EPSILON_DECAY
                self.count_stats.epsilon = max(self.para.MIN_EPSILON, self.count_stats.epsilon)
            # Decay tau
            if self.count_stats.tau > self.para.MIN_TAU:
                self.count_stats.tau *= self.para.TAU_DECAY
                self.count_stats.tau = max(self.para.MIN_TAU, self.count_stats.tau)

    def action_and_train(self, train_on_step):
        """defenitions:\n
        start_state = state where the action is based upon => is the mid_state of the opponent
        mid_state = new state after the chosen action is played

        """
        # ACTION PART

        # get start state = mid state of opponent
        self.step_dict['start_state'][self.env.active_player.player_id].append(np.copy(self.env.featuremap))
        state = self.step_dict['start_state'][self.env.active_player.player_id][-1]
        if self.debug:
            pass
            # self.env.print_feature_space(field=state)

        if self.env.active_player.player_id == 1:
            # use exploration for player 1
            # get probability based action
            # self.count_stats.tau = self.count_stats.epsilon
            # action = self.env.active_player.get_prob_action(state=state, actionspace=self.env.action_space, tau=self.count_stats.tau)
            action = self.env.active_player.get_action(state=state, actionspace=self.env.action_space)
        else:
            action = self.env.active_player.select_cell(state=state, actionspace=self.env.action_space)

        # next_state, [_, reward_p1, reward_p2], done, info = self.env.step(action)
        _, [_, reward_p1, reward_p2], done, info = self.env.step(action)
        next_state = np.copy(self.env.featuremap)

        # log reward
        reward_arr = [None, None, None]
        reward_arr[self.env.player1.player_id] = reward_p1
        reward_arr[self.env.player2.player_id] = reward_p2
        self.step_dict['reward'][self.env.active_player.player_id].append(reward_arr[self.env.active_player.player_id])

        # log taken action
        self.step_dict['action'][self.env.active_player.player_id].append(action)

        # log state after action
        self.step_dict['mid_state'][self.env.active_player.player_id].append(np.copy(next_state))
        # log next state =
        # self.step_dict['next_state'][self.env.active_player.player_id] = self.step_dict['mid_state'][self.env.inactive_player.player_id]

        # TRAINING PART

        #  after the opponents move (or on a current played invalid move).
        #  Training step is based on previous step. (because the new state is when the next player has made his move)
        if self.env._invalid_move_played:
            # train on invalid moves (current player)
            # begin = mid state
            state = self.step_dict['mid_state'][self.env.active_player.player_id][-1]
            # action at begin state
            action = self.step_dict['action'][self.env.active_player.player_id][-1]
            if self.env.active_player.enriched_features and self.env._extra_toprows > 0:
                # make a temporary invalid state, based on the current state + action (coin will be added to toprow)
                next_state = self.env.make_invalid_state(action, state)
                # replace log state after action
                self.step_dict['mid_state'][self.env.active_player.player_id][-1] = np.copy(next_state)
            else:
                # state after invalid move => is same as begin state
                next_state = self.step_dict['mid_state'][self.env.active_player.player_id][-1]
            # reward after invalid move
            reward = self.step_dict['reward'][self.env.active_player.player_id][-1]

            print("\n----------\n")
            self.env.ShowField2(field=state)
            print(f"\naction:{action}\n")
            self.env.ShowField2(field=next_state)
            print(f"\nreward:{reward}\n")

            if train_on_step:
                self.env.active_player.train_model(state, action, reward, next_state, done)  # train with these parameters
                self.env.active_player.train_model_step(state, action, reward, next_state, done)  # train with these parameters
                # self.env.player1.train_model(state, action, reward, next_state, done)  # train with these parameters

                self.env.active_player.print_policy_info(state, title="after training", run_this=self.debug)
        elif not done and train_on_step:
            # train on valid moves (from previous step)
            try:
                # begin state
                state = self.step_dict['start_state'][self.env.inactive_player.player_id][-1]
                # action at begin state
                action = self.step_dict['action'][self.env.inactive_player.player_id][-1]
                # reward of the opponent after current move
                reward = reward_arr[self.env.inactive_player.player_id]
                reward = self.step_dict['reward'][self.env.inactive_player.player_id][-1]
            except IndexError:
                state = None
                action = None
                reward = None

            # state after opponents move
            next_state = np.copy(
                self.step_dict['mid_state'][self.env.active_player.player_id][-1])
            # inverse players turn to simulate is is realy the next state and their turn.
            next_state[:, :, 1] = self.env.active_player.inverse_state(
                next_state[:, :, 1])

            if self.debug:
                print("\n----------\n")
                print(f"Inactive player (id:{self.env.inactive_player.player_id}, value:{self.env.inactive_player.value}) (DONE=false) ")
                if state is not None:
                    self.env.ShowField2(field=state)
                else:
                    print("state is none")
                print(f"action:{action}")
                if next_state is not None:
                    self.env.ShowField2(field=next_state)
                else:
                    print("next state is none")
                print(f"reward:{reward}")

                self.env.inactive_player.print_policy_info(state, title="before training", run_this=False)

            # only train when a previous step has been played. hence action should not be None.
            if action is not None:
                # print("\n----------\n")
                # self.env.ShowField2(field=state)
                # print(f"\naction:{action}\n")
                # self.env.ShowField2(field=next_state)
                self.env.inactive_player.train_model(state, action, reward, next_state, done)   # train with these parameters
                self.env.inactive_player.train_model_step(state, action, reward, next_state, done)   # train with these parameters
                # self.env.player1.train_model(state, action, reward, next_state, done)   # train with these parameters

            self.env.inactive_player.print_policy_info(state, title="after training", run_this=self.debug)

        # train one final time when done==true. Otherwise the final (most important! => win/lose) action is not being trained on / logged.
        if done:
            # add loosing reward to step_dict
            self.step_dict['reward'][self.env.inactive_player.player_id].append(reward_arr[self.env.inactive_player.player_id])

            if train_on_step:
                # self.env.ShowField2()
                if self.debug:
                    self.env.render()
                    print(self.env.Winnerinfo())

                # (active player: winner)
                # get state, action and reward for winner.
                # begin state
                state = self.step_dict['start_state'][self.env.active_player.player_id][-1]
                # action at begin state
                action = self.step_dict['action'][self.env.active_player.player_id][-1]
                # state after winning move
                next_state = self.step_dict['mid_state'][self.env.active_player.player_id][-1]
                # reward after winning move
                # reward = reward_arr[self.env.active_player.player_id]
                reward = self.step_dict['reward'][self.env.active_player.player_id][-1]

                if self.debug:
                    print("\n----------\n")
                    print(f"ACTIVE PLAYER (DONE=true) (id:{self.env.active_player.player_id}, value:{self.env.active_player.value})")
                    # self.env.print_feature_space(field=state)
                    # print(f"action:{action}")
                    # self.env.print_feature_space(field=next_state)
                    # print("\n----------\n")
                    self.env.ShowField2(field=state)
                    print(f"action:{action}")
                    self.env.ShowField2(field=next_state)
                    print(f"reward:{reward}")

                self.env.active_player.train_model(state, action, reward, next_state, done)
                self.env.active_player.train_model_step(state, action, reward, next_state, done)
                # self.env.player1.train_model(state, action, reward, next_state, done)   # train with these parameters

                self.env.active_player.print_policy_info(state, title="after training", run_this=self.debug)

                # (inactive player : loser)
                # get state, action and reward for loser, because it is based on previous step it is always the losing step.
                # begin state
                state = self.step_dict['start_state'][self.env.inactive_player.player_id][-1]
                # action at begin state
                action = self.step_dict['action'][self.env.inactive_player.player_id][-1]
                # state after opponents move
                next_state = np.copy(
                    self.step_dict['mid_state'][self.env.active_player.player_id][-1])
                # inverse players turn to simulate is is realy the next state and their turn.
                next_state[:, :, 1] = self.env.active_player.inverse_state(
                    next_state[:, :, 1])
                # reward after winning move
                # reward = reward_arr[self.env.inactive_player.player_id]
                reward = self.step_dict['reward'][self.env.inactive_player.player_id][-1]

                if self.debug:
                    print("\n----------\n")
                    print(f"INACTIVE PLAYER (DONE=true) (id:{self.env.inactive_player.player_id}, value:{self.env.inactive_player.value})\n")
                    # self.env.print_feature_space(field=state)
                    # print(f"action:{action}")
                    # self.env.print_feature_space(field=next_state)
                    # print("\n----------\n")
                    self.env.ShowField2(field=state)
                    print(f"action:{action}")
                    self.env.ShowField2(field=next_state)
                    print(f"reward:{reward}")

                self.env.inactive_player.print_policy_info(state, title="before training", run_this=False)

                self.env.inactive_player.train_model(state, action, reward, next_state, done)
                self.env.inactive_player.train_model_step(state, action, reward, next_state, done)
                # self.env.player1.train_model(state, action, reward, next_state, done)   # train with these parameters

                self.env.inactive_player.print_policy_info(state, title="after training", run_this=self.debug)

        return reward_p1, reward_p2, done

    def visualize_state_action_rewards(self, states, actions, rewards):

        states_adj, actions_adj, rewards_adj = self.correct_states_actions_rewards(states, actions, rewards)

        print(f'HISTORY of player: {states[0][0,0,1]}')
        for idx, (state, action, reward) in enumerate(zip(states_adj, actions_adj, rewards_adj)):
            print("\n----------")
            print(f"begin state step:{idx}")
            self.env.ShowField2(field=state)
            print(f"new action:{action}")
            print(f"gives reward:{reward}")

        print("end of episode")

    def correct_states_actions_rewards(self, states, actions, rewards):
        """correct the index of the gathered states, actions and rewards."""
        min_len = min(len(states), len(actions), len(rewards))

        states = states[-min_len:]
        actions = actions[-min_len:]
        if rewards[-1] == self.para.reward_dict['lose']:
            rewards_adj = rewards[-(min_len + 1):-2]
            rewards_adj.append(rewards[-1])
        else:
            rewards_adj = rewards[-min_len:]

        return states, actions, rewards_adj

    def update_tensorboard(self, per_x_episode):
        if self.count_stats.episode % per_x_episode == 0:
            #  Calculate over stats
            self.count_stats.aggregate_stats(calc_steps=per_x_episode)
            # update tensorboard
            self.env.player1.tensorboard.update_stats(reward_avg=self.count_stats.average_reward, reward_min=self.count_stats.min_reward, reward_max=self.count_stats.max_reward,
                                                      epsilon=self.count_stats.epsilon, invalidmove_count_succesive=self.count_stats.invalidmove_count_succesive,
                                                      win_count=self.count_stats.win_count, loose_count=self.count_stats.loose_count, draw_count=self.count_stats.draw_count,
                                                      invalidmove_count=self.count_stats.invalidmove_count, win_ratio=self.count_stats.win_ratio,
                                                      turns_count=self.count_stats.turns_count, count_horizontal=self.count_stats.count_horizontal,
                                                      count_vertical=self.count_stats.count_vertical, count_dia_left=self.count_stats.count_dia_left,
                                                      count_dia_right=self.count_stats.count_dia_right, reward_std=self.count_stats.std_reward,
                                                      tau=self.count_stats.tau)

            self.env.player1.tensorboard.update_hist(rewards=self.count_stats.ep_rewards)
            # reset stats
            self.count_stats.reset_stats()

    def collect_stats(self):
        self.count_stats.turns_count += self.env.turns

        if self.env.winner == self.env.player1.value:
            self.count_stats.win_count += 1
            if self.env.winnerhow == "Horizontal":
                self.count_stats.count_horizontal += 1
            if self.env.winnerhow == "Vertical":
                self.count_stats.count_vertical += 1
            if self.env.winnerhow == "Diagnal Right":
                self.count_stats.count_dia_right += 1
            if self.env.winnerhow == "Diagnal Left":
                self.count_stats.count_dia_left += 1
        elif self.env.winner == self.env.player2.value:
            self.count_stats.loose_count += 1
        else:
            self.count_stats.draw_count += 1

        # Append episode reward to a list (for ghraphing)
        self.count_stats.ep_rewards.append(self.count_stats.episode_reward)
        self.count_stats.episodes.append(self.count_stats.episode)

    def print_reward(self, episode, episode_reward):
        print("episode:", episode, "  episode_reward:", episode_reward)

    def save_model(self, player):
        timestamp = player.model.timestamp

        for model_name, model in player.models_dic.items():
            if type(model) == tf.python.keras.engine.training.Model or type(model) == tf.python.keras.engine.training_v1.Model:
                new_model_name = f'models/A2C/{timestamp}-{int(time.time())}_ep{self.count_stats.episode}_{model_name}.model'
                path = os.path.normpath(os.path.join(os.getcwd(), new_model_name))
                model.save(path)

        """if player.actor is not None:
            actor_model_name = f'models/A2C/{timestamp}-{int(time.time())}_ep{self.count_stats.episode}_actor.model'
            critic_model_name = f'models/A2C/{timestamp}-{int(time.time())}_ep{self.count_stats.episode}_critic.model'

            actor_path = os.path.normpath(os.path.join(os.getcwd(), actor_model_name))
            critic_path = os.path.normpath(os.path.join(os.getcwd(), critic_model_name))

            player.actor.save(actor_path)
            player.critic.save(critic_path)
        elif player.twohead is not None:
            twohead_model_name = f'models/A2C/{timestamp}-{int(time.time())}_ep{self.count_stats.episode}_twohead.model'
            twohead_path = os.path.normpath(os.path.join(os.getcwd(), twohead_model_name))
            player.twohead.save(twohead_path)"""

    def save_img(self, per_x_episode):
        if self.count_stats.episode % per_x_episode == 0:
            plt.figure(figsize=(30, 20))
            plt.ylim(4 * self.para.reward_dict['lose'], 2 * self.para.reward_dict['win'])

            ep = self.count_stats.episodes[-min(per_x_episode,
                                                len(self.count_stats.episodes)):]
            score = self.count_stats.ep_rewards[-min(
                per_x_episode, len(self.count_stats.ep_rewards)):]

            plt.plot(ep, score, 'b')
            try:
                plt.savefig(
                    f"output/{time.time()}-Afourinarow_a2c ep{self.count_stats.episode}.png")
            except Exception:
                pass
            plt.close()


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self._log_write_dir = self.log_dir  # argument self._log_write_dir is needed. so
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)
        self.update_hist(**logs)

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
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(name=key, data=value, step=self.step)
            # self.writer.flush()

    def update_hist(self, **hist):
        with self.writer.as_default():
            for key, value in hist.items():
                tf.summary.histogram(name=key, data=value, step=self.step)
