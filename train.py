import logging
import random
import os
import time

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from keras.callbacks import TensorBoard

# own file imports
from logandstats import Stats, ModelLog


class TrainAgent:
    def __init__(self, environment, parameters, debug_flag=False):

        # training parameters, constants and settings
        self.para = parameters

        self.env = environment

        # step_dict is used to train on the two previous played actions (one from the opponent and one from the active player)
        self.step_dict = {}
        self.step_dict['start_state'] = [None, None, None]
        self.step_dict['mid_state'] = [None, None, None]
        self.step_dict['action'] = [None, None, None]

        # For more repetitive results
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

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
        self.log = ModelLog(log_filename, log_flag)
        # self.log.add_player_info_old(self.env.player1, self.env.player2)
        self.log.add_player_info(self.env.player1)
        self.log.add_player_info(self.env.player2)
        self.log.add_modelinfo(1, self.env.player1.actor, "actor")
        self.log.add_modelinfo(1, self.env.player1.critic, "critic")
        self.log.add_constants(self.para)
        self.log.write_to_csv()
        self.log.write_parameters_to_file()

        self.count_stats = Stats()  # set new counter class

    def setup_training(self, train_description):
        # setup tensorboard for training
        # p2modclass = str(self.log.model2_class)
        p2modclass = self.log.get_info(2, "model_class")
        p2modclass = p2modclass.replace('/', '')
        description = f"({train_description}) vs p2={self.env.player2.player_class}@{p2modclass}"
        self.env.player1.setup_for_training(description=description, dir=self.tensorboard_dir)

        self.log.log_text_to_file(f"Training description: {description}\n")

    def run_training(self, start_id=None):
        self.log.log_text_to_file(f"start training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Iterate over episodes
        for self.count_stats.episode in tqdm(range(1, self.para.EPISODES + 1), ascii=True, unit=' episodes'):

            # Restarting episode - reset episode reward
            state = self.env.reset(start_id=start_id)
            self.step_dict['mid_state'][self.env.inactive_player.player_id] = state
            done = False
            self.count_stats.episode_reward = 0

            # check if training needs to proceed or if the model got corrupted
            if self.env.player1.got_NaNs:
                self.log.log_text_to_file(f"NaN as model output at episode {self.count_stats.episode}")
                self.log.log_text_to_file(f"end training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                break  # Nan as model output. useless to continue training

            # Update tensorboard step every episode
            self.env.player1.tensorboard.step = self.count_stats.episode

            while not done:

                self.env.block_invalid_moves(x=self.para.MAX_INVALID_MOVES)  # high number to being able to actualy train upon

                # select action, train and get reward
                reward_p1, reward_p2, done = self.action_and_train()

                # for stats
                if self.env._invalid_move_played and self.env.current_player == 1:
                    self.count_stats.invalidmove_count += 1  # tensorboard stats

                if self.env.prev_invalid_move_reset and self.env.current_player == 1:
                    # only collect the succesive invalidmove count
                    self.count_stats.invalidmove_count_succesive += self.env.prev_invalid_move_count

                self.count_stats.episode_reward += reward_p1

                # render every x episodes
                if not self.env._invalid_move_played and self.para.SHOW_PREVIEW and self.count_stats.episode % self.para.AGGREGATE_STATS_EVERY == 0:
                    self.env.render()
                    print('\n')

                # change player after a valid move
                if not self.env._invalid_move_played and not done:
                    self.env.setNextPlayer()

            # episode is ended
            if self.para.SHOW_PREVIEW and self.count_stats.episode % self.para.AGGREGATE_STATS_EVERY == 0:
                # show winner info
                print(self.env.Winnerinfo())
                # print episode reward
                self.print_reward(self.count_stats.episode, self.count_stats.episode_reward)

            # collect stats for tensorboard after every episode
            self.collect_stats()

            print(f"certainty: {np.round(self.env.player1.certainty_indicator, 3)}")
            print(f"changiness: {np.round(self.env.player1.get_policy_info(), 4)}")

            # update tensorboard every x episode
            self.update_tensorboard(per_x_episode=self.para.AGGREGATE_STATS_EVERY)

            # save an image every x episode
            self.save_img(per_x_episode=self.para.AGGREGATE_STATS_EVERY)

            # save model every x episodes
            if self.count_stats.episode % 100 == 0:
                self.save_model(player=self.env.player1)

            # Decay epsilon
            if self.count_stats.epsilon > self.para.MIN_EPSILON:
                self.count_stats.epsilon *= self.para.EPSILON_DECAY
                self.count_stats.epsilon = max(self.para.MIN_EPSILON, self.count_stats.epsilon)

    def action_and_train(self):
        """defenitions:\n
        start_state = state where the action is based upon => is the mid_state of the opponent
        mid_state = new state after the chosen action is played

        """
        # ACTION PART

        # get start state = mid state of opponent
        # self.step_dict['start_state'][self.env.active_player.player_id] = np.copy(self.step_dict['mid_state'][self.env.inactive_player.player_id])
        self.step_dict['start_state'][self.env.active_player.player_id] = np.copy(self.env.featuremap)

        state = self.step_dict['start_state'][self.env.active_player.player_id]

        if not self.env.active_player.player_id == 0:
            # use exploration for player 1 and 2
            # get probability based action
            self.count_stats.tau = self.count_stats.epsilon
            #action = self.env.active_player.get_prob_action(state=state, actionspace=self.env.action_space, tau=self.count_stats.tau)
            action = self.env.active_player.get_prob_action(state=state, actionspace=self.env.action_space, tau=0.16)
        else:
            action = self.env.active_player.select_cell(state=state, actionspace=self.env.action_space)

        # next_state, [_, reward_p1, reward_p2], done, info = self.env.step(action)
        _, [_, reward_p1, reward_p2], done, info = self.env.step(action)
        next_state = np.copy(self.env.featuremap)

        reward_arr = [None, None, None]
        reward_arr[self.env.player1.player_id] = reward_p1
        reward_arr[self.env.player2.player_id] = reward_p2

        # log taken action
        self.step_dict['action'][self.env.active_player.player_id] = action
        # log state after action
        self.step_dict['mid_state'][self.env.active_player.player_id] = np.copy(next_state)
        # log next state =
        # self.step_dict['next_state'][self.env.active_player.player_id] = self.step_dict['mid_state'][self.env.inactive_player.player_id]

        # TRAINING PART

        #  after the opponents move (or on a current played invalid move).
        #  Training step is based on previous step. (because the new state is when the next player has made his move)
        if self.env._invalid_move_played:
            # train on invalid moves (current player)
            state = self.step_dict['start_state'][self.env.active_player.player_id]      # begin state
            action = self.step_dict['action'][self.env.active_player.player_id]          # action at begin state
            if self.env.active_player.enriched_features and self.env._extra_toprows > 0:
                # make a temporary invalid state, based on the current state + action (coin will be added to toprow)
                next_state = self.env.make_invalid_state(action, state)                      # make a temporary invalid state to train with
            else:
                next_state = self.step_dict['mid_state'][self.env.active_player.player_id]   # state after invalid move => is same as begin state
            reward = reward_arr[self.env.active_player.player_id]                        # reward after invalid move

            # print("\n----------\n")
            # self.env.ShowField2(field=state)
            # print(f"\naction:{action}\n")
            # self.env.ShowField2(field=next_state)
            # print(f"\nreward:{reward}\n")

            self.env.active_player.train_model(state, action, reward, next_state, done)  # train with these parameters
            # self.env.player1.train_model(state, action, reward, next_state, done)  # train with these parameters
        else:
            # train on valid moves (from previous step)
            state = self.step_dict['start_state'][self.env.inactive_player.player_id]   # begin state
            action = self.step_dict['action'][self.env.inactive_player.player_id]       # action at begin state
            next_state = np.copy(self.step_dict['mid_state'][self.env.active_player.player_id])  # state after opponents move
            next_state[:, :, 1] = self.env.active_player.inverse_state(next_state[:, :, 1])  # inverse players turn to simulate is is realy the next state and their turn.
            reward = reward_arr[self.env.inactive_player.player_id]                     # reward of the opponent after current move

            # print("\n----------\n")
            # if state is not None:
            #    self.env.ShowField2(field=state)
            # else:
            #    print("state is none")
            # print(f"\naction:{action}\n")
            # if next_state is not None:
            #    self.env.ShowField2(field=next_state)
            # else:
            #    print("next state is none")

            # only train when a previous step has been played. hence action should not be None.
            if action is not None:
                # print("\n----------\n")
                # self.env.ShowField2(field=state)
                # print(f"\naction:{action}\n")
                # self.env.ShowField2(field=next_state)

                self.env.inactive_player.train_model(state, action, reward, next_state, done)   # train with these parameters
                # self.env.player1.train_model(state, action, reward, next_state, done)   # train with these parameters

        # train one final time when done==true. Otherwise the final (most important! => win/lose) action is not being trained on.
        if done:
            """self.env.ShowField2()
            print(self.env.Winnerinfo())"""

            # (active player: winner)
            # get state, action and reward for winner.
            state = self.step_dict['start_state'][self.env.active_player.player_id]         # begin state
            action = self.step_dict['action'][self.env.active_player.player_id]             # action at begin state
            next_state = self.step_dict['mid_state'][self.env.active_player.player_id]      # state after winning move
            reward = reward_arr[self.env.active_player.player_id]                           # reward after winning move
            
            """self.env.print_feature_space(field=state)
            print(f"\naction:{action}\n")
            self.env.print_feature_space(field=next_state)
            print("\n----------\n")
            self.env.ShowField2(field=state)
            print(f"\naction:{action}\n")
            self.env.ShowField2(field=next_state)"""
            self.env.active_player.train_model(state, action, reward, next_state, done)

            # (inactive player : loser)
            # get state, action and reward for loser, because it is based on previous step it is always the losing step.
            state = self.step_dict['start_state'][self.env.inactive_player.player_id]     # begin state
            action = self.step_dict['action'][self.env.inactive_player.player_id]         # action at begin state
            next_state = np.copy(self.step_dict['mid_state'][self.env.active_player.player_id])  # state after opponents move
            next_state[:, :, 1] = self.env.active_player.inverse_state(next_state[:, :, 1])  # inverse players turn to simulate is is realy the next state and their turn.
            reward = reward_arr[self.env.inactive_player.player_id]                       # reward after winning move

            """self.env.print_feature_space(field=state)
            print(f"\naction:{action}\n")
            self.env.print_feature_space(field=next_state)
            print("\n----------\n")
            self.env.ShowField2(field=state)
            print(f"\naction:{action}\n")
            self.env.ShowField2(field=next_state)"""
            self.env.inactive_player.train_model(state, action, reward, next_state, done)

        """
        print(reward_arr)
        if done and self.env.current_player == 2:
            print("player 2 won")
            print(f"reward : {reward}")
            print(f"reward p1 : {reward_p1}")
            print(f"reward p2 : {reward_p2}")

        if done and self.env.current_player == 1:
            print("player 1 won")
            print(f"reward : {reward}")
            print(f"reward p1 : {reward_p1}")
            print(f"reward p2 : {reward_p2}")
        """
        return reward_p1, reward_p2, done

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
        actor_model_name = f'models/A2C/{timestamp}-{int(time.time())}_ep{self.count_stats.episode}_actor.model'
        critic_model_name = f'models/A2C/{timestamp}-{int(time.time())}_ep{self.count_stats.episode}_critic.model'
        player.actor.save(actor_model_name)
        player.critic.save(critic_model_name)

    def save_img(self, per_x_episode):
        if self.count_stats.episode % per_x_episode == 0:
            plt.figure(figsize=(30, 20))
            plt.ylim(-300, 150)

            ep = self.count_stats.episodes[-min(per_x_episode, len(self.count_stats.episodes)):]
            score = self.count_stats.ep_rewards[-min(per_x_episode, len(self.count_stats.ep_rewards)):]

            plt.plot(ep, score, 'b')
            try:
                plt.savefig(f"output/Afourinarow_a2c ep{self.count_stats.episode}.png")
            except Exception:
                pass
            plt.close()


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
