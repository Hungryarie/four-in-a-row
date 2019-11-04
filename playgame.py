import numpy as np
import random
import matplotlib.pyplot as plt
import pylab
import players
from game import FiarGame
from env import enviroment
from model import ModelLog, load_a_model, model1, model1b, model1c, model1d, model2, model3, model4a, model4b, model5
from model import func_model1, func_model_duel1b, func_model_duel1b1, func_model_duel1b2, func_model5, func_model5_duel1  # functional API specific
from model import AnalyseModel
from keras.optimizers import Adam, SGD, RMSprop
from tqdm import tqdm
from constants import *
import os
import tensorflow as tf
import time
from datetime import datetime
import logging
import sys


class Stats():
    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        self.win_ratio = 0
        self.win_count = 0
        self.loose_count = 0
        self.draw_count = 0
        self.invalidmove_count = 0
        self.turns_count = 0
        self.count_horizontal = 0
        self.count_vertical = 0
        self.count_dia_right = 0
        self.count_dia_left = 0
        self.ep_rewards = []  # [-20]
        self.max_q_list = []

    def aggregate_stats(self, calc_steps):
        self.win_ratio = self.win_count / (self.win_count + self.loose_count)
        self.turns_count = self.turns_count / calc_steps
        self.count_horizontal = self.count_horizontal / calc_steps
        self.count_vertical = self.count_vertical / calc_steps
        self.count_dia_left = self.count_dia_left / calc_steps
        self.count_dia_right = self.count_dia_right / calc_steps
        self.average_reward = sum(self.ep_rewards[-calc_steps:]) / len(self.ep_rewards[-calc_steps:])
        self.min_reward = min(self.ep_rewards[-calc_steps:])
        self.max_reward = max(self.ep_rewards[-calc_steps:])
        self.std_reward = np.std(self.ep_rewards[-calc_steps:])


def trainA2C():

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    p1 = players.A2CAgent('', '')
    p1.name = "A2C on training"
    Model = load_a_model('models/A2C/1572262959_ep63_actor.model')
    #p2 = players.Drunk()
    #p2.name = "drunk"
    p2 = players.DDQNPlayer(Model)
    p2.name = "dqn"

    env = enviroment(p1, p2)

    scores, episodes = [0,0,0,0,0], [0,0,0,0,0]  # start with 5 zero's otherwise when calculate if the mean for saving goes wrong.

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        done = False
        score = 0
        state = env.reset()
        #state = np.reshape(state, [1, state_size])

        while not done:

            env.block_invalid_moves(x=10)

            if env.current_player == 1:
                state3 = state[np.newaxis, :, :]
                #action = env.player1.get_action(state3)
                action = env.player1.select_cell(state=state3, actionspace=env.action_space)
                next_state, [_, reward_p1, _], done, info = env.step(action)
                if done or env._invalid_move_played:
                    state3 = state[np.newaxis, :, :]
                    next_state3 = next_state[np.newaxis, :, :]
                    env.player1.train_model(state3, action, reward_p1, next_state3, done)
            else:
                #state3 = state[np.newaxis, :, :]

                action_opponent = env.get_selfplay_action()

                next_state, [_, reward_p1, _], done, info = env.step(action_opponent)
                if not env._invalid_move_played:
                    state3 = state[np.newaxis, :, :]
                    next_state3 = next_state[np.newaxis, :, :]
                    env.player1.train_model(state3, action, reward_p1, next_state3, done)

            if not env._invalid_move_played:
                if env.player1.render:
                    env.render()
                env.setNextPlayer()

            score += reward_p1
            state = next_state

            if done:
                # every episode, plot the play time
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                try:
                    pylab.savefig(f"output/fourinarow_a2c.png")
                except Exception:
                    pass
                print("episode:", episode, "  score:", score)

                if episode % 100 == 0:
                    actor_model_name = f'models/A2C/{int(time.time())}_ep{episode}_actor.model'
                    critic_model_name = f'models/A2C/{int(time.time())}_ep{episode}_critic.model'
                    env.player1.actor.save(actor_model_name)
                    env.player1.critic.save(critic_model_name)
                # if the mean of scores of last 20 episode is bigger than 190
                # stop training
                if np.mean(scores[-min(20, len(scores)):]) > 190:
                    actor_model_name = f'models/A2C/{int(time.time())}_ep{episode}_actor.model'
                    critic_model_name = f'models/A2C/{int(time.time())}_ep{episode}_critic.model'
                    env.player1.actor.save(actor_model_name)
                    env.player1.critic.save(critic_model_name)
                    sys.exit()


def trainNN(p1_model=None, p2_model=None, log_flag=True, visualize_layers=False, debug_flag=False):
    # For stats
    epsilon = 1
    tau = 0

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    if debug_flag:
        log_flag = True
        log_filename = 'parameters(debug)'
        tensorboard_dir = 'logs(debug)'
        logging.basicConfig(level=logging.DEBUG)
    else:
        log_filename = 'parameters'
        tensorboard_dir = 'logs'
        logging.basicConfig(level=logging.WARNING)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    if p1_model is None:
        # default model to use is:...
        p1_model = model5(input_shape=(6, 7, 1), output_num=7)  # (7, 6, 1)(1, 42)
        # p1_model = load_a_model('models\dense2x128(softmax)_startstamp1567090369_episode9050__170.00max__152.60avg___95.00min__1567092303.model')
    p1 = players.DDQNPlayer(p1_model)
    p1.name = "DDQN on training"

    if p2_model is None:
        # p2_model = load_a_model('models\dense2x128(softmax)_startstamp1567090369_episode9050__170.00max__152.60avg___95.00min__1567092303.model')
        # p2_model = load_a_model('models\model4a_dense2x128(softmax)(flattenLAST,input_shape bug gone lr=0.001)_startstamp1568020820_episode8350__170.00max__141.60avg__-45.00min_1568027932.model')
        # p2_model = load_a_model('models\model1d_3xconv+2xdenseSMALL4x4_startstamp1568634107_episode7900__170.00max___87.30avg_-310.00min_1568639921.model')
        p2 = players.Drunk()
        p2.name = "drunk"
    else:
        p2 = players.DDQNPlayer(p2_model)
        p2.name = "p2 on pretrained model"

    env = enviroment(p1, p2)

    if visualize_layers:
        analyse_model = AnalyseModel()  # make analyse model of each layer
        analyse_model.update_model(env.player1.model)
        analyse_model.reset_figs()

    # for stats
    log = ModelLog(log_filename, log_flag)
    log.add_player_info(p1, p2)
    log.add_constants()
    log.write_to_csv()
    log.write_parameters_to_file()
    count_stats = Stats()  # set new counterclass

    # setup for training
    p2modclass = str(log.model2_class)
    p2modclass = p2modclass.replace('/', '')
    description = f"(train1 prob-sample selfplay) vs p2={log.player2_class}@{p2modclass}"
    env.player1.setup_for_training(description=description, dir=tensorboard_dir)

    log.log_text_to_file(f"Training description: {description}\n")
    log.log_text_to_file(f"start training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        env.player1.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # check if training needs to proceed or if the model got corrupted
        if env.player1.got_NaNs:
            log.log_text_to_file(f"NaN as model output at episode {episode}")
            log.log_text_to_file(f"end training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            break  # Nan as model output. useless to continue training

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            env.block_invalid_moves(x=10)  # was 3

            if env.current_player == 1:
                # Get action
                if np.random.random() > epsilon:
                    # Get action from model
                    # action = np.argmax(env.player1.get_qs(current_state))
                    action_p1 = env.active_player.select_cell(board=env.playingField, state=env.GetState(), actionspace=env.action_space)
                else:
                    # explore!
                    if len(env.player1.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
                        # get probability based action
                        tau = 1 - epsilon
                        action_p1 = env.active_player.get_prob_action(state=env.playingField, actionspace=env.action_space, tau=tau)
                    else:
                        # Get random action
                        action_p1 = np.random.choice(env.action_space)

                # make action
                new_state, [_, reward_p1, _], done, _ = env.step(action_p1)

                if done:
                    # train one final time when winning the game
                    env.player1.update_replay_memory((current_state, action_p1, reward_p1, new_state, done))
                    env.player1.train(done, step)

                if env._invalid_move_played:
                    count_stats.invalidmove_count += 1  # tensorboard stats
                    # train only when invalid move played
                    env.player1.update_replay_memory((current_state, action_p1, reward_p1, new_state, done))
                    env.player1.train(done, step)
                    step += 1
                    logging.info(f" player1 caused the invalid move at action:{env._invalid_move_action}")

                # current_state = new_state
                # step += 1
            else:
                #action_opponent = env.active_player.select_cell(board=env.playingField, state=env.GetState(), actionspace=env.action_space)
                # selfplay
                env.playingField = env.inactive_player.inverse_state(env.playingField)
                action_opponent = env.active_player.select_cell(board=env.playingField, state=env.GetState(), actionspace=env.action_space)
                env.playingField = env.inactive_player.inverse_state(env.playingField)
                new_state, [_, reward_p1, _], done, _ = env.step(action_opponent)

                if not env._invalid_move_played:
                    # Every step after p2 played we update replay memory and train main network
                    env.player1.update_replay_memory((current_state, action_p1, reward_p1, new_state, done))
                    env.player1.train(done, step)

                    current_state = new_state
                    step += 1
                    if visualize_layers and len(env.player1.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
                        analyse_model.visual_debug_train(state=current_state, turns=env.turns, save_to_file=False, print_num=True)
                elif env._invalid_move_played:
                    logging.info(f" player2 caused the invalid move at action:{env._invalid_move_action}")

            if not env._invalid_move_played:
                env.setNextPlayer()

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward_p1

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

        # after every episode:
        if visualize_layers:
            if env.player1.target_update_counter == 0 and len(env.player1.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
                analyse_model.update_model(env.player1.model)
                print(env.player1.analyse_replay_memory())
            analyse_model.reset_figs()

        # for tensorboard stats
        if env.winner == env.player1.player_id:
            count_stats.win_count += 1
        elif env.winner == env.player2.player_id:
            count_stats.loose_count += 1
        else:
            count_stats.draw_count += 1
        count_stats.turns_count += env.turns
        if env.winnerhow == "Horizontal":
            count_stats.count_horizontal += 1
        if env.winnerhow == "Vertical":
            count_stats.count_vertical += 1
        if env.winnerhow == "Diagnal Right":
            count_stats.count_dia_right += 1
        if env.winnerhow == "Diagnal Left":
            count_stats.count_dia_left += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        count_stats.ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            try:
                avg_max_q = sum(env.player1.max_q_list[-AGGREGATE_STATS_EVERY:]) / len(env.player1.max_q_list[-AGGREGATE_STATS_EVERY:])
            except ZeroDivisionError:
                avg_max_q = 0

            try:
                delta_q = sum(env.player1.delta_q_list[-AGGREGATE_STATS_EVERY:]) / len(env.player1.delta_q_list[-AGGREGATE_STATS_EVERY:])
            except ZeroDivisionError:
                delta_q = 0

            #  Calculate over stats
            count_stats.aggregate_stats(calc_steps=AGGREGATE_STATS_EVERY)
            # update tensorboard
            env.player1.tensorboard.update_stats(reward_avg=count_stats.average_reward, reward_min=count_stats.min_reward, reward_max=count_stats.max_reward,
                                                 epsilon=epsilon, tau=tau, avg_max_q=avg_max_q, deta_q=delta_q,
                                                 win_count=count_stats.win_count, loose_count=count_stats.loose_count, draw_count=count_stats.draw_count,
                                                 invalidmove_count=count_stats.invalidmove_count, win_ratio=count_stats.win_ratio,
                                                 turns_count=count_stats.turns_count, count_horizontal=count_stats.count_horizontal,
                                                 count_vertical=count_stats.count_vertical, count_dia_left=count_stats.count_dia_left,
                                                 count_dia_right=count_stats.count_dia_right, reward_std=count_stats.std_reward)
            # reset stats
            count_stats.reset_stats()

            if visualize_layers:
                analyse_model.visual_debug_train(state=current_state, turns=env.turns, print_num=False, save_to_file=True, prefix=f'episode {episode}')

            # Save model, but only when avg reward is greater or equal a set value
            if count_stats.average_reward >= MIN_REWARD:
                model_temp_name = f'models/{log.model1_class}_{log.model1_name}_startstamp{log.model1_timestamp}_episode{episode}_{count_stats.max_reward:_>7.2f}max_{count_stats.average_reward:_>7.2f}avg_{count_stats.min_reward:_>7.2f}min_{int(time.time())}.model'
                env.player1.model.save(model_temp_name)
                log.log_text_to_file(f"model saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                log.log_text_to_file(f' {model_temp_name}\n')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # finally save model after training.
    model_temp_name = f'models/{log.model1_class}_{log.model1_name}_startstamp{log.model1_timestamp}_endtraining_{count_stats.max_reward:_>7.2f}max_{count_stats.average_reward:_>7.2f}avg_{count_stats.min_reward:_>7.2f}min_{int(time.time())}.model'
    env.player1.model.save(model_temp_name)
    log.log_text_to_file(f"model saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.log_text_to_file(f' {model_temp_name}')
    log.log_text_to_file(f"end training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def batch_train():
    model_list = []
    """

    model_list.append(model5(input_shape=(6, 7, 1), output_num=7,
                      par_loss='mse', par_opt=Adam(lr=0.001), par_metrics='accuracy'))

    model_list.append(model2(input_shape=(6, 7, 1), output_num=7))
    """
    #model_list.append(func_model5_duel1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='mse', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'))

    #model_list.append(func_model_duel1b1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='logcosh', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.01), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.001, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.01, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))

    model_list.append(func_model_duel1b2(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b2(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.01), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b2(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.001, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b2(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.01, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))

    model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.01), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.001, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.01, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))

    #model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'))
    #model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='categorical_crossentropy', par_opt=Adam(lr=0.01), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'))

    #model_list.append(model5(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss=model5.huber_loss(), par_opt=Adam(lr=0.001), par_metrics='accuracy'))

    #model_list.append(func_model5(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss=func_model5.huber_loss(), par_opt=Adam(lr=0.001), par_metrics='accuracy'))

    #model_list.append(func_model5_duel1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='mse', par_opt=Adam(lr=0.001), par_metrics='accuracy'))

    #clipnorm=1.0, clipvalue=0.5

    #model2 = load_a_model('models/func_model1_3xconv+2xdenseSMALL4x4(func)(mse^Adam^lr=0.001)_startstamp1569842535_episode7450__170.00max___66.60avg_-205.00min_1569845018.model')
    model2 = None
    for model in model_list:
        trainNN(p1_model=model, p2_model=model2, visualize_layers=False, debug_flag=True)


if __name__ == '__main__':
    #batch_train()
    trainA2C()
