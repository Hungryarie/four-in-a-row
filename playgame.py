import numpy as np
import random
import matplotlib.pyplot as plt
import players
from game import FiarGame
from env import enviroment
#
from model import ModelLog, load_a_model, model1, model1b, model2, model3, model4, model4a, model4b, model4catcross, model5
from tqdm import tqdm
from constants import *
import os
import tensorflow as tf
import time
from datetime import datetime


SHOW_EVERY = 50


def trainNN(Model=None):
    # For stats
    ep_rewards = [-20]
    epsilon = 1

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    if Model is None:
        Model = model5(input_shape=(6, 7, 1), output_num=7)  # (7, 6, 1)(1, 42)
        #Model = load_a_model('models\dense2x128(softmax)_startstamp1567090369_episode9050__170.00max__152.60avg___95.00min__1567092303.model')
        #Model2 = load_a_model('models\dense2x128(softmax)_startstamp1567090369_episode9050__170.00max__152.60avg___95.00min__1567092303.model')
    Model2 = load_a_model('models\model4a_dense2x128(softmax)(flattenLAST,input_shape bug gone lr=0.001)_startstamp1568020820_episode8350__170.00max__141.60avg__-45.00min_1568027932.model')

    p1 = players.DDQNPlayer(Model)
    #p2 = players.Drunk()
    p2 = players.DDQNPlayer(Model2)
    p1.name = "DDQN on training"
    p2.name = "trained model on drunk"
    env = enviroment(p1, p2)


    #for stats
    log = ModelLog('parameters.log')
    log.add_player_info(p1, p2)
    log.add_constants()
    log.write_to_csv()
    log.write_parameters_to_file()
    win_count = 0
    loose_count = 0
    draw_count = 0
    invalidmove_count = 0
    turns_count = 0
    count_horizontal = 0
    count_vertical = 0
    count_dia_right = 0
    count_dia_left = 0

    log.log_text_to_file(f"start training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # setup for training
    p2modclass = str(log.model2_class)
    p2modclass = p2modclass.replace('/', '')
    env.player1.setup_for_training(description=f"vs p2={log.player2_class}@{p2modclass}")

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        env.player1.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            env.block_invalid_moves(x=3)

            if env.current_player == 1:
                # Get action
                if np.random.random() > epsilon:
                    # Get action from model
                    #action = np.argmax(env.player1.get_qs(current_state))
                    action = env.active_player.select_cell(board=env.playingField, state=env.GetState(), actionspace=env.action_space)
                else:
                    # Get random action
                    #action = np.random.randint(0, env.action_space_n)
                    action = np.random.choice(env.action_space)

                new_state, reward, done, _ = env.step(action)

                # Every step we update replay memory and train main network
                env.player1.update_replay_memory((current_state, action, reward, new_state, done))
                env.player1.train(done, step)

                if env._invalid_move_played:
                    invalidmove_count += 1  # tensorboard stats

                current_state = new_state
                step += 1
            else:
                action_opponent = env.active_player.select_cell(board=env.playingField, state=env.GetState(), actionspace=env.action_space)
                new_state, reward, done, _ = env.step(action_opponent)

                if done:
                    # train one final time when losing the game
                    #print('LOST GAME: reward:{reward}, done:{done}')
                    # Every step we update replay memory and train main network
                    env.player1.update_replay_memory((current_state, action, reward, new_state, done))
                    env.player1.train(done, step)

                current_state = new_state

            if not env._invalid_move_played:
                env.setNextPlayer()

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

        # tensorboard stats
        if env.winner == env.player1.player_id:
            win_count += 1
        elif env.winner == env.player2.player_id:
            loose_count += 1
        else:
            draw_count += 1
        turns_count += env.turns
        if env.winnerhow == "Horizontal":
            count_horizontal += 1
        if env.winnerhow == "Vertical":
            count_vertical += 1
        if env.winnerhow == "Diagnal Right":
            count_dia_right += 1
        if env.winnerhow == "Diagnal Left":
            count_dia_left += 1


        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            win_ratio = win_count / (win_count + loose_count)
            turns_count = turns_count / AGGREGATE_STATS_EVERY
            count_horizontal = count_horizontal / AGGREGATE_STATS_EVERY
            count_vertical = count_vertical / AGGREGATE_STATS_EVERY
            count_dia_left = count_dia_left / AGGREGATE_STATS_EVERY
            count_dia_right = count_dia_right / AGGREGATE_STATS_EVERY
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            env.player1.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon,
                                                 win_count=win_count, loose_count=loose_count, draw_count=draw_count, invalidmove_count=invalidmove_count,
                                                 win_ratio=win_ratio, turns_count=turns_count, count_horizontal=count_horizontal,
                                                 count_vertical=count_vertical, count_dia_left=count_dia_left, count_dia_right=count_dia_right)

            # reset stats
            win_count = 0
            loose_count = 0
            draw_count = 0
            invalidmove_count = 0
            count_horizontal = 0
            count_vertical = 0
            count_dia_left = 0
            count_dia_right = 0

            # Save model, but only when avg reward is greater or equal a set value
            if average_reward >= MIN_REWARD:
                model_temp_name = f'models/{log.model1_class}_{log.model1_name}_startstamp{log.model1_timestamp}_episode{episode}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_{int(time.time())}.model'
                env.player1.model.save(model_temp_name)
                log.log_text_to_file(f"model saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.log_text_to_file(f' {model_temp_name}\n')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # finally save model after training.
    model_temp_name = f'models/{log.model1_class}_{log.model1_name}_startstamp{log.model1_timestamp}_endtraining_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_{int(time.time())}.model'
    env.player1.model.save(model_temp_name)
    log.log_text_to_file(f"model saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.log_text_to_file(f' {model_temp_name}\n')
    log.log_text_to_file(f"end training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # remove objects
    del p1
    del p2
    del env
    del log


def PlayInEnv():
    print("play in enviroment\n")
    #Model = load_a_model('models/PreLoadedModel_model4_dense2x128(softmax)_startstamp1563365714_endtraining_startstamp1563370404_episode7050____7.00max____5.23avg__-14.00min__1563372123.model')
    #Model = load_a_model('models\dense2x128(softmax)_startstamp1567090369_episode9050__170.00max__152.60avg___95.00min__1567092303.model')
    #Model = load_a_model('models\model4_dense2x128(softmax)_startstamp1567106898_endtraining__155.00max___19.30avg_-215.00min_1567109405.model')
    Model = load_a_model('models\model4catcross_dense2x128(softmax+CatCrossEntr)_startstamp1568050818_episode9600__165.00max__136.40avg__-50.00min_1568052142.model')
    Model2 = load_a_model('models\model4a_dense2x128(softmax)(flattenLAST,input_shape bug gone lr=0.001)_startstamp1568020820_episode8350__170.00max__141.60avg__-45.00min_1568027932.model')
    p1 = players.DDQNPlayer(Model)
    #p2 = players.Human()
    p2 = players.DDQNPlayer(Model)

    p1.name = "DDQN"
    p2.name = "arnoud"
    env = enviroment(p1, p2)
    env.env_info()

    rew, _ = env.test(render=True)
    print(f"reward: {rew}")
    print(env.Winnerinfo())


def TestInEnv():
    #Model = load_a_model('models\PreLoadedModel_model4_dense2x128(softmax)_startstamp1563365714_endtraining_startstamp1563370404_episode7050____7.00max____5.23avg__-14.00min__1563372123.model')
    #Model = load_a_model('models\dense2x128(softmax)_startstamp1567090369_episode9050__170.00max__152.60avg___95.00min__1567092303.model')
    Model = load_a_model('models\model4catcross_dense2x128(softmax+CatCrossEntr)_startstamp1568050818_episode9600__165.00max__136.40avg__-50.00min_1568052142.model')
    Model2 = load_a_model('models\model4a_dense2x128(softmax)(flattenLAST,input_shape bug gone lr=0.001)_startstamp1568020820_episode8350__170.00max__141.60avg__-45.00min_1568027932.model')
    p1 = players.DDQNPlayer(Model)
    #p1 = players.Drunk()
    #p2 = players.Drunk()
    p2 = players.DDQNPlayer(Model2)

    p1.name = "trained against model"
    p2.name = "trained againt random"
    env = enviroment(p1, p2)
    env.env_info()

    print("evaluate Training...")
    rewards_history = []
    for i_episode in range(201):
        observation, *_ = env.reset()
        # print (observation)
        rew, winner = env.test(render=False)

        rewards_history.append(rew)
        #print(f"Episode {i_episode} finished with rewardpoints: {rew}")

        if i_episode % SHOW_EVERY == 0:
            print(f"{SHOW_EVERY} ep mean: {np.mean(rewards_history[-SHOW_EVERY:])}")

    print(f"{i_episode} ep mean (total): {np.mean(rewards_history[:])}")
    plt.style.use('seaborn')
    plt.plot(0, len(rewards_history), rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def batch_train():
    model_list = []
    model_list.append(model1b(input_shape=(6, 7, 1), output_num=7))
    model_list.append(model1b(input_shape=(6, 7, 1), output_num=7))
    model_list.append(model1b(input_shape=(6, 7, 1), output_num=7))
    #model_list.append(model4catcross(input_shape=(6, 7, 1), output_num=7))
    #model_list.append(model5(input_shape=(6, 7, 1), output_num=7))
    #model_list.append(model5(input_shape=(6, 7, 1), output_num=7))


    for model in model_list:
        trainNN(model)

if __name__ == '__main__':

    #TestInEnv()
    #PlayInEnv()
    batch_train()
