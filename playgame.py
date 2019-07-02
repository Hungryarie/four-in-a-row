import numpy as np 
import random
import matplotlib.pyplot as plt
import players
from game import FiarGame
from env import enviroment
#
from model import model_1, model_2
from tqdm import tqdm
from constants import *
import os
import tensorflow as tf
import time


SHOW_EVERY = 50


def trainNN():
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

    Model = model_2(input_shape=(6, 7), output_num=7)  # (7, 6, 1)(1, 42)
    p1 = players.DDQNPlayer(Model)
    p2 = players.Drunk()
    p1.name = "DDQN"
    p2.name = "Drunk Henk"
    env = enviroment(p1, p2)

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        p1.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            if env.current_player == 1:
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(p1.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, env.action_space_n)

                new_state, reward, done, _ = env.step(action)

                # Every step we update replay memory and train main network
                p1.update_replay_memory((current_state, action, reward, new_state, done))
                p1.train(done, step)

                current_state = new_state
                step += 1
            else:
                action = env.active_player.select_cell(board=env.playingField, state=env.GetState(), actionspace=env.GetActionSpace())
                current_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            p1.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                p1.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    env.test()


def PlayInEnv():

    p1 = players.Drunk()
    p2 = players.Drunk()
    p1.name = "Arnoud"
    p2.name = "Henk"
    env = enviroment(p1, p2)

    print("evaluate Training...")
    rewards_history = []
    for i_episode in range(101):
        observation, *_ = env.reset()
        # print (observation)
        rew = env.test(render=False)
        
        rewards_history.append(rew)
        print(f"Episode {i_episode} finished with rewardpoints: {rew}")

        if i_episode % SHOW_EVERY == 0:
            print(f"{SHOW_EVERY} ep mean: {np.mean(rewards_history[-SHOW_EVERY:])}")

    plt.style.use('seaborn')
    plt.plot(0, len(rewards_history), rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def playAgainstRandom():
    p1 = players.Human()
    p2 = players.Drunk()
    p1.name = "Arnoud"
    p2.name = "Henk"
    Game = FiarGame(p1, p2)

    print(Game.GetActionSpace())
    print(Game.GetState())

    print(f"the game of '{Game.player1.name}' vs '{Game.player2.name}'")

    while not Game.done:
        #print (f"{Game.opponentColor}:{Game.opponentName}'s turn")
        #print (f"cell random:{Game.active_player.select_cell(Game.playingField)}")
        print(f"> Turn: {Game.active_player.name} ({Game.active_player.color})")
        
        ColumnNo = Game.active_player.select_cell(board=Game.playingField, state=Game.GetState(), actionspace=Game.GetActionSpace())  #random.randint(0,Game.columns-1)

        if Game.addCoin(ColumnNo, Game.current_player):
            Game.ShowField2()
            if Game.CheckGameEnd():
                print(Game.Winnerinfo())
                break
            Game.setNextPlayer()


if __name__ == '__main__':
    #playAgainstRandom()
    #PlayInEnv()
    trainNN()
