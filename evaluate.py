import numpy as np
import matplotlib.pyplot as plt
import players
from env import enviroment
from model import load_a_model


def test_in_env():

    SHOW_EVERY = 50

    # Model = load_a_model('models\PreLoadedModel_model4_dense2x128(softmax)_startstamp1563365714_endtraining_startstamp1563370404_episode7050____7.00max____5.23avg__-14.00min__1563372123.model')
    # Model = load_a_model('models\dense2x128(softmax)_startstamp1567090369_episode9050__170.00max__152.60avg___95.00min__1567092303.model')
    # Model = load_a_model('models\model4catcross_dense2x128(softmax+CatCrossEntr)_startstamp1568050818_episode9600__165.00max__136.40avg__-50.00min_1568052142.model')
    Model = load_a_model('models/model1d_3xconv+2xdenseSMALL4x4_startstamp1568634107_episode7900__170.00max___87.30avg_-310.00min_1568639921.model')
    # Model2 = load_a_model('models\model4a_dense2x128(softmax)(flattenLAST,input_shape bug gone lr=0.001)_startstamp1568020820_episode8350__170.00max__141.60avg__-45.00min_1568027932.model')
    Model2 = load_a_model('models/model1d_3xconv+2xdenseSMALL4x4_startstamp1568634107_episode7900__170.00max___87.30avg_-310.00min_1568639921.model')
    p1 = players.DDQNPlayer(Model)
    # p1 = players.Drunk()
    # p2 = players.Drunk()
    p2 = players.DDQNPlayer(Model2)

    p1.name = "trained against model"
    p2.name = "trained againt random"
    env = enviroment(p1, p2)
    env.env_info()

    print("evaluate Training...")
    rewards_history = []
    rewards_history_p1 = []
    rewards_history_p2 = []
    for i_episode in range(201):
        observation, *_ = env.reset()
        # print (observation)
        [rew, rew_p1, rew_p2], winner = env.test(render=False)

        rewards_history.append(rew)
        rewards_history_p1.append(rew_p1)
        rewards_history_p2.append(rew_p2)
        # print(f"Episode {i_episode} finished with rewardpoints: {rew}")

        if i_episode % SHOW_EVERY == 0:
            print(f"{SHOW_EVERY} ep mean: {np.mean(rewards_history[-SHOW_EVERY:])}")

    print(f"{i_episode} ep mean (total): {np.mean(rewards_history[:])}")
    print(f"{i_episode} ep mean (total p1): {np.mean(rewards_history_p1[:])}")
    print(f"{i_episode} ep mean (total p2): {np.mean(rewards_history_p2[:])}")
    plt.style.use('seaborn')
    plt.plot(0, len(rewards_history), rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def play_in_env():
    print("play in enviroment\n")
    # Model = load_a_model('models/func_model_duel1b1_dueling_3xconv+2xdenseSMALL4x4_catCros_SGD+extra dense Functmethod1_startstamp1568924178_endtraining__170.00max__115.00avg_-280.00min_1568928710.model')

    # against drunk
    Model = load_a_model('models/A2C/selfplay/1572276090_ep300_actor.model')
    # against model above
    # Model = load_a_model('models/func_model1_3xconv+2xdenseSMALL4x4(func)(mse^Adam^lr=0.001)_startstamp1569850212_episode9800__170.00max__165.40avg__150.00min_1569854021.model')

    p1 = players.DDQNPlayer(Model)
    p1 = players.Human()
    p2 = players.Human()
    # p2 = players.DDQNPlayer(Model2)

    p1.name = "DDQN"
    p2.name = "arnoud"
    env = enviroment(p1, p2)
    env.env_info()

    [rew, rew_p1, rew_p2], _ = env.test(render=True, visualize_layers=False)
    print(f"reward: {rew}")
    print(f"reward_p1: {rew_p1}")
    print(f"reward_p2: {rew_p2}")
    print(env.Winnerinfo())


if __name__ == '__main__':
    # test_in_env()
    play_in_env()
