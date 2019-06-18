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
        #FiarGame.__init__(self, *args)
        super().__init__(*args)
        #self.observation_space = self.GetObservationSpace()
        self.observation_space_n = self.GetObservationSize()
        self.action_space = self.GetActionSpace()
        self.action_space_n = self.GetActionSize()

    def reset(self):
        """
        resets the game and 
        returns the observationspace / state
        """
        super().reset()
        return self.GetState(), 0, False, None
    
    def render(self):
        """
        reders the game
        """
        self.ShowField2()

    def sample(self):
        """
        returns a random action from the actionspace.
        """
        return random.randint(0, self.action_space_n-1)

    def reward(self):
        reward = 0
        if self.winner==0 and self.done==True:
            reward = self.REWARD_TIE
        if self.winner==1:
            reward = self.REWARD_WINNING
        if self.winner==2:
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
        
        return self.GetState(), self.reward(), self.done, self.info()
     
    def info(self):
        dicti = {"active_player" : self.active_player.name,
                 "moves_played" : self.turns}
        return dicti
    
    def test(self, render=False):
        observation, done, ep_reward = self.reset(), False, 0

        while not done:
            action = self.sample()
            observation, reward, done, info = self.step(action)

            if not self._invalid_move_played:
                self.setNextPlayer()
            ep_reward += reward

            if render:
                self.render()
            print(observation)
            print (f"reward: {reward}")
            if done:
                self.render()
                print(self.Winnerinfo())
                print(f"Episode finished after {self.turns} timesteps")
                print("\n")
        return ep_reward


p1 = players.Human()
p2 = players.Drunk()
p1.name = "Arnoud"
p2.name = "Henk"
env = enviroment(p1,p2)


# print(env.observation_space_n)
# for i_episode in range(3):
#     print (f"episode: {i_episode}")
#     ep_reward = env.test(False)
#     print (f"ep_reward: {ep_reward}")

print ("evaluate Training...")
rewards_history = []
for i_episode in range(40):
    observation = env.reset()
    print (observation)
    rew = env.test(render=False)
    
    rewards_history.append(rew)
    print(f"Episode finished after {rew} timesteps")

plt.style.use('seaborn')
plt.plot(0, len(rewards_history), rewards_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
    