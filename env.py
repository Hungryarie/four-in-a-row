# trying to implement:
# https://github.com/jaromiru/AI-blog/blob/master/CartPole-DQN.py
# https://github.com/keon/deep-q-learning/blob/master/ddqn.py

import numpy as np 
import random
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

    def step(self, action):
        """
        returns
        observation, reward, done, info
        """
        self.addCoin(action, self.active_player.player_id)
            
        self.checkForWinner()
        reward=1
        return self.GetState(), reward, self.done, self.info()
     
    def info(self):
        dicti = {"active_player" : self.active_player.name,
                 "moves_played" : self.turns}
        return dicti
    
    def test(self, episodes, render=False):
        for i_episode in range(episodes):
            print (f"episode: {i_episode}")
            observation = self.reset()
            for t in range(self.observation_space_n):
                action = self.sample()
                observation, reward, done, info = self.step(action)

                if not self._invalid_move_played:
                    self.setNextPlayer()
                    t-=1 # 
                if render:
                    self.render()
                print(observation)
                if done:
                    print(self.Winnerinfo())
                    print(f"Episode finished after {t+1} timesteps")
                    print("\n")
                    break


p1 = players.Human()
p2 = players.Drunk()
p1.name = "Arnoud"
p2.name = "Henk"
env = enviroment(p1,p2)

print(env.observation_space_n)
env.test(2, True)


    
