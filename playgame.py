import numpy as np 
import random
import matplotlib.pyplot as plt
import players
from game import FiarGame
from env import enviroment
from model import Model

SHOW_EVERY = 50


def trainNN():
    #p1 = players.Human()
    model = Model(num_actions=7)
    p1 = players.DQNPlayer(model)
    p2 = players.Drunk()
    p1.name = "DQN"
    p2.name = "Drunk Henk"
    env = enviroment(p1,p2)

    env.test()



def PlayInEnv():

    p1 = players.Drunk()
    p2 = players.Drunk()
    p1.name = "Arnoud"
    p2.name = "Henk"
    env = enviroment(p1,p2)

    print ("evaluate Training...")
    rewards_history = []
    for i_episode in range(101):
        observation, *_ = env.reset()
        #print (observation)
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

    print (f"the game of '{Game.player1.name}' vs '{Game.player2.name}'")

    while not Game.done:
        #print (f"{Game.opponentColor}:{Game.opponentName}'s turn")
        #print (f"cell random:{Game.active_player.select_cell(Game.playingField)}")
        print (f"> Turn: {Game.active_player.name} ({Game.active_player.color})")
        
        ColumnNo=Game.active_player.select_cell(board=Game.playingField, state=Game.GetState(), actionspace=Game.GetActionSpace()) #random.randint(0,Game.columns-1)

        if Game.addCoin(ColumnNo,Game.current_player):
            Game.ShowField2()
            if Game.CheckGameEnd():
                print(Game.Winnerinfo())
                break
            Game.setNextPlayer()
        

if __name__ == '__main__':
    #playAgainstRandom()
    PlayInEnv()
    #trainNN()
