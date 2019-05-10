import numpy as np 
import random
import players
from fiar import FiarGame

def playgame():
    print ("Starting game")
    #PlayerName = input('Player name:')
    PlayerName="Arnoud"
    print (f"playername: {PlayerName}")
    Game = FiarGame(PlayerName)
    Game.ShowField2()

    while True:
        if Game.nextTurn==1:
            print (f"{Game.playerColor}:{Game.playerName}'s turn")
            ColumnNo= input('column no:')
        if Game.nextTurn==2:
            print (f"{Game.opponentColor}:{Game.opponentName}'s turn")
            ColumnNo=random.randint(0,Game.columns-1)

        if Game.addCoin(ColumnNo,Game.nextTurn):
            Game.ShowField2()
            if Game.checkFull(): # check if the game is over (a draw)
                break
            if Game.checkForWinner()!=0:
                break

def playAgainstRandom():
    p1 = players.Human()
    p2 = players.Drunk()
    game = FiarGame(p1.name)
    pass

if __name__ == '__main__':
    #playgame()
    playAgainstRandom()
