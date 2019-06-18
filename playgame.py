import numpy as np 
import random
import players
from game import FiarGame

def playAgainstRandom():
    p1 = players.Human()
    p2 = players.Drunk()
    p1.name = "Arnoud"
    p2.name = "Henk"
    Game = FiarGame(p1, p2)

    print(Game.GetActionSpace())
    print(Game.GetState())

    print (f"the game of '{Game.player1.name}' vs '{Game.player2.name}'")

    while Game.winner==0:
        #print (f"{Game.opponentColor}:{Game.opponentName}'s turn")
        #print (f"cell random:{Game.active_player.select_cell(Game.playingField)}")
        print (f"> Turn: {Game.active_player.name} ({Game.active_player.color})")
        
        ColumnNo=Game.active_player.select_cell(Game.playingField) #random.randint(0,Game.columns-1)

        if Game.addCoin(ColumnNo,Game.current_player):
            Game.ShowField2()
            if Game.checkFull(): # check if the game is over (a draw)
                break
            if Game.checkForWinner()!=0:
                print(Game.Winnerinfo())
                break
            Game.setNextPlayer()
        

if __name__ == '__main__':
    #playgame()
    playAgainstRandom()
