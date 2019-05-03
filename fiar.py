import numpy as np 
import random

def main():
    print ("Starting game")
    #PlayerName = input('Player name:')
    PlayerName="Arnoud"
    print (f"playername: {PlayerName}")
    Game = StartGame(PlayerName)

    while True:
        if Game.nextTurn==1:
            print (f"{Game.playerName}'s turn")
            
            ColumnNo= int(input('column no:'))

            Game.addCoin(ColumnNo,Game.nextTurn)
        if Game.nextTurn==2:
            print (f"{Game.opponentName}'s turn")

            Game.addCoin(random.randint(0,Game.columns-1),Game.nextTurn)
        
        Game.ShowField()
        if Game.checkForWinner()!=0:
            break


def checkFourOnARow( x ):
    #print ("x:", x)
    count=0
    same=0
    win=False
    winner=0
    for y in x:
        #print (y)
        if same==y and y!=0:
            count+=1
        else:
            count=0
        if count>=3:
            # 3 checks = 4 on a row
            win=True
            winner=same
            break
        same=y

    #print (f"x: {x} >> countsame: {count+1} >> win: {win} >> winner: {winner}")
    return winner #sum(x)

class StartGame:
    def __init__(self, playerName):
        self.playerName = playerName
        self.player=1
        self.opponent=2
        self.rows = 6
        self.columns = 7
        self.opponentColor = "R"
        self.playerColor = "Y"
        self.opponentName = "Computer"
        self.nextTurn=random.randint(1,2)
        self.playingField= np.zeros([self.rows,self.columns], dtype=int)
        self.winner=0                   # winner id. 0 is no winner yet
        self.counts=0                   # amount of tries before winning

    def checkForWinner(self):

        if self.winner==0:
            # check for horizontal winner
            self.winner=self.checkForWinnerHor()
        if self.winner==0:
            # no winner? check for vertical winner
            self.winner=self.checkForWinnerVer()
        if self.winner==0:
            # no winner? check for diagnal winner
            # 2do adding diagnal check
            self.checkForWinnerDiaRight()
            pass
        if self.winner!=0:
            print (f"winner:{self.winner}")
        else:
            print ("no winner yet")

        return self.winner

    def checkForWinnerDiaRight (self):
        #2do adding a 0 to row 0, adding 2 zeros to row 1 etc.
        rows, cols =self.playingField.shape

        return sum(np.apply_along_axis( checkFourOnARow, axis=0, arr=self.playingField )) #axis=0: vertical
        
    def checkForWinnerHor (self): #, WhosTurn):
        print("Check for a Horizontal Winner")
        #print (np.apply_along_axis( checkFourOnARow, axis=1, arr=self.playingField ))
        """
        if sum(np.apply_along_axis( checkFourOnARow, axis=1, arr=self.playingField ))==WhosTurn:
            return True
        else:
            return False 
        """
        return sum(np.apply_along_axis( checkFourOnARow, axis=1, arr=self.playingField ))      
        
    def checkForWinnerVer (self): #, WhosTurn):
        print("Check for a Vertical Winner")
        return sum(np.apply_along_axis( checkFourOnARow, axis=0, arr=self.playingField ))

    def addCoin (self, inColumn, WhosTurn):
        print (f"adding {WhosTurn} coin in column {inColumn}")

        i=1
        success = False
        while True:
            try:
                if self.playingField[self.rows-i, inColumn] == 0:
                    self.playingField[self.rows-i, inColumn] = WhosTurn
                    success = True
                    break
                else:
                    # print (f"row {self.rows-i} is already filled with {self.playingField[self.rows-i, inColumn]}")
                    i+=1

            except:
                print (f"column {inColumn} is already totally filled")
                break
        
        if success:
            # Set the next turn
            print (f"current player={self.nextTurn}. next = {abs(self.nextTurn -2)+1}")
            self.nextTurn=abs(self.nextTurn -2)+1
            # iterate the number of tries
            self.counts +=1

    def ShowField(self):
        print (" |0|1|2|3|4|5|6| << colums")
        print (self.playingField)

if __name__ == '__main__':
    main()
