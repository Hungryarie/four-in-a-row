import numpy as np 
import random

#test3
def main():
    print ("Starting game")
    #PlayerName = input('Player name:')
    PlayerName="Arnoud"
    print (f"playername: {PlayerName}")
    Game = StartGame(PlayerName)
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
            if Game.checkForWinner()!=0:
                break


def NProtate45(array):
    # Rotate numpy array 45 degrees
    rows, cols = array.shape
    rot = np.zeros([rows,cols+rows-1],dtype=int)
    for i in range(rows):
        for j in range(cols):
            rot[i,i + j] = array[i,j]
    return rot

def NProtate275(array):
    # Rotate numpy array 275 degrees
    rows, cols = array.shape
    rot = np.zeros([rows,cols+rows-1],dtype=int)
    for i in range(rows):
        for j in range(cols):
            rot[i,i - j] = array[i,j]
    return rot

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
        self.rows = 4#6
        self.columns = 5#7
        self.opponentColor = "R"
        self.playerColor = "Y"
        self.opponentName = "Computer"
        self.nextTurn=random.randint(1,2)
        #self.playingField= np.zeros([self.rows,self.columns], dtype=int)
        self.reset()
        self.winner=0                   # winner id. 0 is no winner yet
        self.winnerhow="none"
        self.counts=0                   # amount of tries before winning
    
    def reset(self):
        self.playingField= np.zeros([self.rows,self.columns], dtype=int)

    def checkForWinner(self):

        if self.winner==0:
            # check for horizontal winner
            self.winner=self.checkForWinnerHor()
            self.winnerhow="Horizontal"
        if self.winner==0:
            # no winner? check for vertical winner
            self.winner=self.checkForWinnerVer()
            self.winnerhow="Vertical"
        if self.winner==0:
            # no winner? check for diagnal winner Right
            self.winner=self.checkForWinnerDiaRight()
            self.winnerhow="Diagnal Right"
        if self.winner==0:
            # no winner? check for diagnal winner Right
            self.winner=self.checkForWinnerDiaLeft()
            self.winnerhow="Diagnal Left"
        if self.winner!=0:
            print (f"winner:{self.winner}, how:{self.winnerhow}")
        else:
            print ("no winner yet")

        return self.winner

    def checkForWinnerDiaRight (self):
        print("Check for a Diagnal Right Winner")
        array= NProtate45(self.playingField) # scew the playingfield 45degrees 
        return sum(np.apply_along_axis( checkFourOnARow, axis=0, arr=array )) #axis=0: vertical

    def checkForWinnerDiaLeft (self):
        print("Check for a Diagnal Left Winner")
        array= NProtate275(self.playingField) # scew the playingfield minus 45degrees 
        return sum(np.apply_along_axis( checkFourOnARow, axis=0, arr=array )) #axis=0: vertical       

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

        try:
            inColumn=int(inColumn)
        except:
            print (f"'{inColumn}' is no integer, try again")
            return False
        if inColumn>=self.columns or inColumn<0:
            print (f"'{inColumn}' is out of bounds, try again")
            return False
        i=1
        while True:
            try:
                if self.playingField[self.rows-i, inColumn] == 0:
                    self.playingField[self.rows-i, inColumn] = WhosTurn
                    # Set the next turn
                    #print (f"current player={self.nextTurn}. next = {abs(self.nextTurn -2)+1}")
                    self.nextTurn=abs(self.nextTurn -2)+1
                    # iterate the number of tries
                    self.counts +=1
                    return True
                else:
                    # print (f"row {self.rows-i} is already filled with {self.playingField[self.rows-i, inColumn]}")
                    i+=1
            except:
                print (f"column {inColumn} is already totally filled")
                return False


    def ShowField(self):
        print (" |0|1|2|3|4|5|6| << colums")
        print (self.playingField)
    
    def ShowField2(self):
        print ("|0|1|2|3|4|5|6| << colums")
        for i in self.playingField:
            row=""
            for j in i:
                if j==0:
                    j=" "
                elif j==1:
                    j=self.playerColor
                elif j==2:
                    j=self.opponentColor
                row+="|"+str(j)   
            row+="|"       
            print (row)

if __name__ == '__main__':
    main()
