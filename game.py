import numpy as np
import random


class FiarGame:
    REWARD_WINNING = 20  # 10
    REWARD_LOSING = -10
    REWARD_TIE = -5  #
    REWARD_INVALID_MOVE = -0.5  # -2
    REWARD_STEP = -0.5

    def __init__(self, player1, player2):

        self.rows = 6
        self.columns = 7

        self.player1 = player1
        self.player1.color = "Y"
        self.player1.player_id = 1

        self.player2 = player2
        self.player2.color = "R"
        self.player2.player_id = 2

        self.reset()

    def reset(self):
        self.playingField = np.zeros([self.rows, self.columns], dtype=int)
        self.playingField = self.playingField[:, :, np.newaxis]
        self.winner = 0                       # winner id. 0 is no winner yet
        self.winnerhow = "none"
        self.done = False
        self.turns = 0                       # amount of tries before winning
        self.nextTurn = random.randint(1, 2)   # random pick a player to start
        self.current_player = random.randint(1, 2)   # random pick a player to start
        self._invalid_move_played = False

    @property
    def active_player(self):
        if self.current_player == 1:
            return self.player1
        else:
            return self.player2

    @property
    def inactive_player(self):
        if self.current_player == 2:
            return self.player1
        else:
            return self.player2

    def setNextPlayer(self):
        # Set the next turn
        # print (f"current player={self.nextTurn}. next = {abs(self.nextTurn -2)+1}")
        self.nextTurn = abs(self.nextTurn - 2) + 1
        self.current_player = abs(self.current_player - 2) + 1

    def getPlayerById(self, id):
        if self.player1.player_id == id:
            return self.player1
        else:
            return self.player2

    def GetActionSpace(self):
        ActionList = []
        ActionList.extend(range(self.columns))
        return ActionList

    def GetActionSize(self):
        return self.columns

    def GetState(self):
        flatStateArray = self.playingField.flatten()
        return flatStateArray

    def GetObservationSize(self):
        return len(self.GetState())

    def Winnerinfo(self):
        """
        returns winner information:
        - name
        - color
        - id
        - how it won (mode)
        - in # of turns
        """
        if self.winner != 0:
            return (f"winner:{self.getPlayerById(self.winner).name} "
                    f"({self.getPlayerById(self.winner).color}), "
                    f"id:{self.winner}, how:{self.winnerhow}, in #turns:{self.turns}")
        else:
            return ("no winner (yet)")

    def CheckGameEnd(self):
        """
        returns whether the game has ended.
        """
        if self.winner == 0:
            # check for horizontal winner
            self.winner = self.checkForWinnerHor()
            self.winnerhow = "Horizontal"
        if self.winner == 0:
            # no winner? check for vertical winner
            self.winner = self.checkForWinnerVer()
            self.winnerhow = "Vertical"
        if self.winner == 0:
            # no winner? check for diagnal winner Right
            self.winner = self.checkForWinnerDiaRight()
            self.winnerhow = "Diagnal Right"
        if self.winner == 0:
            # no winner? check for diagnal winner Right
            self.winner = self.checkForWinnerDiaLeft()
            self.winnerhow = "Diagnal Left"
        if self.winner != 0:
            self.done = True
        if self.checkFull():
            # check for a tie / draw
            self.done = True
        # print(self.Winnerinfo())

        return self.done

    def checkFull(self):
        if self.turns >= self.rows * self.columns:
            # print("a draw!!!!")
            self.winnerhow = "draw / tie"
            return True
            #self.reset()

    def checkForWinnerDiaRight(self):
        #print("Check for a Diagnal Right Winner")
        array = self.NProtate45(self.playingField)  # scew the playingfield 45degrees 
        return sum(np.apply_along_axis(self.checkFourOnARow, axis=0, arr=array))  #axis=0: vertical

    def checkForWinnerDiaLeft(self):
        #print("Check for a Diagnal Left Winner")
        array = self.NProtate275(self.playingField)  # scew the playingfield minus 45degrees 
        return sum(np.apply_along_axis(self.checkFourOnARow, axis=0, arr=array))  #axis=0: vertical       

    def checkForWinnerHor(self):
        #print("Check for a Horizontal Winner")
        #print (np.apply_along_axis( self.checkFourOnARow, axis=1, arr=self.playingField ))
        """
        if sum(np.apply_along_axis( self.checkFourOnARow, axis=1, arr=self.playingField ))==WhosTurn:
            return True
        else:
            return False
        """
        return sum(np.apply_along_axis(self.checkFourOnARow, axis=1, arr=self.playingField))      

    def checkForWinnerVer(self):
        #print("Check for a Vertical Winner")
        return sum(np.apply_along_axis(self.checkFourOnARow, axis=0, arr=self.playingField))

    @staticmethod
    def NProtate45(array):
        # Rotate numpy array 45 degrees
        rows, cols, _ = array.shape
        rot = np.zeros([rows, cols + rows - 1], dtype=int)
        for i in range(rows):
            for j in range(cols):
                rot[i, i + j] = array[i, j]
        return rot

    @staticmethod
    def NProtate275(array):
        # Rotate numpy array 275 degrees
        rows, cols, _ = array.shape
        rot = np.zeros([rows, cols + rows - 1], dtype=int)
        for i in range(rows):
            for j in range(cols):
                rot[i, i - j] = array[i, j]
        return rot

    @staticmethod
    def checkFourOnARow(x):
        #print ("x:", x)
        count = 0
        same = 0
        win = False
        winner = 0
        for y in x:
            #print (y)
            if same == y and y != 0:
                count += 1
            else:
                count = 0
            if count >= 3:
                # 3 checks = 4 on a row
                win = True
                winner = same
                break
            same = y

        #print (f"x: {x} >> countsame: {count+1} >> win: {win} >> winner: {winner}")
        return winner  #sum(x)

    def addCoin(self, inColumn, ActivePlayer):
        #print (f"adding {ActivePlayer} coin in column {inColumn}")

        try:
            inColumn = int(inColumn)
        except:
            print(f"'{inColumn}' is no integer, try again")
            return False
        if inColumn >= self.columns or inColumn < 0:
            print(f"'{inColumn}' is out of bounds, try again")
            return False
        i = 1
        while True:
            try:
                if self.playingField[self.rows - i, inColumn] == 0:
                    self.playingField[self.rows - i, inColumn] = ActivePlayer
                    self._invalid_move_played = False
                    #self.setNextPlayer() # Set the next turn  
                    self.turns += 1  # iterate the number of turns
                    return True
                else:
                    # print (f"row {self.rows-i} is already filled with {self.playingField[self.rows-i, inColumn]}")
                    i += 1
            except:
                # print(f"column {inColumn} is already totally filled")
                self._invalid_move_played = True
                return False

    def ShowField(self):
        print(" |0|1|2|3|4|5|6| << colums")
        print(self.playingField)

    def ShowField2(self):
        print("|0|1|2|3|4|5|6| << colums")
        for i in self.playingField:
            row = ""
            for j in i:
                if j == 0:
                    j = " "
                elif j == 1:
                    j = self.player1.color
                elif j == 2:
                    j = self.player2.color
                row += "|" + str(j)   
            row += "|"
            print(row)
