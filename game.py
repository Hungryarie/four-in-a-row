import numpy as np
import random


class FiarGame:
    # REWARD_WINNING = 100  # 200  # 20
    # REWARD_LOSING = -100  #-100  #-10
    # REWARD_TIE = -99  #-99  # -5
    # REWARD_INVALID_MOVE = -10   #-50  # -10  # -0.5
    # REWARD_STEP = -1  #-5  # -0.5

    def __init__(self):

        self.rows = 6
        self.columns = 7
        self._extra_toprows = 1  # extra rows at the top of the playingfield (can be used for feature engineering)

        self.playingField = np.zeros([self.rows + self._extra_toprows, self.columns], dtype=float)  # int
        self.playingField = self.playingField[:, :, np.newaxis]
        self.featuremap = np.zeros([self.rows + self._extra_toprows, self.columns, 4], dtype=float)  # int

        # self.reset()

    def add_players(self, player1, player2):
        self.player1 = player1
        self.player1.color = "1"
        self.player1.player_id = 1
        self.player1.value = 1.

        self.player2 = player2
        self.player2.color = "2"
        self.player2.player_id = 2
        self.player2.value = -1

        self.reset()

    def reset(self):
        self.playingField.fill(0)               # reset array with zero
        self.featuremap.fill(0)                 # reset array with zero

        self.winner = 0                         # winner id. 0 is no winner yet
        self.winnerhow = "none"
        self.done = False
        self.turns = 0                          # amount of tries before winning
        self.current_player = random.choice([self.player1.player_id, self.player2.player_id])  # random pick a player to start
        self.current_player_value = self.getPlayerById(self.current_player).value
        self._invalid_move_played = False
        self._invalid_move_count = 0
        self._invalid_move_action = None
        self.prev_invalid_move_count = 0        # for collecting the max in a row invalidmove count
        self.prev_invalid_move_reset = True

        self.featuremap_dict = {}
        self.enrich_feature_space()

    def enrich_feature_space(self):
        self._feature_space_field()
        self._feature_space_free_space()
        self._feature_space_active_player()
        self._feature_space_next_move()

    def _feature_space_field(self):
        """# layer 0 = the playingField"""
        layer = 0
        self.featuremap_dict['0'] = "the playingField"
        self.featuremap[:, :, layer] = self.playingField[:, :, 0]

    def _feature_space_active_player(self):
        """# layer 1 = active player"""
        layer = 1
        self.featuremap_dict['1'] = "active player"
        active_arr = np.full((self.rows + self._extra_toprows, self.columns), self.current_player_value)
        self.featuremap[:, :, layer] = active_arr

    def _feature_space_free_space(self):
        """# layer 2 = free space"""
        layer = 2
        self.featuremap_dict['2'] = "free space"
        for id_row, row in enumerate(reversed(self.featuremap)):
            for col in row:
                if col[0] == 0 and id_row < self.rows:
                    col[layer] = 1
                else:
                    col[layer] = 0

    def _feature_space_next_move(self):
        """# layer 3 = next move space"""
        layer = 3
        self.featuremap_dict['3'] = "next move space"

        col_flag = np.full((self.columns), False)
        for id_row, row in enumerate(reversed(self.featuremap)):
            for idx, col in enumerate(row):
                if col[0] == 0 and not col_flag[idx] and id_row < self.rows:  # value is zero,  column is not highlighted yet, and rows is not beyond max height
                    col[layer] = 1
                    col_flag[idx] = True
                else:
                    col[layer] = 0

    def print_feature_space(self, field=None):
        if field is None:
            field = self.featuremap
        print("Feature space (raw data):")
        print("> layer 0 = the playingField:")
        print(field[:, :, 0])
        print("\n> layer 1 = active player:")
        print(field[:, :, 1])
        print("\n> layer 2 = free space")
        print(field[:, :, 2])
        print("\n> layer 3 = next move space")
        print(field[:, :, 3])
        # print("\n")

    @property
    def active_player(self):
        if self.current_player == self.player1.player_id:
            return self.player1
        else:
            return self.player2

    @property
    def inactive_player(self):
        if self.current_player == self.player2.player_id:
            return self.player1
        else:
            return self.player2

    def setNextPlayer(self):
        # Set the next turn
        self.current_player = abs(self.current_player - 2) + 1
        # self.current_player = self.current_player * -1
        self.current_player_value = self.getPlayerById(self.current_player).value

        # enrich the featurespace with the current player
        # self.enrich_feature_space()  # is already done at env.step()
        self._feature_space_active_player()
        # self.print_feature_space()

    def getPlayerById(self, id):
        """Get the player by there player_id\n\n
        input: player_id\n
        output: player instance"""
        if self.player1.player_id == id:
            return self.player1
        else:
            return self.player2

    def get_player_by_value(self, value):
        """Get the player by there 'raw' value\n\n
        input: 'raw' value\n
        output: player instance"""
        if self.player1.value == value:
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
        """flattend array of the state"""
        raise DeprecationWarning
        flatStateArray = self.playingField.flatten()
        return flatStateArray

    def GetObservationSize(self):
        raise DeprecationWarning
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
            return (f"winner:{self.get_player_by_value(self.winner).name} "
                    f"({self.get_player_by_value(self.winner).color}), "
                    f"id:{self.get_player_by_value(self.winner).player_id}, how:{self.winnerhow}, in #turns:{self.turns}")
        else:
            return ("no winner (yet)")

    def CheckGameEnd(self):
        """
        returns whether the game has ended. Also sets the self.done flag.
        """
        if self.winner == 0:
            # check for horizontal winner
            self.winner = self.checkForWinnerHor()
        if self.winner == 0:
            # no winner? check for vertical winner
            self.winner = self.checkForWinnerVer()
        if self.winner == 0:
            # no winner? check for diagnal winner Right
            self.winner = self.checkForWinnerDiaRight()
        if self.winner == 0:
            # no winner? check for diagnal winner Right
            self.winner = self.checkForWinnerDiaLeft()
        if self.winner != 0:
            self.done = True
        if self.checkFull():
            # check for a tie / draw
            self.done = True

        return self.done

    def checkFull(self):
        if self.turns >= self.rows * self.columns:
            # print("a draw!!!!")
            self.winnerhow = "draw / tie"
            return True
        else:
            return False

    def checkForWinnerDiaRight(self):
        # print("Check for a Diagnal Right Winner")
        array = self.NProtate45(self.playingField)  # skew the playingField 45degrees
        winner = sum(np.apply_along_axis(self.checkFourOnARow, axis=0, arr=array))  # axis=0: vertical
        if winner != 0:
            self.winnerhow = "Diagnal Right"
        return winner

    def checkForWinnerDiaLeft(self):
        # print("Check for a Diagnal Left Winner")
        array = self.NProtate275(self.playingField)  # skew the playingField minus 45degrees
        winner = sum(np.apply_along_axis(self.checkFourOnARow, axis=0, arr=array))  # axis=0: vertical
        if winner != 0:
            self.winnerhow = "Diagnal Left"
        return winner

    def checkForWinnerHor(self):
        # print("Check for a Horizontal Winner")
        winner = sum(np.apply_along_axis(self.checkFourOnARow, axis=1, arr=self.playingField))
        if winner != 0:
            self.winnerhow = "Horizontal"
        return winner

    def checkForWinnerVer(self):
        # print("Check for a Vertical Winner")
        winner = sum(np.apply_along_axis(self.checkFourOnARow, axis=0, arr=self.playingField))
        if winner != 0:
            self.winnerhow = "Vertical"
        return winner

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
        # print(f"x:{x}")
        count = 0
        same = 0.
        win = False
        winner = 0.
        for y in x:
            # print (y)
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

        # print (f"x: {x} >> countsame: {count+1} >> win: {win} >> winner: {winner}")
        return winner  # sum(x)

    def _xd_to_2d(self, state, channel=0):
        """convert an array to a 2d array by removing anything but the given channel (default = 0)"""

        dims = len(state.shape)

        if dims == 2:
            return state
        elif dims == 3:
            # remove dimension in order to proceed
            return state[:, :, channel]
        else:
            logging.error("passing in a wrong dimension array")
            raise Exception("passing in a wrong dimension array")

    def _get_column_stackheight(self, column, state):
        """get the index of the highest coin in the action-column\n
        input: column and (2d) state\n
        output: height of the stack at the given column"""

        # convert state to 2d (if not already)
        state = self._xd_to_2d(state)

        # get column with values
        column_arr = state[:, int(column)]

        # return stackheight
        col_height = len(column_arr)
        for i, val in enumerate(column_arr):
            if val != 0:
                stack_height = col_height - i
                # print(f"stackheight of column {column} ({column_arr}) is {stack_height}")
                return stack_height
        return 0

    def _check_invalid_action_type(self, action):
        """checking whether or not an action is invalid, in terms of types etc.\n
        so no floats and strings etc, but an integer.\n
        And if the action is not out of bounds. \n
        ! Does not check on actionspace! (as this is an important value to train with)\n
        return true: invalid action type\n
        return false: no invalid action type """

        # check for not being a integer
        try:
            action = int(action)
        except Exception:
            print(f"'{action}' is no integer, try again")
            return True

        # check for out of bounds
        if action >= self.columns or action < 0:
            print(f"'{action}' is out of bounds, try again")
            return True
        else:
            return False

    def addCoin(self, inColumn, coin_value):
        # print (f"adding {coin_value} coin in column {inColumn}")

        """try:
            inColumn = int(inColumn)
        except Exception:
            print(f"'{inColumn}' is no integer, try again")
            self._invalid_move_played = True
            return False
        if inColumn >= self.columns or inColumn < 0:
            self._invalid_move_played = True
            print(f"'{inColumn}' is out of bounds, try again")
            return False
            """
        if self._check_invalid_action_type(inColumn):
            # invalid action played
            self._invalid_move_played = True
            self._invalid_move_action = inColumn
            return False

        # convert to int
        inColumn = int(inColumn)

        # get stackheight at the given column
        stackheight = self._get_column_stackheight(inColumn, self.playingField[:, :, 0])
        if stackheight >= self.rows:
            # print(f"column {inColumn} is already totally filled")
            self._invalid_move_played = True
            self._invalid_move_count += 1
            self._invalid_move_action = inColumn
            self.prev_invalid_move_reset = False
            return False
        else:
            # place coin in the right spot
            nb_rows, _ = self._xd_to_2d(self.playingField).shape
            row = nb_rows - stackheight - 1
            self.playingField[row, inColumn] = coin_value

            # set all flags correct
            self._invalid_move_played = False
            self._invalid_move_action = None
            self.turns += 1  # iterate the number of turns
            self.prev_invalid_move_count = self._invalid_move_count  # log previous invalid move count
            self.prev_invalid_move_reset = True  # set flag to being read
            self._invalid_move_count = 0  # reset invalid move counter to zero
            return True

        """inColumn = int(inColumn)
        self._invalid_move_action = None
        i = 1
        while True:
            try:
                if i > int(self.rows):
                    raise Exception("no checking further neccecary")
                elif self.playingField[self.rows - i, inColumn] == 0:
                    self.playingField[self.rows - i, inColumn] = coin_value
                    self._invalid_move_played = False
                    self.turns += 1  # iterate the number of turns
                    self.prev_invalid_move_count = self._invalid_move_count  # log previous invalid move count
                    self.prev_invalid_move_reset = True  # set flag to being read
                    self._invalid_move_count = 0  # reset invalid move counter to zero
                    return True
                else:
                    # print (f"row {self.rows-i} is already filled with {self.playingField[self.rows-i, inColumn]}")
                    i += 1
            except Exception:
                # print(f"column {inColumn} is already totally filled")
                self._invalid_move_played = True
                self._invalid_move_count += 1
                self._invalid_move_action = inColumn
                self.prev_invalid_move_reset = False
                return False"""

    def make_invalid_state(self, action, state):
        """Get the invalid state of an action. Can be used to train with"""
        # print(state[:, :, 0])
        state = np.copy(state)
        if self._extra_toprows == 1:
            state[0, action, 0] = self.active_player.value
            # print(state[:, :, 0])
            return state
        else:
            return state

    def _get_field_no_toprows(self, state):
        """returns the field without extra toprows (self._extra_toprows)"""

        # convert state array to 2d
        state = self._xd_to_2d(state)

        # get height
        r, _ = state.shape

        return state[(r - self.rows):, :]

    def ShowField(self):
        print(" |0|1|2|3|4|5|6| << colums")
        print(self.playingField)

    def ShowField2(self, field=None):
        if field is None:
            field = self.playingField

        field = self._get_field_no_toprows(field)

        print("|0|1|2|3|4|5|6| << colums")
        for i in field:
            row = ""
            for j in i:
                if j == 0:
                    coin_collor = " "
                elif j == self.player1.value:
                    coin_collor = self.player1.color
                elif j == self.player2.value:
                    coin_collor = self.player2.color
                row += "|" + str(coin_collor)
            row += "|"
            print(row)
