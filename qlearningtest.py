
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from time import time

seed = 1546847731  # or try a new seed by using: seed = int(time())
random.seed(seed)
print('Seed: {}'.format(seed))

class Game:
    board = None
    board_size = 0
    
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.reset()
    
    def reset(self):
        self.board = np.zeros(self.board_size)
    
    def play(self, cell):
        # returns a tuple: (reward, game_over?)
        if self.board[cell] == 0:
            self.board[cell] = 1
            game_over = len(np.where(self.board == 0)[0]) == 0
            return (1,game_over)
        else:
            return (-1,False)