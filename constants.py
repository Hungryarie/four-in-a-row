"""# Constants
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 10_000  # 25_000  # 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MIN_REWARD = -60  # 120 # For model save
MEMORY_FRACTION = 0.20  # not so relevant yet (only gpu)

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False
"""


class TrainingParameters():
    def __init__(self):
        # Constants
        self.DISCOUNT = 0.99
        self.REPLAY_MEMORY_SIZE = 10_000  # 25_000  # 50_000  # How many last steps to keep for model training
        self.MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
        self.MIN_REWARD = -60  # 120 # For model save
        self.MEMORY_FRACTION = 0.20  # not so relevant yet (only gpu)

        # Environment settings
        self.EPISODES = 20_000
        self.MAX_INVALID_MOVES = 200

        # Exploration settings
        #self.epsilon = 1  # not a constant, going to be decayed
        self.EPSILON_DECAY = 0.99975
        self.MIN_EPSILON = 0.001

        #  Stats settings
        self.AGGREGATE_STATS_EVERY = 50  # episodes
        self.SHOW_PREVIEW = False

        # Training reward setting
        self.reward_dict = {}
        self.reward_dict['win'] = 100
        self.reward_dict['lose'] = -100
        self.reward_dict['tie'] = -99
        self.reward_dict['invalid'] = -10
        self.reward_dict['step'] = -1

