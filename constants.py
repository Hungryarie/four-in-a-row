# Constants
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 25_000  # 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 10# 5  # Terminal states (end of episodes)
MIN_REWARD = 5.5  # For model save
MEMORY_FRACTION = 0.20 # not so relevant yet

# Environment settings
EPISODES = 10_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

#LOAD_MODEL = None
