
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
        self.EPSILON_DECAY = 0.99975 # 0.99900
        self.TAU_DECAY = 0.99900
        self.MIN_EPSILON = 0.001 # 0.25 

        #  Stats settings
        self.AGGREGATE_STATS_EVERY = 10  # episodes
        self.SHOW_PREVIEW = True

        # Training reward setting
        self.reward_dict = {}
        #self.reward_dict['win'] = 100
        #self.reward_dict['lose'] = -100
        #self.reward_dict['tie'] = -99
        #self.reward_dict['invalid'] = -10
        #self.reward_dict['step'] = 0 # -0.5
        self.reward_dict['win'] = 1.
        self.reward_dict['lose'] = -1.
        self.reward_dict['tie'] = -0.9
        self.reward_dict['invalid'] = -0.03  #-0.1
        self.reward_dict['step'] = -0.01  #-0.05
