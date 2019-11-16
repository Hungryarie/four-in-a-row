import numpy as np


class Stats:
    def __init__(self):
        self.reset_stats()
        self.episode_reward = 0
        self.ep_rewards = []  # [-20]
        self.max_q_list = []
        self.epsilon = 1
        self.tau = 0
        self.episode = 0
        self.episodes = []

    def reset_stats(self):
        """Reset stats"""

        self.win_ratio = 0
        self.win_count = 0
        self.loose_count = 0
        self.draw_count = 0
        self.invalidmove_count = 0
        self.invalidmove_count_succesive = 0
        self.turns_count = 0
        self.count_horizontal = 0
        self.count_vertical = 0
        self.count_dia_right = 0
        self.count_dia_left = 0

    def aggregate_stats(self, calc_steps):
        try:
            self.win_ratio = self.win_count / (self.win_count + self.loose_count)
        except ZeroDivisionError:
            self.win_ratio = 0.5
        self.turns_count = self.turns_count / calc_steps
        self.count_horizontal = self.count_horizontal / calc_steps
        self.count_vertical = self.count_vertical / calc_steps
        self.count_dia_left = self.count_dia_left / calc_steps
        self.count_dia_right = self.count_dia_right / calc_steps
        self.average_reward = sum(self.ep_rewards[-calc_steps:]) / len(self.ep_rewards[-calc_steps:])
        self.min_reward = min(self.ep_rewards[-calc_steps:])
        self.max_reward = max(self.ep_rewards[-calc_steps:])
        self.std_reward = np.std(self.ep_rewards[-calc_steps:])
