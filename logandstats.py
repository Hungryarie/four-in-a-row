import csv
import time

import numpy as np

from datetime import datetime


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


class ModelLog():
    def __init__(self, file, log_flag=True):
        self.logfilepath = file
        self.log_flag = log_flag

    def add_player_info(self, p1, p2):
        self.player1_class = p1.player_class
        self.player2_class = p2.player_class

        self.player1_name = p1.name
        self.player2_name = p2.name

        self.timestamp = int(time.time())
        self.timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            self.model1_name = p1.model.model_name
            self.model1_class = p1.model.model_class
            self.model1_timestamp = p1.model.timestamp
            self.model1_fullname = f'{self.model1_class}_{self.model1_name}_startstamp{self.model1_timestamp}'
            self.model1_opt_name = p1.model.optimizer_name
            self.model1_lr = p1.model._lr
            self.model1_loss = p1.model.loss
            self.model1_acc = p1.model.metrics
            self.model1_fin_activation = p1.model.fin_activation
        except AttributeError:
            self.model1_name = 'n/a'
            self.model1_class = 'n/a'
            self.model1_timestamp = 'n/a'
            self.model1_fullname = 'n/a'
            self.model1_opt_name = 'n/a'
            self.model1_lr = 'n/a'
            self.model1_loss = 'n/a'
            self.model1_acc = 'n/a'
            self.model1_fin_activation = 'n/a'

        try:
            self.model2_name = p2.model.model_name
            self.model2_class = p2.model.model_class
            self.model2_timestamp = p2.model.timestamp
            self.model2_fullname = f'{self.model2_class}_{self.model2_name}_startstamp{self.model2_timestamp}'
            self.model2_opt_name = p2.model.optimizer_name
            self.model2_lr = p2.model._lr
            self.model2_loss = p2.model.loss
            self.model2_acc = p2.model.metrics
            self.model2_fin_activation = p2.model.fin_activation
        except AttributeError:
            self.model2_name = 'n/a'
            self.model2_class = 'n/a'
            self.model2_timestamp = 'n/a'
            self.model2_fullname = 'n/a'
            self.model2_opt_name = 'n/a'
            self.model2_lr = 'n/a'
            self.model2_loss = 'n/a'
            self.model2_acc = 'n/a'
            self.model2_fin_activation = 'n/a'

        try:
            self.model1_used_path = p1.model.model_used_path
        except AttributeError:
            self.model1_used_path = 'n/a'
        try:
            self.model2_used_path = p2.model.model_used_path
        except AttributeError:
            self.model2_used_path = 'n/a'

    def add_constants(self, parameters):
        """parameters = object with attributes of trainingparameters"""
        members = [attr for attr in dir(parameters) if not callable(getattr(parameters, attr)) and not attr.startswith("__")]
        # print(members)
        self.constants = {}
        for member in members:
            self.constants[member] = getattr(parameters, member)

        """self.constants = {
            "MIN_REWARD": MIN_REWARD,
            "EPISODES": EPISODES,
            "REWARD_WINNING": FiarGame.REWARD_WINNING,
            "REWARD_LOSING": FiarGame.REWARD_LOSING,
            "REWARD_TIE": FiarGame.REWARD_TIE,
            "REWARD_INVALID_MOVE": FiarGame.REWARD_INVALID_MOVE,
            "REWARD_STEP": FiarGame.REWARD_STEP,
            "DISCOUNT": DISCOUNT,
            "REPLAY_MEMORY_SIZE": REPLAY_MEMORY_SIZE,
            "MIN_REPLAY_MEMORY_SIZE": MIN_REPLAY_MEMORY_SIZE,
            "MINIBATCH_SIZE": MINIBATCH_SIZE,
            "UPDATE_TARGET_EVERY": UPDATE_TARGET_EVERY,
            "MEMORY_FRACTION": MEMORY_FRACTION,
            "epsilon": epsilon,
            "EPSILON_DECAY": EPSILON_DECAY,
            "MIN_EPSILON": MIN_EPSILON,
            "AGGREGATE_STATS_EVERY": AGGREGATE_STATS_EVERY,
            "SHOW_PREVIEW": SHOW_PREVIEW
        }"""

    def write_parameters_to_file(self):
        if self.log_flag is False:
            return  # do not write to file

        f = open(f'{self.logfilepath}.log', "a+")

        f.write(f"=================================================\n\n")
        f.write(f"loaded models at = {self.timenow}\n")
        f.write(f" timestamp = {self.timestamp}\n")

        # player/model info
        f.write(f"player1\n")
        f.write(f" class = {self.player1_class}\n")
        f.write(f" name = {self.player1_name}\n")
        f.write(f" model class = {self.model1_class}\n")
        f.write(f" model name = {self.model1_name}\n")
        f.write(f" model loss func = {self.model1_loss}\n")
        f.write(f" model optimizer = {self.model1_opt_name} lr={self.model1_lr}\n")
        f.write(f" model final activation = {self.model1_fin_activation}\n")
        f.write(f" model startstamp = {self.model1_timestamp}\n")
        f.write(f" model fullname = {self.model1_fullname}\n")
        f.write(f" model used path = {self.model1_used_path}\n")

        f.write(f"player2\n")
        f.write(f" class = {self.player2_class}\n")
        f.write(f" name = {self.player2_name}\n")
        f.write(f" model class = {self.model2_class}\n")
        f.write(f" model name = {self.model2_name}\n")
        f.write(f" model loss func = {self.model2_loss}\n")
        f.write(f" model optimizer = {self.model2_opt_name} lr={self.model2_lr}\n")
        f.write(f" model final activation = {self.model2_fin_activation}\n")
        f.write(f" model startstamp = {self.model2_timestamp}\n")
        f.write(f" model fullname = {self.model2_fullname}\n")
        f.write(f" model used path = {self.model2_used_path}\n")

        # constantsinfo
        for key, constant in self.constants.items():
            if isinstance(constant, dict):
                # constant is a dictionary
                f.write(f"{key}:\n")
                for k, v in constant.items():
                    f.write(f" {k} = {v}\n")
            else:
                f.write(f"{key} = {constant}\n")
        f.write("\n")

        # closing file
        f.close()

    def log_text_to_file(self, text):
        if self.log_flag is False:
            return  # do not write to file

        f = open(f'{self.logfilepath}.log', "a+")
        f.write(f"{text}\n")
        f.close()

    def write_to_csv(self):
        if self.log_flag is False:
            return  # do not write to file

        with open(f'{self.logfilepath}.csv', mode='a+', newline='') as parameters_file:
            parameter_writer = csv.writer(parameters_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            parameter_writer.writerow([self.timestamp, self.timenow, 1, self.player1_class, self.player1_name, self.model1_class, self.model1_name, self.model1_loss, self.model1_opt_name, self.model1_lr, self.model1_timestamp, self.model1_fullname, self.model1_used_path])
            parameter_writer.writerow([self.timestamp, self.timenow, 2, self.player2_class, self.player2_name, self.model2_class, self.model2_name, self.model2_loss, self.model2_opt_name, self.model2_lr, self.model2_timestamp, self.model2_fullname, self.model2_used_path])
        parameters_file.close()
