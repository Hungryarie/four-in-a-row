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
        self.chosen_column = [[], [], []]

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
        self.chosen_column = [[], [], []]

    def aggregate_stats(self, calc_steps):
        try:
            self.win_ratio = self.win_count / (self.win_count + self.loose_count)
        except ZeroDivisionError:
            self.win_ratio = 0.5
        self.turns_count = self.turns_count / calc_steps
        total_wins = max(self.count_horizontal + self.count_vertical + self.count_dia_left + self.count_dia_right, 1)
        self.count_horizontal = self.count_horizontal / total_wins
        self.count_vertical = self.count_vertical / total_wins
        self.count_dia_left = self.count_dia_left / total_wins
        self.count_dia_right = self.count_dia_right / total_wins
        self.average_reward = sum(self.ep_rewards[-calc_steps:]) / len(self.ep_rewards[-calc_steps:])
        self.min_reward = min(self.ep_rewards[-calc_steps:])
        self.max_reward = max(self.ep_rewards[-calc_steps:])
        self.std_reward = np.std(self.ep_rewards[-calc_steps:])


class ModelLog():
    def __init__(self, file, log_flag=True):
        self.logfilepath = file
        self.log_flag = log_flag

        self.timestamp = int(time.time())
        self.timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.player_dict = []   # list of dicts.
        self.model_dict = []    # list of dicts.

    def add_player_info(self, player):
        """add player info to log\n\n
        input:\n
         - player -> instance of player\n
        output:\n
        - self.player_dict"""
        p_dict = {'player_id': player.player_id}
        p_dict['player_class'] = player.player_class
        p_dict['player_name'] = player.name
        self.player_dict.append(p_dict)

    def add_modelinfo(self, player_id, model, model_info):
        """add model info to dictionary\n\n
        input:
         - player_id -> to assign the model to the correct player\n
         - model -> model where the info is distilled from.\n
         - model_info -> extra info about the model\n
         output:\n
          - self.model_dict is filled"""

        dic = {}
        dic['player_id'] = player_id
        dic['model_info'] = model_info
        dic['model_loss'] = model.loss
        dic['model_acc'] = model.metrics
        dic.update(model.hyper_dict)  # extent dic with model hyper dictionary (eg. name, class, timestamp, optimizer name, learning rate, final activation layertype etc.)
        dic['model_fullname'] = f'{dic["model_class"]}_{dic["model_name"]}_startstamp{dic["timestamp"]}'
        self.model_dict.append(dic)

    def add_player_info_old(self, p1, p2):
        raise DeprecationWarning("use add_player_info instead")
        self.player1_class = p1.player_class
        self.player2_class = p2.player_class

        self.player1_name = p1.name
        self.player2_name = p2.name

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
        """add the constants parameters object to self.constants\n\n
        input:\n
         - parameters = object with attributes of trainingparameters"""

        members = [attr for attr in dir(parameters) if not callable(getattr(parameters, attr)) and not attr.startswith("__")]
        # print(members)
        self.constants = {}
        for member in members:
            self.constants[member] = getattr(parameters, member)

    def write_parameters_to_file(self):
        if self.log_flag is False:
            return  # do not write to file

        f = open(f'{self.logfilepath}.log', "a+")

        f.write(f"=================================================\n\n")
        f.write(f"loaded models at = {self.timenow}\n")
        f.write(f" timestamp = {self.timestamp}\n\n")
        """
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
        """
        # player en modelinfo
        for idx, dic in enumerate(self.player_dict):
            player_id = 0
            f.write(f"Player {idx + 1}:\n")
            for k, v in dic.items():
                f.write(f" {k} = {v}\n")
                if k == 'player_id':
                    player_id = v

            for m_dict in self.model_dict:
                if m_dict['player_id'] == player_id:
                    f.write(f"  model:\n")
                    for k, v in m_dict.items():
                        f.write(f"   {k} = {v}\n")
            f.write("\n")

        # constantsinfo
        for key, constant in self.constants.items():
            if isinstance(constant, dict):
                # if constant is a dictionary
                f.write(f"{key}:\n")
                for k, v in constant.items():
                    f.write(f" {k} = {v}\n")
            else:
                f.write(f"{key} = {constant}\n")
        f.write("\n")

        # closing file
        f.close()

    def get_info(self, player_id, attribute):
        """get the info from the self.model_dict or self.player_dict\n\n
        when attribute can't be fount. n/a is returned"""

        default = 'n/a'
        dict_list = [self.model_dict, self.player_dict]
        for lists in dict_list:
            for dic in lists:
                if dic['player_id'] == player_id:
                    value = dic.get(attribute, default)
                    if value != default:
                        return value

        return default

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

            # parameter_writer.writerow([self.timestamp, self.timenow, 1, self.player1_class, self.player1_name, self.model1_class, self.model1_name, self.model1_loss, self.model1_opt_name, self.model1_lr, self.model1_timestamp, self.model1_fullname, self.model1_used_path])
            # parameter_writer.writerow([self.timestamp, self.timenow, 2, self.player2_class, self.player2_name, self.model2_class, self.model2_name, self.model2_loss, self.model2_opt_name, self.model2_lr, self.model2_timestamp, self.model2_fullname, self.model2_used_path])
            for i in range(1, 3):
                parameter_writer.writerow([self.timestamp, self.timenow, i, self.get_info(i, "player_class"), self.get_info(i, "player_name"), self.get_info(i, "model_class"), self.get_info(i, "model_name"), self.get_info(i, "model_loss"), self.get_info(i, "model_opt_name"), self.get_info(i, "model_lr"), self.get_info(i, "model_timestamp"), self.get_info(i, "model_fullname"), self.get_info(i, "model_used_path")])
        parameters_file.close()
