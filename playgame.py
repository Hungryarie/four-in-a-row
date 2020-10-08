import numpy as np
import random
import matplotlib.pyplot as plt
import pylab
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# from keras.optimizers import Adam, SGD, RMSprop
from tqdm import tqdm
import os

import time
from datetime import datetime
import logging
import sys

# own file imports
import players
from game import FiarGame
from env import environment
from logandstats import Stats, ModelLog
from train import TrainAgent
from constants import TrainingParameters
from model import load_a_model, model1, model1b, model1c, model1d, model2, model3, model4a, model4b, model5
# functional API specific
from model import func_model1_small_conv, func_model1, func_model_duel1b, func_model_duel1b1, func_model_duel1b2, func_model2, func_model5, func_model5_duel1
from model import ACmodel1, ACmodel2, PolicyModel1, ACmodel2PHIL
from analyse import AnalyseModel


def train_in_class():
    # For more repetitive results
    random.seed(2)
    np.random.seed(2)
    tf.random.set_seed(2)

    #tf.compat.v1.disable_eager_execution()

    # load training parameters
    param = TrainingParameters()
    enriched = False

    # load environment
    env = environment(reward_dict=param.reward_dict)


    # load models
    input_shape = env.get_feature_size(enriched=enriched)  # get environment shape
    output_num = input_shape[1]
    #actor = func_model1(input_shape=input_shape, output_num=output_num,
    #                    par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001, clipnorm=1.0, clipvalue=0.25), par_metrics='accuracy', par_final_act='softmax', par_layer_multiplier=1)  # , par_layer_multiplier=2
    #critic = func_model2(input_shape=input_shape, output_num=1,
    #                     par_loss='mse', par_opt=Adam(lr=0.005, clipnorm=1.0, clipvalue=0.5), par_metrics='accuracy', par_final_act='linear', par_layer_multiplier=2)  # , par_layer_multiplier=1
    #policymodel = PolicyModel1(input_shape=input_shape, output_num=output_num,
    #                          par_loss=['categorical_crossentropy', 'mse'], par_opt=Adam(lr=0.001, clipnorm=1.0, clipvalue=0.25), par_metrics='accuracy', par_final_act='softmax', par_layer_multiplier=1)
    #acmodel = ACmodel2PHIL(input_shape=input_shape, output_num=output_num,
    #                       par_loss=['categorical_crossentropy', 'mse'], par_opt=[Adam(lr=0.0001, clipnorm=1.0, clipvalue=0.25), Adam(lr=0.00005, clipnorm=1.0, clipvalue=0.25)], par_metrics='accuracy', par_final_act='softmax', par_layer_multiplier=1)

    acmodel = ACmodel2(input_shape=input_shape, output_num=output_num,
                       par_loss=['loss_fn1', 'squared'], par_opt=[Adam(lr=0.0003, clipnorm=1.0, clipvalue=0.25)], par_metrics='accuracy', par_final_act='softmax', par_layer_multiplier=1)


    # load players
    #p1 = players.A2CAgent(actor_model=actor, critic_model=critic, discount=param.DISCOUNT,
    #                      enriched_features=True)
    #p1 = players.A2CAgent(actor_model=None, critic_model=None, twohead_model=acmodel, discount=param.DISCOUNT,
    #                      enriched_features=True)
    #p1 = players.PolicyAgent(models=policymodel, discount=param.DISCOUNT, enriched_features=True)
    #p1 = players.newA2CAgent(models=acmodel, discount=param.DISCOUNT, enriched_features=enriched)
    p1 = players.newestA2CAgent(models=acmodel, discount=param.DISCOUNT, enriched_features=enriched)
    p1.name = "A2C on training"
    #p2 = players.Selfplay(p1)
    #p2.name = "selfplay"
    p2 = players.Stick(persistent=False)
    p2.name = "sticky"

    p2.set_enriched_features(p1.enriched_features)      # set enriched_features status from p1 to p2
    description = f"enriched features ={enriched} entropyloss"

    env.add_players(p1, p2)
    training = TrainAgent(env, parameters=param, debug_flag=True)
    training.setup_training(train_description=description)
    training.run_training(start_id=p2.player_id)
    training.save_model(player=p1)  # save model after training


def batch_train():
    model_list = []
    """
    model_list.append(model5(input_shape=(6, 7, 1), output_num=7,
                      par_loss='mse', par_opt=Adam(lr=0.001), par_metrics='accuracy'))

    model_list.append(model2(input_shape=(6, 7, 1), output_num=7))
    """
    # model_list.append(func_model5_duel1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='mse', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'))

    # model_list.append(func_model_duel1b1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='logcosh', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'))
    """
    model_list.append(func_model_duel1b1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.01), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.001, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.01, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))

    model_list.append(func_model_duel1b2(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b2(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.01), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b2(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.001, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model_duel1b2(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.01, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))

    model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.001), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=Adam(lr=0.01), par_metrics='accuracy', par_final_act='linear'))
    model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
                      par_loss='logcosh', par_opt=SGD(lr=0.001, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))
    """
    model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
                                  par_loss='logcosh', par_opt=SGD(lr=0.01, momentum=0.9), par_metrics='accuracy', par_final_act='linear'))

    # model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='categorical_crossentropy', par_opt=Adam(lr=0.001), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'))
    # model_list.append(func_model1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='categorical_crossentropy', par_opt=Adam(lr=0.01), clipnorm=1.0, clipvalue=0.5, par_metrics='accuracy', par_final_act='linear'))

    # model_list.append(model5(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss=model5.huber_loss(), par_opt=Adam(lr=0.001), par_metrics='accuracy'))

    # model_list.append(func_model5(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss=func_model5.huber_loss(), par_opt=Adam(lr=0.001), par_metrics='accuracy'))

    # model_list.append(func_model5_duel1(input_shape=(6, 7, 1), output_num=7,
    #                  par_loss='mse', par_opt=Adam(lr=0.001), par_metrics='accuracy'))

    # clipnorm=1.0, clipvalue=0.5

    # model2 = load_a_model('models/func_model1_3xconv+2xdenseSMALL4x4(func)(mse^Adam^lr=0.001)_startstamp1569842535_episode7450__170.00max___66.60avg_-205.00min_1569845018.model')
    model2 = None
    for model in model_list:
        trainNN(p1_model=model, p2_model=model2,
                visualize_layers=False, debug_flag=True)


if __name__ == '__main__':
    # batch_train()
    # trainA2C()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    train_in_class()
