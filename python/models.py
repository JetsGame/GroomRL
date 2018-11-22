from GroomEnv import GroomEnv
from Groomer import Groomer
from create_image import Jets
import numpy as np

from DQNAgentGroom import DQNAgentGroom
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
from hyperopt import STATUS_OK
from time import time

import os, argparse, pickle, pprint


#---------------------------------------------------------------------- 
# https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
def build_model(hps, input_dim):
    """Create a DQN agent to be used on lund inputs."""

    print('[+] Constructing DQN agent, model setup:')
    pprint.pprint(hps)

    model = Sequential()
    if hps['architecture']=='Dense':
        model.add(Flatten(input_shape=(1,) + input_dim))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(150))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dropout(0.05))
        model.add(Dense(hps['nb_actions']))
        model.add(Activation('linear'))
    elif hps['architecture']=='LSTM':
        model.add(LSTM(64, input_shape = (1,max(input_dim))))
        model.add(Dropout(0.05))
        model.add(Dense(hps['nb_actions']))
        model.add(Activation('linear'))
    print(model.summary())
    
    # set up the DQN agent
    memory = SequentialMemory(limit=500000, window_length=1)
    policy = BoltzmannQPolicy()
    agent = DQNAgentGroom(model=model, nb_actions=hps['nb_actions'],
                          memory=memory, nb_steps_warmup=500,
                          target_model_update=1e-2, policy=policy)
    agent.compile(Adam(lr=hps['learning_rate']), metrics=['mae'])
    
    return agent


#---------------------------------------------------------------------- 
def build_and_train_model(groomer_agent_setup):
    """Run a test model"""    
    args = groomer_agent_setup['groomer_env']
    groomer_env = GroomEnv(args['fn'], mass=args['mass'],
                           mass_width=args['width'], 
                           nev=args['nev'], target_prec=0.05)

    dqn = build_model(groomer_agent_setup, 
                      groomer_env.observation_space.shape)

    logdir = 'logs/{}'.format(time())
    print(f'[+] Constructing tensorboard log in {logdir}')
    tensorboard = TensorBoard(log_dir=logdir)

    print('[+] Fitting DQN agent...')
    r = dqn.fit(groomer_env, nb_steps=groomer_agent_setup['nstep'],
                visualize=False, verbose=1, callbacks=[tensorboard])

    # After training is done, we save the final weights.
    model_name = '../models/DQN_%s_nev%i_nstep%i.h5' % (groomer_agent_setup['architecture'], 
                                                        groomer_env.nev, groomer_agent_setup['nstep'])
    print(f'[+] Saving weights to {model_name}')
    dqn.save_weights(model_name, overwrite=True)

    if groomer_agent_setup['scan']:        
        # compute nominal reward after training
        loss = np.max(np.median(r.history['episode_reward']))
        print(f'[+] MAX MEDIAN REWARD: {loss}')
        res = {'loss': -loss, 'status': STATUS_OK}
    else:
        res = dqn       
    return res