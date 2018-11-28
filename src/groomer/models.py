from groomer.GroomEnv import GroomEnv
import numpy as np

from groomer.DQNAgentGroom import DQNAgentGroom
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from hyperopt import STATUS_OK
from time import time

import pprint, json

def build_model(hps, input_dim):
    """Construct the underlying model used by the DQN."""
    model = Sequential()
    if hps['architecture']=='Dense':
        model.add(Flatten(input_shape=(1,) + input_dim))
        for i in range(hps['nb_layers']):
            model.add(Dense(hps['nb_units']))
            model.add(Activation('relu'))
        if hps['dropout']>0.0:
            model.add(Dropout(hps['dropout']))
        model.add(Dense(hps['nb_actions']))
        model.add(Activation('linear'))
    elif hps['architecture']=='LSTM':
        model.add(LSTM(hps['nb_units'], input_shape = (1,max(input_dim))))
        if hps['dropout']>0.0:
            model.add(Dropout(hps['dropout']))
        model.add(Dense(hps['nb_actions']))
        model.add(Activation('linear'))
    print(model.summary())
    return model

#---------------------------------------------------------------------- 
# https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
def build_dqn(hps, input_dim):
    """Create a DQN agent to be used on lund inputs."""

    print('[+] Constructing DQN agent, model setup:')
    pprint.pprint(hps)
    
    # set up the DQN agent
    model = build_model(hps, input_dim)
    memory = SequentialMemory(limit=500000, window_length=1)
    policy = BoltzmannQPolicy()
    agent = DQNAgentGroom(model=model, nb_actions=2,
                          memory=memory, nb_steps_warmup=500,
                          target_model_update=1e-2, policy=policy)
    agent.compile(Adam(lr=hps['learning_rate']), metrics=['mae'])
    
    return agent


#---------------------------------------------------------------------- 
def build_and_train_model(groomer_agent_setup):
    """Run a test model"""    
    env_setup = groomer_agent_setup.get('groomer_env')
    groomer_env = GroomEnv(env_setup)

    agent_setup = groomer_agent_setup.get('groomer_agent')
    dqn = build_dqn(agent_setup, groomer_env.observation_space.shape)
 
    logdir = '%s/logs/{}'.format(time()) % groomer_agent_setup['output']
    print(f'[+] Constructing tensorboard log in {logdir}')
    tensorboard = TensorBoard(log_dir=logdir)

    print('[+] Fitting DQN agent...')
    r = dqn.fit(groomer_env, nb_steps=agent_setup['nstep'],
                visualize=False, verbose=1, callbacks=[tensorboard])

    # After training is done, we save the final weights.
    weight_file = '%s/weights.h5' % groomer_agent_setup['output']
    print(f'[+] Saving weights to {weight_file}')
    dqn.save_weights(weight_file, overwrite=True)
    
    # save the model architecture in json
    model_file = '%s/model.json' % groomer_agent_setup['output']
    print(f'[+] Saving model to {model_file}')    
    with open(model_file, 'w') as outfile:
        json.dump(dqn.model.to_json(), outfile)

    if groomer_agent_setup['scan']:        
        # compute nominal reward after training
        loss = np.median(r.history['episode_reward'])
        print(f'[+] MAX MEDIAN REWARD: {loss}')
        res = {'loss': -loss, 'status': STATUS_OK}
    else:
        res = dqn       
    return res
