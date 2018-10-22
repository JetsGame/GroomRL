from GroomEnv import GroomEnv
import numpy as np

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.optimizers import Adam

import os

def model_construct(hps):
    """Construct a neural network to use with the DQN agent."""
    model = Sequential()
    if hps['architecture']=='Dense':
        model.add(Flatten(input_shape=(1,) + hps['input_dim']))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(hps['nb_actions']))
        model.add(Activation('linear'))
    elif hps['architecture']=='LSTM':
        model.add(LSTM(64, input_shape = (1,max(hps['input_dim']))))
        model.add(Dropout(0.1))
        model.add(Dense(hps['nb_actions']))
        model.add(Activation('linear'))
    print(model.summary())
    return model
    
# construct a DQN network to be used on lund input
# https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
def dqn_construct(hps):
    """Create a DQN agent with a simple model using dense layers."""
    # we build a very simple model consisting of 4 dense layers or LSTM
    model = model_construct(hps)
    
    # set up the DQN agent
    memory = SequentialMemory(limit=100000, window_length=1)
    policy = BoltzmannQPolicy()
    agent = DQNAgent(model=model, nb_actions=hps['nb_actions'],
                     memory=memory, nb_steps_warmup=500,
                     target_model_update=1e-2, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    
    return agent

def run_model(network='Dense', fn = '../constit-long.json.gz',
              nev=400000, outfn=None):
    """Run a test model"""

    # set up environment
    env = GroomEnv(fn, nev, outfn=outfn, low=np.array([0.0, -6.0]),
                   high=np.array([10.0, 8.0]), mass=80.385,
                   target_prec = 0.05, mass_width = 1.0)

    # hyperparameters
    dqn_hps = {
        'input_dim': env.observation_space.shape,
        'nb_actions': 2,
        'architecture': network,
        'description': 'test model',
        'model_name': 'DQN_Test_Model_%s' % network
    }

    print('Constructing DQN agent...')
    dqn = dqn_construct(dqn_hps)

    print('Fitting DQN agent...')
    dqn.fit(env, nb_steps=500000, visualize=False, verbose=1)

    print('Saving weights...')
    # After training is done, we save the final weights.
    dqn.save_weights('../models/'+dqn_hps['model_name']+'.h5', overwrite=True)

    return dqn
    
if __name__ == "__main__":
    network='Dense'
    # create the DQN agent and train it.
    dqn = run_model(network=network)

    fnres = 'test_%s.pickle' % network
    # create an environment for the test sample
    env = GroomEnv('../constit.json.gz', 500, outfn=fnres, low=np.array([0.0, -6.0]),
                   high=np.array([10.0, 8.0]), mass=80.385,
                   target_prec = 0.1, mass_width = 2)
    # test the groomer on 500 events (saved as "test.pickle")
    if os.path.exists(fnres):
        os.remove(fnres)
    dqn.test(env, nb_episodes=500, visualize=True)
