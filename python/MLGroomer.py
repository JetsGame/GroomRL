from GroomEnv import GroomEnv
import numpy as np

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import os, argparse

#---------------------------------------------------------------------- 
def model_construct(hps):
    """Construct a neural network to use with the DQN agent."""
    model = Sequential()
    if hps['architecture']=='Dense':
        model.add(Flatten(input_shape=(1,) + hps['input_dim']))
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
        model.add(LSTM(64, input_shape = (1,max(hps['input_dim']))))
        model.add(Dropout(0.05))
        model.add(Dense(hps['nb_actions']))
        model.add(Activation('linear'))
    print(model.summary())
    return model
    
#---------------------------------------------------------------------- 
# https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
def dqn_construct(hps):
    """Create a DQN agent to be used on lund inputs."""
    # we build a very simple model consisting of 4 dense layers or LSTM
    model = model_construct(hps)
    
    # set up the DQN agent
    memory = SequentialMemory(limit=500000, window_length=1)
    policy = BoltzmannQPolicy()
    agent = DQNAgent(model=model, nb_actions=hps['nb_actions'],
                     memory=memory, nb_steps_warmup=500,
                     target_model_update=1e-2, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    
    return agent

#---------------------------------------------------------------------- 
def run_model(network, fn, mass, width, nstep, nev=-1, logname='log'):
    """Run a test model"""

    # set up environment
    env = GroomEnv(fn, mass=mass, mass_width=width, nev=nev, target_prec=0.05)

    # hyperparameters
    dqn_hps = {
        'input_dim': env.observation_space.shape,
        'nb_actions': 2,
        'architecture': network,
        'description': 'test model',
        'model_name': 'DQN_%s_nev%i_nstep%i' % (network, nev, nstep)
    }

    print('Constructing DQN agent...')
    dqn = dqn_construct(dqn_hps)

    print('Constructing callbacks')
    tensorboard = TensorBoard(log_dir='logs/%s' % logname)

    print('Fitting DQN agent...')
    dqn.fit(env, nb_steps=nstep, visualize=False, verbose=1, callbacks=[tensorboard])

    print('Saving weights...')
    # After training is done, we save the final weights.
    dqn.save_weights('../models/'+dqn_hps['model_name']+'.h5', overwrite=True)

    return dqn, env
    
#---------------------------------------------------------------------- 
if __name__ == "__main__":
    # read command line arguments
    parser = argparse.ArgumentParser(description='Train an ML groomer.')
    parser.add_argument('--lstm',action='store_true',dest='lstm')
    parser.add_argument('--nev',type=int, default=500000,dest='nev')
    parser.add_argument('--nstep',type=int, default=500000,dest='nstep')
    parser.add_argument('--file',action='store',
                        default='../constit-long.json.gz',dest='fn')
    parser.add_argument('--testfile',action='store',
                        default='../sample_WW_2TeV_CA.json.gz',
                        dest='testfn')
    parser.add_argument('--massgoal',type=float, default=80.385,dest='mass')
    parser.add_argument('--masswidth',type=float, default=1.0,dest='width')
    parser.add_argument('--logname',type=str, default='DEFAULT')
    parser.add_argument('--testname',type=str, default=None)

    args = parser.parse_args()

    if args.lstm:
        network='LSTM'
    else:
        network='Dense'
    # create the DQN agent and train it.
    dqn, env = run_model(network, args.fn, args.mass, args.width, args.nstep, args.nev, args.logname)

    if args.testname:
        fnres = args.testname
    else:
        fnres = 'test_%s.pickle' % network
        
    env.testmode(fnres, args.testfn)
    # test the groomer on 5000 events (saved as "test_network.pickle")
    if os.path.exists(fnres):
        os.remove(fnres)
    print('Done with training, now testing on sample set')
    dqn.test(env, nb_episodes=10000, visualize=True, verbose=0)
