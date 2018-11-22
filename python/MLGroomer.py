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
from hyperopt import STATUS_OK
from time import time

import os, argparse, pickle


#---------------------------------------------------------------------- 
# https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
def build_model(hps):
    """Create a DQN agent to be used on lund inputs."""

    print('Constructing DQN agent, model setup:', hps)
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
    
    # set up the DQN agent
    memory = SequentialMemory(limit=500000, window_length=1)
    policy = BoltzmannQPolicy()
    agent = DQNAgentGroom(model=model, nb_actions=hps['nb_actions'],
                          memory=memory, nb_steps_warmup=500,
                          target_model_update=1e-2, policy=policy)
    agent.compile(Adam(lr=hps['learning_rate']), metrics=['mae'])
    
    return agent


#---------------------------------------------------------------------- 
def build_and_train_model(groomer_env, groomer_agent_setup):
    """Run a test model"""
    dqn = build_model(groomer_agent_setup)

    logdir = 'logs/{}'.format(time())
    print(f'Constructing tensorboard log in {logdir}')
    tensorboard = TensorBoard(log_dir=logdir)

    print('Fitting DQN agent...')
    r = dqn.fit(groomer_env, nb_steps=groomer_agent_setup['nstep'],
                visualize=False, verbose=1, callbacks=[tensorboard])

    # After training is done, we save the final weights.
    model_name = '../models/DQN_%s_nev%i_nstep%i.h5' % (groomer_agent_setup['architecture'], 
                                                        groomer_env.nev, groomer_agent_setup['nstep'])
    print(f'Saving weights to {model_name}')
    dqn.save_weights(model_name, overwrite=True)

    # compute nominal reward after training
    loss = np.max(np.median(r.history['episode_reward']))
    print(f'MAX MEDIAN REWARD: {loss}')

    return loss, dqn


#----------------------------------------------------------------------
def run_hyperparameter_scan(groomer_env):
    """Running a hyperparameter scan using hyperopt.
    TODO: implement cross-validation, e.g. k-fold, or randomized cross-validation.
    TODO: use test data as hyper. optimization goal.
    TODO: better import/export for the best model, wait to DQNAgentGroom
    """
    # control scan parameters
    # change this flags
    search_space = {
        'nb_actions': 2,
        'architecture': hp.choice('architecture', ['Dense', 'LSTM']),
        'learning_rate': hp.loguniform('learning_rate', -10, 0),
        'nstep': hp.choice('nstep', [100, 1000]),
        'input_dim': groomer_env.observation_space.shape
    }

    print('Performing hyperparamter scan...')
    trials = Trials()
    best = fmin(lambda p: {'loss': -build_and_train_model(groomer_env, p)[0], 'status': STATUS_OK},
                search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    
    best_setup = space_eval(search_space, best)
    print('\nBest scan setup:')
    print(best_setup)

    log = 'hyperopt_log_{}.pickle'.format(time())
    with open(log,'wb') as wfp:
        print(f'Saving trials in {log}')
        pickle.dump(trials.trials, wfp)


#----------------------------------------------------------------------
def main(args):
    # groomer common environment setup
    groomer_env = GroomEnv(args.fn, mass=args.mass,
                           mass_width=args.width, nev=args.nev, target_prec=0.05)
    if args.scan:
        run_hyperparameter_scan(groomer_env)
    else:
        # create the DQN agent and train it.
        groomer_agent_setup = {
            'nb_actions': 2,
            'architecture': 'LSTM' if args.lstm else 'Dense',
            'learning_rate': 1e-3,
            'nstep': args.nstep,
            'input_dim': groomer_env.observation_space.shape
        }
        _, dqn = build_and_train_model(groomer_env, groomer_agent_setup)

        if args.testname:
            fnres = args.testname
        else:
            fnres = 'test_%s.pickle' % 'LSTM' if args.lstm else 'Dense'

        print('Done with training, now testing on sample set')
        if os.path.exists(fnres):
            os.remove(fnres)
            
        # now use model trained by DQN to groom test sample
        groomer = dqn.groomer()
        reader = Jets(args.testfn, 10000)
        events = reader.values()
        groomed_jets = []
        for jet in events:
            groomed_jets.append(groomer(jet))
        with open(fnres,'wb') as wfp:
            pickle.dump(groomed_jets, wfp)


#---------------------------------------------------------------------- 
if __name__ == "__main__":
    """Parsing command line arguments"""
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
    parser.add_argument('--testname',type=str, default=None)
    parser.add_argument('--scan', action='store_true', dest='scan')

    args = parser.parse_args()
    main(args)
