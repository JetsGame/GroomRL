from GroomEnv import GroomEnv
from Groomer import Groomer
from create_image import Jets
import numpy as np

from DQNAgentGroom import DQNAgentGroom
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
from hyperopt import STATUS_OK
from time import time

import os, argparse, pickle, pprint
import models


#----------------------------------------------------------------------
def run_hyperparameter_scan(args):
    """Running a hyperparameter scan using hyperopt.
    TODO: implement cross-validation, e.g. k-fold, or randomized cross-validation.
    TODO: use test data as hyper. optimization goal.
    TODO: better import/export for the best model, wait to DQNAgentGroom
    """
    # control scan parameters
    # change this flags
    search_space = {
        'scan': True,
        'nb_actions': 2,
        'architecture': hp.choice('architecture', ['Dense', 'LSTM']),
        'learning_rate': hp.loguniform('learning_rate', -10, 0),
        'nstep': hp.choice('nstep', [100, 1000]),
        'groomer_env': vars(args)
    }

    print('[+] Performing hyperparamter scan...')
    if args.cluster:
        trials = MongoTrials('mongo://localhost:1234/groomer/jobs', exp_key='exp1')
    else:
        trials = Trials()
    best = fmin(models.build_and_train_model, search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    
    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)

    log = 'hyperopt_log_{}.pickle'.format(time())
    with open(log,'wb') as wfp:
        print(f'[+] Saving trials in {log}')
        pickle.dump(trials.trials, wfp)

    # disable scan for final fit
    best_setup['scan'] = False
    return best_setup


#----------------------------------------------------------------------
def main(args):
    # groomer common environment setup
    if args.scan:
        groomer_agent_setup = run_hyperparameter_scan(args)
    else:
        # create the DQN agent and train it.
        groomer_agent_setup = {
            'scan': False,
            'nb_actions': 2,
            'architecture': 'LSTM' if args.lstm else 'Dense',
            'learning_rate': 1e-3,
            'nstep': args.nstep,
            'groomer_env': vars(args)
        }
    
    print('[+] Training best model:')
    dqn = models.build_and_train_model(groomer_agent_setup)

    if args.testname:
        fnres = args.testname
    else:
        fnres = 'test_%s.pickle' % 'LSTM' if args.lstm else 'Dense'

    print('[+] Done with training, now testing on sample set')
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
    parser.add_argument('--cluster', action='store_true', dest='cluster')    

    args = parser.parse_args()
    main(args)
