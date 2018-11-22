from create_image import Jets
from models import build_and_train_model
from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
from time import time
import os, argparse, pickle, pprint, json
from shutil import copyfile


#----------------------------------------------------------------------
def run_hyperparameter_scan(search_space):
    """Running a hyperparameter scan using hyperopt.
    TODO: implement cross-validation, e.g. k-fold, or randomized cross-validation.
    TODO: use test data as hyper. optimization goal.
    TODO: better import/export for the best model, wait to DQNAgentGroom
    """

    print('[+] Performing hyperparamter scan...')
    if search_space['cluster']['enable']:
        url = search_space['cluster']['url']
        key = search_space['cluster']['exp_key']
        trials = MongoTrials(url, exp_key=key)
    else:
        trials = Trials()
    best = fmin(build_and_train_model, search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    
    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)

    log = '%s/hyperopt_log_{}.pickle'.format(time()) % search_space['output']
    with open(log,'wb') as wfp:
        print(f'[+] Saving trials in {log}')
        pickle.dump(trials.trials, wfp)

    # disable scan for final fit
    best_setup['scan'] = False
    return best_setup


#----------------------------------------------------------------------
def main(setup):
    # groomer common environment setup
    if setup.get('scan'):
        groomer_agent_setup = run_hyperparameter_scan(setup)
    else:
        # create the DQN agent and train it.
        groomer_agent_setup = setup
    
    print('[+] Training best model:')
    dqn = build_and_train_model(groomer_agent_setup)
    
    fnres = '%s/test_predictions.pickle' % setup['output']

    print('[+] Done with training, now testing on sample set')
    if os.path.exists(fnres):
        os.remove(fnres)
    
    # now use model trained by DQN to groom test sample
    groomer = dqn.groomer()
    reader = Jets(setup['groomer_env']['testfn'], 10000)
    events = reader.values()
    groomed_jets = []
    for jet in events:
        groomed_jets.append(groomer(jet))
    with open(fnres,'wb') as wfp:
        pickle.dump(groomed_jets, wfp)


def load_json(runcard_file):
    """Loads json, execute python expressions, and sets
    scan flags accordingly to the syntax.
    """
    with open(runcard_file, 'r') as f:
        runcard = json.load(f)
    runcard['scan'] = False
    for key, value in runcard.get('groomer_agent').items():
        if 'hp' in str(value):
            runcard['groomer_agent'][key] = eval(value)
            runcard['scan'] = True
    return runcard


def makedir(folder):
    """Create directory."""
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:    
        raise Exception('Output folder already exists.')


#---------------------------------------------------------------------- 
if __name__ == "__main__":
    """Parsing command line arguments"""
    # read command line arguments
    parser = argparse.ArgumentParser(description='Train an ML groomer.')
    parser.add_argument('runcard', action='store', help='A json file with the setup.')
    parser.add_argument('--output', '-o', type=str, default=None, help='The output folder.')
    args = parser.parse_args()

    # load json
    setup = load_json(args.runcard)

    # create output folder
    base = os.path.basename(args.runcard)
    out = os.path.splitext(base)[0] 
    if args.output is not None:
        out = args.output
    makedir(out)
    setup['output'] = out

    # copy runcard to output folder
    copyfile(args.runcard, f'{out}/{base}')

    # run main
    main(setup)
