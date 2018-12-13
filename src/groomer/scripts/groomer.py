"""
    groomer.py: the entry point for the groomer.
"""
from groomer.read_data import Jets
from groomer.models import build_and_train_model, load_runcard
from groomer.diagnostics import plot_mass, plot_lund
from groomer.keras_to_cpp import keras_to_cpp, check_model
from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
from time import time
from shutil import copyfile
from copy import deepcopy
import os, argparse, pickle, pprint, json, ast
#import cProfile

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
    max_evals = search_space['cluster']['max_evals']
    best = fmin(build_and_train_model, search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

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
def load_json(runcard_file):
    """Loads json, execute python expressions, and sets
    scan flags accordingly to the syntax.
    """
    runcard = load_runcard(runcard_file)
    runcard['scan'] = False
    for key, value in runcard.get('groomer_env').items():
        if 'hp.' in str(value):
            runcard['groomer_env'][key] = eval(value)
            runcard['scan'] = True
    for key, value in runcard.get('groomer_agent').items():
        if 'hp.' in str(value):
            runcard['groomer_agent'][key] = eval(value)
            runcard['scan'] = True
    return runcard

#----------------------------------------------------------------------
def makedir(folder):
    """Create directory."""
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        raise Exception('Output folder %s already exists.' % folder)


#----------------------------------------------------------------------
def main():
    """Parsing command line arguments"""
    # read command line arguments
    parser = argparse.ArgumentParser(description='Train an ML groomer.')
    parser.add_argument('runcard', action='store', help='A json file with the setup.')
    parser.add_argument('--output', '-o', type=str, default=None, help='The output folder.')
    parser.add_argument('--plot',action='store_true',dest='plot')
    parser.add_argument('--cpp',action='store_true',dest='cpp')
    parser.add_argument('--nev', '-n', type=float, default=10000, help='Number of events.')
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
    copyfile(args.runcard, f'{out}/runcard.json')

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
    reader = Jets(setup['test']['fn'], args.nev)
    events = reader.values()
    groomed_jets = []
    for jet in events:
        gr_jet=deepcopy(groomer(jet))
        groomed_jets.append(gr_jet)
    with open(fnres,'wb') as wfp:
        pickle.dump(groomed_jets, wfp)

    # if requested, add plotting
    if args.plot:
        plotdir='%s/plots' % setup['output']
        makedir(plotdir)
        # generating invmass plot
        plot_mass(groomer, setup['test']['fn'], mass_ref=setup['groomer_env']['mass'],
                  output_folder=plotdir, nev=args.nev)
        # generate lund plane plot
        plot_lund(groomer, setup['test']['fn'], output_folder=plotdir, nev=args.nev)

        if 'fn_bkg' in setup['test']:
            # generating plots for the background
            plot_mass(groomer, setup['test']['fn_bkg'], mass_ref=setup['groomer_env']['mass'],
                      output_folder=plotdir, nev=args.nev, background=True)
            plot_lund(groomer, setup['test']['fn_bkg'], output_folder=plotdir,
                      nev=args.nev, background=True)


    # if requested, add cpp output
    if args.cpp:
        cppdir = '%s/cpp' % setup['output']
        os.mkdir(cppdir)
        cpp_fn = '%s/model.nnet' % cppdir
        arch_dic=ast.literal_eval(groomer.model.to_json().replace('true','True').replace('null','None'))
        check_model(groomer_agent_setup['groomer_agent'])
        keras_to_cpp(groomer.model, arch_dic['config']['layers'], cpp_fn)
