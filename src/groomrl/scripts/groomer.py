# This file is part of GroomRL by S. Carrazza and F. A. Dreyer

"""
    groomer.py: the entry point for the groomrl.
"""
from groomrl.read_data import Jets
from groomrl.models import build_and_train_model, load_runcard
from groomrl.diagnostics import plot_mass, plot_lund
from groomrl.Groomer import Groomer
from groomrl.keras_to_cpp import keras_to_cpp, check_model
from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
from time import time
from shutil import copyfile
from copy import deepcopy
import os, argparse, pickle, pprint, json, ast, shutil
#import cProfile

#----------------------------------------------------------------------
def run_hyperparameter_scan(search_space):
    """Running a hyperparameter scan using hyperopt.
    TODO: implement cross-validation, e.g. k-fold, or randomized cross-validation.
    TODO: use test data as hyper. optimization goal.
    TODO: better import/export for the best model, wait to DQNAgentGroom
    """

    print('[+] Performing hyperparameter scan...')
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
    parser.add_argument('runcard', action='store', nargs='?', default=None,
                        help='A json file with the setup.')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='The input model.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='The output folder.')
    parser.add_argument('--plot',action='store_true',dest='plot')
    parser.add_argument('--force', '-f', action='store_true',dest='force',
                        help='Overwrite existing files if present')
    parser.add_argument('--cpp',action='store_true',dest='cpp')
    parser.add_argument('--data', type=str, default=None, dest='data',
                        help='Data on which to apply the groomer.')
    parser.add_argument('--nev', '-n', type=float, default=-1,
                        help='Number of events.')
    args = parser.parse_args()

    # check that input is coherent
    if (not args.model and not args.runcard) or (args.model and args.runcard):
        raise ValueError('Invalid options: requires either input runcard or model.')
    elif args.runcard and not os.path.isfile(args.runcard):
        raise ValueError('Invalid runcard: not a file.')
    elif args.model and not (args.plot or args.cpp or args.data):
        raise ValueError('Invalid options: no actions requested.')
    if args.force:
        print('WARNING: Running with --force option will overwrite existing model')

    if args.runcard:
        # load json
        setup = load_json(args.runcard)
        
        # create output folder
        base = os.path.basename(args.runcard)
        out = os.path.splitext(base)[0]
        if args.output is not None:
            out = args.output
        try:
            makedir(out)
        except Exception as error:
            if args.force:
                print(f'WARNING: Overwriting {out} with new model')
                shutil.rmtree(out)
                makedir(out)
            else:
                print(error)
                print('Delete or run with "--force" to overwrite.')
                exit(-1)
        setup['output'] = out
        
        # copy runcard to output folder
        copyfile(args.runcard, f'{out}/input-runcard.json')
        
        # groomer common environment setup
        if setup.get('scan'):
            groomer_agent_setup = run_hyperparameter_scan(setup)
        else:
            # create the DQN agent and train it.
            groomer_agent_setup = setup
        
        print('[+] Training best model:')
        dqn = build_and_train_model(groomer_agent_setup)
        
        # save the final runcard
        with open(f'{out}/runcard.json','w') as f:
            json.dump(groomer_agent_setup, f, indent=4)
        
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
        # define the folder where to do the plotting/cpp conversation
        folder = setup['output']

    elif args.model:
        folder = args.model.strip('/')
        # loading json card
        setup = load_runcard('%s/runcard.json' % folder)
        groomer_agent_setup = setup
        # loading groomer
        groomer = Groomer()
        modelwgts_fn = '%s/weights.h5' % folder
        modeljson_fn = '%s/model.json' % folder
        groomer.load_with_json(modeljson_fn, modelwgts_fn)
        
    # if requested, add plotting
    if args.plot:
        plotdir='%s/plots' % folder
        try:
            makedir(plotdir)
        except:
            print(f'[+] Ignoring plot instruction: {plotdir} already exists')
        else:
            print(f'[+] Creating test plots in {plotdir}')
            # generating invmass plot
            plot_mass(groomer, setup['test']['fn'],
                      mass_ref=setup['groomer_env']['mass'],
                      output_folder=plotdir, nev=args.nev)
            # generate lund plane plot
            plot_lund(groomer, setup['test']['fn'],
                      output_folder=plotdir, nev=args.nev)
            
            if 'fn_bkg' in setup['test']:
                # generating plots for the background
                plot_mass(groomer, setup['test']['fn_bkg'],
                          mass_ref=setup['groomer_env']['mass'],
                          output_folder=plotdir, nev=args.nev, background=True)
                plot_lund(groomer, setup['test']['fn_bkg'], output_folder=plotdir,
                          nev=args.nev, background=True)


    # if a data set was given as input, produce plots from it
    if args.data:
        fn = os.path.basename(args.data).split(os.extsep)[0]
        plotdir='%s/%s' % (folder, fn)
        try:
            makedir(plotdir)
        except:
            print(f'[+] Ignoring data instruction: {plotdir} already exists')
        else:
            print(f'[+] Creating mass and lune plane plots in {plotdir}')
            # generating invmass plot
            plot_mass(groomer, args.data, mass_ref=setup['groomer_env']['mass'],
                      output_folder=plotdir, nev=args.nev)
            # generate lund plane plot
            plot_lund(groomer, args.data, output_folder=plotdir, nev=args.nev)

    # if requested, add cpp output
    if args.cpp:
        check_model(groomer_agent_setup['groomer_agent'])
        cppdir = '%s/cpp' % folder
        try:
            makedir(cppdir)
        except:
            print(f'[+] Ignoring cpp instruction: {cppdir} already exists')
        else:
            print(f'[+] Adding cpp model in {cppdir}')
            cpp_fn = '%s/model.nnet' % cppdir
            arch_dic=ast.literal_eval(groomer.model.to_json()
                                      .replace('true','True')
                                      .replace('null','None'))
            keras_to_cpp(groomer.model, arch_dic['config']['layers'], cpp_fn)
            
