#!/usr/bin/env python
import pickle
import argparse
import matplotlib.pyplot as plt


#---------------------------------------------------------------------- 
def main(args):
    """Load trials and generate plots"""    
    with open(args.trials, 'rb') as f:
        trials = pickle.load(f)
    
    # plot loss
    nplots = len(trials[0]['misc']['vals'].keys())
    f, axs = plt.subplots(1, nplots+1, sharey=True, figsize=(30,10))
    x = [t['tid'] for t in trials]
    loss = [t['result']['loss'] for t in trials]
    axs[0].scatter(x, loss)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    
    # plot features
    for p, k in enumerate(trials[0]['misc']['vals'].keys()):
        val = [t['misc']['vals'][k] for t in trials]
        axs[p+1].scatter(val, loss)
        axs[p+1].set_xlabel(k)

    plt.savefig(f'{args.trials}.png')


#---------------------------------------------------------------------- 
if __name__ == "__main__":
    """read command line arguments"""
    parser = argparse.ArgumentParser(description='Train an ML groomer.')
    parser.add_argument('trials', help='Pickle file with trials.')
    args = parser.parse_args()
    main(args)
