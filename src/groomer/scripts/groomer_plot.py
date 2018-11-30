"""
    groomer_plot.py: the entry point for groomer-plot.
"""
from groomer.Groomer import Groomer
from groomer.JetTree import *
from groomer.diagnostics import plot_mass, plot_lund
from groomer.models import load_runcard
import json
import argparse, os


#----------------------------------------------------------------------
def main():
    """Starting point"""
    parser = argparse.ArgumentParser(description='Diagnostics for groomer.')
    parser.add_argument('fit_folder', action='store', help='The fit folder')
    parser.add_argument('--nev', '-n', type=float, default=10000, help='Number of events.')
    args = parser.parse_args()

    # building output folder
    folder = args.fit_folder.strip('/')
    output = '%s/plots' % folder
    os.mkdir(output)

    # loading json card
    runcard = load_runcard('%s/runcard.json' % folder)

    # loading groomer
    groomer = Groomer()
    modelwgts_fn = '%s/weights.h5' % folder
    modeljson_fn = '%s/model.json' % folder
    groomer.load_with_json(modeljson_fn, modelwgts_fn)

    # generating invmass plot
    plot_mass(groomer, runcard['test']['fn'], mass_ref=runcard['groomer_env']['mass'],
              output_folder=output, nev=args.nev)

    # generate lund plane plot
    plot_lund(groomer, runcard['test']['fn'], output_folder=output, nev=args.nev)

    if 'fn_bkg' in runcard['test']:
        # generating plots for the background
        plot_mass(groomer, runcard['test']['fn_bkg'], mass_ref=runcard['groomer_env']['mass'],
                  output_folder=output, nev=args.nev, background=True)
        plot_lund(groomer, runcard['test']['fn_bkg'], output_folder=output,
                  nev=args.nev, background=True)
