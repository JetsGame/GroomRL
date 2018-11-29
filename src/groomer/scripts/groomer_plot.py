"""
    groomer_plot.py: the entry point for groomer-plot.
"""
from groomer.Groomer import Groomer
from groomer.JetTree import *
from groomer.diagnostics import plot_mass, plot_lund
from groomer.models import build_model
import json
import argparse, os


#----------------------------------------------------------------------
def main():
    """Starting point"""
    parser = argparse.ArgumentParser(description='Diagnostics for groomer.')
    parser.add_argument('fit_folder', action='store', help='The fit folder')
    args = parser.parse_args()

    # building output folder
    folder = args.fit_folder.strip('/')
    output = '%s/plots' % folder
    os.mkdir(output)

    # loading json card
    with open('%s/runcard.json' % folder) as f:
        runcard = json.load(f)

    # loading groomer
    groomer = Groomer()
    modelwgts_fn = '%s/weights.h5' % folder
    modeljson_fn = '%s/model.json' % folder
    groomer.load_with_json(modeljson_fn, modelwgts_fn)

    sample_fn = runcard['testfn']

    # generating invmass plot
    plot_mass(groomer, sample_fn, mass_ref=runcard['groomer_env']['mass'],
              output_folder=output)

    # generate lund plane plot
    plot_lund(groomer, sample_fn, output_folder=output)
