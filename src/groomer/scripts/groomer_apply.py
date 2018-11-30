"""
    groomer_apply.py: the entry point for groomer-apply.
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
    parser.add_argument('data', action='store', help='The data file')
    parser.add_argument('--nev', '-n', type=float, default=10000, help='Number of events.')
    args = parser.parse_args()

    # building output folder
    folder = args.fit_folder.strip('/')
    fn = os.path.basename(args.data).split(os.extsep)[0]
    output = '%s/%s' % (folder,fn)
    os.mkdir(output)

    # loading json card
    runcard = load_runcard('%s/runcard.json' % folder)

    # loading groomer
    groomer = Groomer()
    modelwgts_fn = '%s/weights.h5' % folder
    modeljson_fn = '%s/model.json' % folder
    groomer.load_with_json(modeljson_fn, modelwgts_fn)

    # generating invmass plot
    plot_mass(groomer, args.data, mass_ref=runcard['groomer_env']['mass'],
              output_folder=output, nev=args.nev)

    # generate lund plane plot
    plot_lund(groomer, args.data, output_folder=output, nev=args.nev)
