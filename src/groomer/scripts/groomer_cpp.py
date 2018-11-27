from groomer.Groomer import Groomer, RSD
from groomer.JetTree import *
from groomer.observables import *
from groomer.read_clustseq_json import Jets
from groomer.keras_to_cpp import keras_to_cpp, check_model
from groomer.models import build_model
from keras.models import model_from_json
import numpy as np
import pickle, json
import argparse, os, ast

#----------------------------------------------------------------------
def main():
    """Starting point"""
    parser = argparse.ArgumentParser(description='Convert the groomer model to cpp file.')
    parser.add_argument('fit_folder', action='store', help='The fit folder')
    parser.add_argument('-v',action='store_true',dest='verbose')
    args = parser.parse_args()

    # building output folder
    folder = args.fit_folder.strip('/')
    output = '%s/cpp' % folder
    os.mkdir(output)

    # loading json card
    with open('%s/runcard.json' % folder) as f:
        runcard = json.load(f)
        
    # read architecture card
    with open('%s/model.json' % folder) as f:
        arch = json.load(f)
    arch_dic=ast.literal_eval(arch.replace('true','True').replace('null','None'))
    
    modelwgts_fn = '%s/weights.h5' % folder
    model = model_from_json(arch)
    model.load_weights(modelwgts_fn)

    cpp_fn = '%s/model.nnet'%output
    check_model(runcard['groomer_agent'])
    keras_to_cpp(model, arch_dic['config']['layers'], cpp_fn, args.verbose)
