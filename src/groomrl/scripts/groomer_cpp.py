# This file is part of GroomRL by S. Carrazza and F. A. Dreyer

"""
    groomer_cpp.py: the entry point for groomrl-cpp.
"""
from groomrl.keras_to_cpp import keras_to_cpp, check_model
from groomrl.models import load_runcard
from keras.models import model_from_json
import pickle, json
import argparse, os, ast

#----------------------------------------------------------------------
def main():
    """Starting point"""
    parser = argparse.ArgumentParser(description='Convert the groomer model to cpp file.')
    parser.add_argument('fit_folder', action='store', help='The fit folder')
    args = parser.parse_args()

    # set output folder
    folder = args.fit_folder.strip('/')
    output = '%s/cpp' % folder

    # loading json card
    runcard = load_runcard('%s/runcard.json' % folder)
    check_model(runcard['groomer_agent'])

    # create output folder
    os.mkdir(output)
        
    # read architecture card
    with open('%s/model.json' % folder) as f:
        arch = json.load(f)
    arch_dic=ast.literal_eval(arch.replace('true','True').replace('null','None'))
    
    modelwgts_fn = '%s/weights.h5' % folder
    model = model_from_json(arch)
    model.load_weights(modelwgts_fn)

    cpp_fn = '%s/model.nnet'%output
    keras_to_cpp(model, arch_dic['config']['layers'], cpp_fn)
