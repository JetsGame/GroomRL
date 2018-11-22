from tools import declusterings, kinematics_node
from rl.policy import GreedyQPolicy

import numpy as np
import math
#----------------------------------------------------------------------
class Groomer:
    """Class to handle jet grooming using an internal keras model and policy."""
    #---------------------------------------------------------------------- 
    def __init__(self, model, policy=GreedyQPolicy()):
        """Initialisation of the groomer."""
        # read in the events
        self.policy = policy
        self.model  = model

    #----------------------------------------------------------------------
    def __call__(self, jet):
        """Apply grooming to a jet and returned groomed kinematics."""
        # get a declustering list
        declusts = declusterings(jet)
        groomed_jet = [jet.px(), jet.py(), jet.pz(), jet.E()]
        #grooming steps
        groomed_branches = []
        for declust in declusts:
            node, children, tag, parents, j1, j2 = declust
            if not set(children+[tag]).isdisjoint(groomed_branches):
                # if this branch is already groomed, move to next node
                continue
            state    = kinematics_node(declust)
            q_values = self.model.predict_on_batch(np.array([[state]])).flatten()
            action   = self.policy.select_action(q_values=q_values)

            if action==1:
                # remove the soft emission from final groomed jet four-momentum
                groomed_jet =  [a - b for a, b in zip(groomed_jet, j2)]
                # add soft subjet ID to the list of groomed branches
                if parents[1]>0:
                    groomed_branches+=[parents[1]]
        # return four-momentum of groomed jet
        return groomed_jet

    def save(self, filepath, overwrite=False, include_optimizer=True):
        """Save the model to file."""
        self.model.save(filepath, overwrite=overwrite,
                        include_optimizer=include_optimizer)

    def load_model(self, filepath, custom_objects=None, compile=True):
        """Load model from file"""
        self.model = load_model(filepath, custom_objects=custom_objects,
                                compile=compile)

    def save_weights(self, filepath, overwrite=False):
        """Save the weights of model to file."""
        self.model.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        """Load weights of model from file"""
        self.model.load_weights(filepath)
