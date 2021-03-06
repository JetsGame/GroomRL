# This file is part of GroomRL by S. Carrazza and F. A. Dreyer

from groomrl.JetTree import JetTree
import numpy as np
import math, json
from abc import ABC, abstractmethod
from rl.policy import GreedyQPolicy
from keras.models import model_from_json

#======================================================================
class AbstractGroomer(ABC):
    """AbstractGroomer class."""

    #----------------------------------------------------------------------
    def __call__(self, jet, returnTree = False):
        """Apply the groomer after casting the jet to a JetTree, and return groomed momenta."""
        # TODO: replace result by reclustered jet of all remaining constituents.
        if type(jet)==JetTree:
            tree = jet
        else:
            tree = JetTree(jet)
        self._groom(tree)
        if returnTree:
            return tree
        return tree.jet()

    #----------------------------------------------------------------------
    @abstractmethod
    def _groom(self, tree):
        pass
        
#======================================================================
class Groomer(AbstractGroomer):
    """Groomer class that acts on a JetTree using keras model and policy."""

    #---------------------------------------------------------------------- 
    def __init__(self, model=None, policy=GreedyQPolicy()):
        """Initialisation of the groomer."""
        self.model = model
        self.policy = policy

    #----------------------------------------------------------------------
    def _groom(self, tree):
        """Apply grooming to a jet."""
        if not tree.lundCoord:
            # current node has no subjets => no grooming
            return
        state=tree.state()
        # get an action from the policy using the state and model
        q_values = self.model.predict_on_batch(np.array([[state]])).flatten()
        action   = self.policy.select_action(q_values=q_values)
        if action==1:
            # call internal method to remove soft branch and replace
            # current tree node with harder branch
            tree.remove_soft()
            # now we groom the new tree, since both nodes have been changed
            self._groom(tree)
            
        else:
            # if we don't groom the current soft branch, then continue
            # iterating on both subjets
            if tree.harder:
                self._groom(tree.harder)
            if tree.softer:
                self._groom(tree.softer)

    #----------------------------------------------------------------------
    def load_with_json(self, jsonfile, weightfile):
        """
        Load model from a json file with the architecture, and an h5 file with weights.
        """
        # read architecture card
        with open(jsonfile) as f:
            arch = json.load(f)
        self.model = model_from_json(arch)
        self.model.load_weights(weightfile)

    #----------------------------------------------------------------------
    def save(self, filepath, overwrite=False, include_optimizer=True):
        """Save the model to file."""
        self.model.save(filepath, overwrite=overwrite,
                        include_optimizer=include_optimizer)

    #----------------------------------------------------------------------
    def load_model(self, filepath, custom_objects=None, compile=True):
        """Load model from file"""
        self.model = load_model(filepath, custom_objects=custom_objects,
                                compile=compile)

    #----------------------------------------------------------------------
    def save_weights(self, filepath, overwrite=False):
        """Save the weights of model to file."""
        self.model.save_weights(filepath, overwrite=overwrite)

    #----------------------------------------------------------------------
    def load_weights(self, filepath):
        """Load weights of model from file"""
        self.model.load_weights(filepath)


#======================================================================
class RSD(AbstractGroomer):
    """RSD applies Recursive Soft Drop grooming to a JetTree."""

    #----------------------------------------------------------------------
    def __init__(self, zcut=0.05, beta=1.0, R0=1.0):
        """Initialize RSD with its parameters."""
        self.zcut = zcut
        self.beta = beta
        self.R0   = R0

    #----------------------------------------------------------------------
    def _groom(self, tree):
        """Apply RSD grooming to a jet."""
        if not tree.lundCoord:
            # current node has no subjets => no grooming
            return
        state=tree.state()
        if not state.size>0:
            # current node has no subjets => no grooming
            return
        # check the SD condition
        z     = math.exp(state[0])
        delta = math.exp(state[1])
        remove_soft = (z < self.zcut * math.pow(delta/self.R0, self.beta))
        if remove_soft:
            # call internal method to remove soft branch and replace
            # current tree node with harder branch
            tree.remove_soft()
            # now we groom the new tree, since both nodes have been changed
            self._groom(tree)
        else:
            # if we don't groom the current soft branch, then continue
            # iterating on both subjets
            if tree.harder:
                self._groom(tree.harder)
            if tree.softer:
                self._groom(tree.softer)

