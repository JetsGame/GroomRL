import fastjet as fj
import numpy as np
import math

#----------------------------------------------------------------------
class LundCoordinates:
    """
    LundCoordinates takes two subjets associated with a declustering,
    and store the corresponding Lund coordinates."""

    # number of dimensions for the state() method
    dimension = 2
    
    #----------------------------------------------------------------------
    def __init__(self, j1, j2):
        """Define a number of variables associated with the declustering."""
        delta = j1.delta_R(j2)
        z     = j2.pt()/(j1.pt() + j2.pt())
        self.lnKt    = math.log(j2.pt()*delta)
        self.lnMsq   = math.log((j1 + j2).m2())
        self.lnz     = math.log(z)
        self.lnDelta = math.log(delta)
        self.lnKappa = math.log(z * Delta)
        self.psi     = math.atan((j1.rap() - j2.rap())/(j1.phi() - j2.phi()))

    #----------------------------------------------------------------------
    def state():
        return np.array([self.lnz, self.lnDelta])


#----------------------------------------------------------------------
class JetTree:
    """JetTree keeps track of the tree structure of a jet declustering."""

    #----------------------------------------------------------------------
    def __init__(self, pseudojet, child=None):
        """Initialize a new node, and create its two parents if they exist."""
        self.harder = None
        self.softer = None
        # first define the current node
        self.node   = pseudojet
        # if it has a direct child (i.e. one level further up in the
        # tree), give a link to the corresponding tree object here
        self.child  = child
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        if self.node and self.node.has_parents(j1,j2):
            # order the parents in pt
            if (j2.pt() > j1.pt()):
                j1,j2=j2,j1
            # then create two new tree nodes with j1 and j2
            self.harder = JetTree(j1, self)
            self.softer = JetTree(j2, self)

    #----------------------------------------------------------------------
    def lundCoord():
        """Return LundCoordinates corresponding to current node."""
        if not self.harder or not self.softer:
            return None
        return LundCoordinates(self.harder.node, self.softer.node)


    
from abc import ABC, abstractmethod
from rl.policy import GreedyQPolicy
#----------------------------------------------------------------------
class AbtractGroomer(ABC):
    """BaseGroomer class."""
    def __remove_soft(tree):
        """Remove the softer branch of a JetTree node."""
        # start by removing softer parent momentum from the rest of the tree
        child = tree.child
        while(child):
            child.node-=tree.softer.node
            child=child.child 
        # then move the harder branch to the current node,
        # effectively deleting the soft branch
        newTree = tree.harder
        tree.node   = newTree.node
        tree.softer = newTree.softer 
        tree.harder = newTree.harder
        # NB: tree.child doesn't change, we are just moving up the part
        # of the tree below it

    #----------------------------------------------------------------------
    def __call__(self, tree):
        self.__groom(tree)


    #----------------------------------------------------------------------
    @abstractmethod
    def __groom(self, tree):
        pass
        
#----------------------------------------------------------------------
class Groomer(BaseGroomer):
    """Groomer class that acts on a JetTree using keras model and policy."""

    #---------------------------------------------------------------------- 
    def __init__(self, model, policy=GreedyQPolicy()):
        """Initialisation of the groomer."""
        # read in the events
        self.policy = policy
        self.model  = model

    #----------------------------------------------------------------------
    def __groom(self, tree):
        """Apply grooming to a jet."""
        state    = tree.lundCoord().state()
        if not state:
            # current node has no subjets => no grooming
            return
        # get an action from the policy using the state and model
        q_values = self.model.predict_on_batch(np.array([[state]])).flatten()
        action   = self.policy.select_action(q_values=q_values)
        if action==1:
            # call internal method to remove soft branch and replace
            # current tree node with harder branch
            self.__remove_soft(tree)
            # now we groom the new tree, since both nodes have been changed
            self.__groom(tree)
            
        else:
            # if we don't groom the current soft branch, then continue
            # iterating on both subjets
            if tree.harder:
                self.__groom(tree.harder)
            if tree.softer:
                self.__groom(tree.softer)


#----------------------------------------------------------------------
class GroomerRSD:
    """GroomerRSD applies Recursive Soft Drop grooming to a JetTree."""

    #----------------------------------------------------------------------
    def __init__(self, zcut=0.05, beta=1.0, R0=1.0):
        """Initialize RSD with its parameters."""
        self.zcut = zcut
        self.beta = beta
        self.R0   = R0

    #----------------------------------------------------------------------
    def __call__(self, tree):
        """Apply RSD grooming to a jet."""
        state  = tree.lundCoord().state()
        if not state:
            # current node has no subjets => no grooming
            return
        # check the SD condition
        z     = math.exp(state[0])
        delta = math.exp(state[1])
        remove_soft = (z < self.zcut * math.pow(delta/self.R0, self.beta))
        if remove_soft:
            # call internal method to remove soft branch and replace
            # current tree node with harder branch
            self.__remove_soft(tree)
            # now we groom the new tree, since both nodes have been changed
            self.__groom(tree)
        else:
            # if we don't groom the current soft branch, then continue
            # iterating on both subjets
            if tree.harder:
                self.__groom(tree.harder)
            if tree.softer:
                self.__groom(tree.softer)
