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
        self.msq     = (j1 + j2).m2()
        self.lnKt    = math.log(j2.pt()*delta)
        self.lnz     = math.log(z)
        self.lnDelta = math.log(delta)
        self.lnKappa = math.log(z * delta)
        self.psi     = math.atan((j1.rap() - j2.rap())/(j1.phi() - j2.phi()))

    #----------------------------------------------------------------------
    def state(self):
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
    def lundCoord(self):
        """Return LundCoordinates corresponding to current node."""
        if not self.harder or not self.softer:
            return None
        return LundCoordinates(self.harder.node, self.softer.node)


