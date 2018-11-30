import fastjet as fj
import numpy as np
import math

#======================================================================
class LundCoordinates:
    """
    LundCoordinates takes two subjets associated with a declustering,
    and store the corresponding Lund coordinates."""

    # number of dimensions for the state() method
    dimension = 2
    # the lower and upper bounds of the state vector
    low  = np.array([-10.0, -8.0])
    high = np.array([0.0, 0.0])
    # internal array for resizing
    __low_full  = np.array([-10.0, -8.0, -4.0, -1.5708, 0.0])
    __high_full = np.array([0.0, 0.0, 8.0, 1.5708, 8.0])
    
    #----------------------------------------------------------------------
    def __init__(self, j1, j2):
        """Define a number of variables associated with the declustering."""
        delta = j1.delta_R(j2)
        z     = j2.pt()/(j1.pt() + j2.pt())
        self.lnm     = 0.5*math.log(abs((j1 + j2).m2()))
        self.lnKt    = math.log(j2.pt()*delta)
        self.lnz     = math.log(z)
        self.lnDelta = math.log(delta)
        self.lnKappa = math.log(z * delta)
        self.psi     = math.atan((j1.rap() - j2.rap())/(j1.phi() - j2.phi()))

    #----------------------------------------------------------------------
    @staticmethod
    def change_dimension(n):
        LundCoordinates.dimension=n
        LundCoordinates.low  = LundCoordinates.__low_full[:n]
        LundCoordinates.high = LundCoordinates.__high_full[:n]
        
    #----------------------------------------------------------------------
    def state(self):
        # WARNING: For consistency with other parts of the code,
        #          lnz and lnDelta need to be the first two components
        return np.array([self.lnz, self.lnDelta, self.psi,
                         self.lnm, self.lnKt][:LundCoordinates.dimension])


#======================================================================
class JetTree:
    """JetTree keeps track of the tree structure of a jet declustering."""

    #----------------------------------------------------------------------
    def __init__(self, pseudojet, child=None):
        """Initialize a new node, and create its two parents if they exist."""
        self.harder = None
        self.softer = None
        self.delta2 = 0.0
        self.lundCoord = None
        # first define the current node
        self.node = np.array([pseudojet.px(),pseudojet.py(),pseudojet.pz(),pseudojet.E()])
        # if it has a direct child (i.e. one level further up in the
        # tree), give a link to the corresponding tree object here
        self.child  = child
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        if pseudojet and pseudojet.has_parents(j1,j2):
            # order the parents in pt
            if (j2.pt() > j1.pt()):
                j1,j2=j2,j1
            # then create two new tree nodes with j1 and j2
            self.harder = JetTree(j1, self)
            self.softer = JetTree(j2, self)
            self.delta2 = j1.squared_distance(j2)
            self.lundCoord = LundCoordinates(j1, j2)

    #-------------------------------------------------------------------------------
    def remove_soft(self):
        """Remove the softer branch of the JetTree node."""
        # start by removing softer parent momentum from the rest of the tree
        child = self.child
        while(child):
            child.node-=self.softer.node
            child=child.child 
        del self.softer
        # then move the harder branch to the current node,
        # effectively deleting the soft branch
        newTree = self.harder
        self.node   = newTree.node
        self.softer = newTree.softer 
        self.harder = newTree.harder
        self.delta2 = newTree.delta2
        self.lundCoord = newTree.lundCoord
        # finally set the child pointer in the two parents to
        # the current node
        if self.harder:
            self.harder.child = self
        if self.softer:
            self.softer.child = self
        # NB: self.child doesn't change, we are just moving up the part
        #     of the tree below it

    #----------------------------------------------------------------------
    def state(self):
        """Return state of lund coordinates."""
        if not self.lundCoord:
            return np.zeros(LundCoordinates.dimension)
        return self.lundCoord.state()
    
    #----------------------------------------------------------------------
    def jet(self, pseudojet=False):
        """Return the kinematics of the JetTree."""
        #TODO: implement pseudojet option which returns a pseudojet
        #      with the reclustered constituents (after grooming)
        if not pseudojet:
            return self.node
        else:
            raise ValueError("JetTree: jet() with pseudojet return value not implemented.")

    #----------------------------------------------------------------------
    def __lt__(self, other_tree):
        """Comparison operator needed for priority queue."""
        return self.delta2 > other_tree.delta2

    #----------------------------------------------------------------------
    def __del__(self):
        """Delete the node."""
        if self.softer:
            del self.softer
        if self.harder:
            del self.harder
        del self.node
        del self

#======================================================================
class LundImage:
    """Class to create Lund images from a jet tree."""

    #----------------------------------------------------------------------
    def __init__(self, xval = [0.0, 7.0], yval = [-3.0, 7.0],
                 npxlx = 50, npxly = None):
        """Set up the LundImage instance."""
        # set up the pixel numbers
        self.npxlx = npxlx
        if not npxly:
            self.npxly = npxlx
        else:
            self.npxly = npxly
        # set up the bin edge and width
        self.xmin = xval[0]
        self.ymin = yval[0]
        self.x_pxl_wdth = (xval[1] - xval[0])/self.npxlx
        self.y_pxl_wdth = (yval[1] - yval[0])/self.npxly

    #----------------------------------------------------------------------
    def __call__(self, tree):
        """Process a jet tree and return an image of the primary Lund plane."""
        res = np.zeros((self.npxlx,self.npxly))

        self.fill(tree, res)
        return res

    #----------------------------------------------------------------------
    def fill(self, tree, res):
        """Fill the res array recursively with the tree declusterings of the hard branch."""
        if(tree and tree.lundCoord):
            x = -tree.lundCoord.lnDelta
            y =  tree.lundCoord.lnKt
            xind = math.ceil((x - self.xmin)/self.x_pxl_wdth - 1.0)
            yind = math.ceil((y - self.ymin)/self.y_pxl_wdth - 1.0)
            if (xind < self.npxlx and yind < self.npxly and min(xind,yind) >= 0):
                res[xind,yind] += 1
            self.fill(tree.harder, res)
            #self.fill(tree.softer, res)

