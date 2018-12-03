import json, gzip, sys
from abc import ABC, abstractmethod
import numpy as np
from math import log, ceil, floor, pi
import fastjet as fj

#======================================================================
class Reader(object):
    """
    Reader for files consisting of a sequence of json objects.
    Any pure string object is considered to be part of a header (even if it appears at the end!)
    """

    #----------------------------------------------------------------------
    def __init__(self, infile, nmax = -1):
        """Initialize the reader."""
        self.infile = infile
        self.nmax = nmax
        self.reset()

    #----------------------------------------------------------------------
    def reset(self):
        """
        Reset the reader to the start of the file, clear the header and event count.
        """
        self.stream = gzip.open(self.infile,'r')
        self.n = 0
        self.header = []
        
        
    #----------------------------------------------------------------------
    def __iter__(self):
        # needed for iteration to work 
        return self

    #----------------------------------------------------------------------
    def __next__(self):
        ev = self.next_event()
        if (ev is None): raise StopIteration
        else           : return ev

    #----------------------------------------------------------------------
    def next(self): return self.__next__()

    #----------------------------------------------------------------------
    def next_event(self):
        # we have hit the maximum number of events
        if (self.n == self.nmax):
            print ("# Exiting after having read nmax jet declusterings")
            return None
        
        try:
            line = self.stream.readline()
            j = json.loads(line.decode('utf-8'))
        except IOError:
            print("# got to end with IOError (maybe gzip structure broken?) around event", self.n, file=sys.stderr)
            return None
        except EOFError:
            print("# got to end with EOFError (maybe gzip structure broken?) around event", self.n, file=sys.stderr)
            return None
        except ValueError:
            print("# got to end with ValueError (empty json entry) around event", self.n, file=sys.stderr)
            return None

        # skip this
        if (type(j) is str):
            self.header.append(j)
            return self.next_event()
        self.n += 1
        return j

#======================================================================
class Image(ABC):
    """Image which transforms point-like information into pixelated 2D
    images which can be processed by convolutional neural networks."""
    def __init__(self, infile, nmax):
        self.reader = Reader(infile, nmax)

    #----------------------------------------------------------------------
    @abstractmethod
    def process(self, event):
        pass

    #----------------------------------------------------------------------
    def values(self):
        res = []
        while True:
            event = self.reader.next_event()
            if event!=None:
                res.append(self.process(event))
            else:
                break
        self.reader.reset()
        return res

#======================================================================
class Jets(Image):
    """Read input file with jet constituents and transform into python jets."""

    #----------------------------------------------------------------------
    def __init__(self, infile, nmax, pseudojets=True):
        Image.__init__(self, infile, nmax)
        self.jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.0)
        self.pseudojets = pseudojets

    #----------------------------------------------------------------------
    def process(self, event):
        constits = []
        if self.pseudojets:
            for p in event[1:]:
                constits.append(fj.PseudoJet(p['px'],p['py'],p['pz'],p['E']))
            jets = self.jet_def(constits)
            if (len(jets) > 0):
                return jets[0]
            return fj.PseudoJet()
        else:
            for p in event[1:]:
                constits.append([p['px'],p['py'],p['pz'],p['E']])
            return constits

#======================================================================
class JetImage(Image):
    """
    Take input file and transforms it into parsable jet image.
    """

    #----------------------------------------------------------------------
    def __init__(self, infile, nmax, npxl, R=1.0):
        Image.__init__(self, infile, nmax)
        self.npxl = npxl
        self.pxl_wdth = 2.0 * R / npxl
        self.R = R

    #----------------------------------------------------------------------
    def process(self, event):
        res = np.zeros((2,self.npxl,self.npxl))
        j = event[0]
        tot_pt = 0.0
        wgt_avg_rap = 0.0
        wgt_avg_phi = 0.0
        for p in event[1:]:
            phi = p['phi']
            if (p['phi'] - j['phi'] > 2 * self.R):
                phi -= 2.0*pi
            if (p['phi'] - j['phi'] < -2 * self.R):
                phi += 2.0*pi
            tot_pt += p['pt']
            wgt_avg_rap += p['rap']*p['pt']
            wgt_avg_phi += phi     *p['pt']
        wgt_avg_rap = wgt_avg_rap / tot_pt
        wgt_avg_phi = wgt_avg_phi / tot_pt

        rap_pt_cent_indx = int(ceil(wgt_avg_rap/self.pxl_wdth - 0.5) - floor(self.npxl/2.0))
        phi_pt_cent_indx = int(ceil(wgt_avg_phi/self.pxl_wdth - 0.5) - floor(self.npxl/2.0))

        L1norm = 0.0
        for p in event[1:]:
            phi = p['phi']
            if (p['phi'] - j['phi'] > 2 * self.R):
                phi -= 2.0*pi
            if (p['phi'] - j['phi'] < -2 * self.R):
                phi += 2.0*pi
            rap_indx = int(ceil(p['rap']/self.pxl_wdth - 0.5) - rap_pt_cent_indx)
            phi_indx = int(ceil(phi     /self.pxl_wdth - 0.5) - phi_pt_cent_indx)
            # print(rap_pt_cent_indx,rap_indx,phi_pt_cent_indx,phi_indx,
            #       ':',p['pt'])
            if (rap_indx < 0 or rap_indx >= self.npxl or
                phi_indx < 0 or phi_indx >= self.npxl):
                continue
            res[0,phi_indx,rap_indx] += p['pt']
            res[1,phi_indx,rap_indx] += 1
            L1norm += p['pt']

        if L1norm > 0.0:
            res[0] = res[0]/L1norm

        return res
        
#======================================================================
class LundImage(Image):
    """
    Class to take input file (or a reader) of jet declusterings in json
    format, one json entry per line, and transform it into lund images.

    - infile: a filename or a reader
    - nmax: the maximum number of jets to process
    - npxl: the number of bins (pixels) in each dimension
    - xval: the range of x (ln 1/Delta) values
    - yval: the range of y (ln kt) values

    Once you've created the class, call values() (inherited the abstract
    base class) and you will get a python list of images (formatted as
    2d numpy arrays).
    """
    #----------------------------------------------------------------------
    def __init__(self, infile, nmax, npxl, xval = [0.0, 7.0], yval = [-3.0, 7.0]):
        Image.__init__(self, infile, nmax)
        self.npxl = npxl
        self.xmin = xval[0]
        self.ymin = yval[0]
        self.x_pxl_wdth = (xval[1] - xval[0])/npxl
        self.y_pxl_wdth = (yval[1] - yval[0])/npxl

    #----------------------------------------------------------------------
    def process(self, event):
        """Process an event and return an image of the primary Lund plane."""
        res = np.zeros((self.npxl,self.npxl))

        for declust in event:
            x = log(1.0/declust['Delta'])
            y = log(declust['kt'])
            psi = declust['psi']
            xind = ceil((x - self.xmin)/self.x_pxl_wdth - 1.0)
            yind = ceil((y - self.ymin)/self.y_pxl_wdth - 1.0)
            # print((x - self.xmin)/self.x_pxl_wdth,xind,
            #       (y - self.ymin)/self.y_pxl_wdth,yind,':',
            #       declust['delta_R'],declust['pt2'])
            if (max(xind,yind) < self.npxl and min(xind, yind) >= 0):
                res[xind,yind] += 1
        return res
    
#======================================================================
class LundDense(Image):
    """
    Class to take input file (or a reader) of jet declusterings in json
    format, one json entry per line, and reduces them to the minimal
    information needed as an input to LSTM or dense network learning.

    - infile: a filename or a reader
    - nmax: the maximum number of jets to process
    - nlen: the size of the output array (for each jet), zero padded
            if the declustering sequence is shorter; if the declustering
            sequence is longer, entries beyond nlen are discarded.

    Calling values() returns a python list, each entry of which is a
    numpy array of dimension (nlen,2). values()[i,:]  =
    (log(1/Delta),log(kt)) for declustering i.
    """
    #----------------------------------------------------------------------
    def __init__(self,infile, nmax, nlen):
        Image.__init__(self, infile, nmax)
        self.nlen      = nlen
        
    #----------------------------------------------------------------------
    def process(self, event):
        """Process an event and return an array of declusterings."""
        res = np.zeros((self.nlen, 2))
        # go over the declusterings and fill the res array
        # with the Lund coordinates
        for i in range(self.nlen):
            if (i >= len(event)):
                break
            res[i,:] = self.fill_declust(event[i])
            
        return res

    #----------------------------------------------------------------------
    def fill_declust(self,declust):
        """ 
        Create an array of size two and fill it with the Lund coordinates
        (log(1/Delta),log(kt)).  
        """
        res = np.zeros(2)
        res[0] = log(1.0/declust['Delta'])
        res[1] = log(declust['kt'])
        return res
