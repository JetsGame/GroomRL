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
