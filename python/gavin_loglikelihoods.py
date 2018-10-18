#!/usr/bin/env python3
from __future__ import print_function, division
#import ujson as json
import json
import gzip
from operator import *
from math import *
import newhistogram as nh
import sys
import argparse
import numpy as np
import cProfile

nh.set_compact2D(True)

args = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest='outfile', default=sys.stdout, type=argparse.FileType('w'))
    parser.add_argument("--var", default="MZ()", help='creates class for the primary emission discriminator (e.g. "MZ()", "mMDTZ(0.025)", etc.')
    parser.add_argument("--discrim", default="Primary", help='Initial part of name of the overall discriminator class (Primary, Double, DoubleDiff)')
    parser.add_argument("--two-levels", action = 'store_true')
    parser.add_argument("--ptmin", default='2000')
    parser.add_argument("--nev-train", default=500000, type=int)
    parser.add_argument("--nev-ROC",   default=200000, type=int)
    parser.add_argument("--Lund-uses-var", action = 'store_true', help="when true, the Lund plane uses the same variable ordering than what has been specified by the --var option (instead of the MZ() ordering)")
    parser.add_argument("--no-leading-outflow", action = 'store_true', help="when true, discard information from the ouflow for the leading emission")
    parser.add_argument("--delphes", action = 'store_true', help="when true, use the json files including Delphes detector simulation")
    parser.add_argument("--mj-chg", action = 'store_true', help="when true, use the json files including minijet-based charged tracks rescaling")
    parser.add_argument("--mj-chg-phot", action = 'store_true', help="when true, use the json files including minijet-based charged tracks and photons rescaling")
    parser.add_argument("--lund-spacing", type=float, default=0.5, help="size of the binning in the lund plne (default 0.5x0.5)")
    
    #TODO:parser.add_argument("--flag-train", default='', type=str, help='extra flag to append to the json file used for training')
    #TODO:parser.add_argument("--flag-ROC",   default='', type=str, help='extra flag to append to the json file used for computing the ROC')
    global args
    args = parser.parse_args()
    
    sys.stdout = args.outfile

    print("# "+" ".join(sys.argv))
    print ("# args=",args)
    

    if (args.ptmin == '2000'):
        runstring = '1800-ptmin2000'
    elif (args.ptmin == '500'):
        runstring = '450-ptmin500'
    else:
        print("Unsupported value of ptmin = {}".format(args.ptmin), file=sys.stderr)
        raise ValueError

    if (args.delphes):
        runstring += "-delphes"
    if (args.mj_chg):
        runstring += "-mjchg0.12"
    if (args.mj_chg_phot):
        runstring += "-mjchgphot0.12"
        

    #bkgd   = Reader('gavin-results/lhc14-pythia8.230-Monash13-dijets1800-LCL-ptmin2000.dat.json.gz')
    #signal = Reader('gavin-results/lhc14-pythia8.230-Monash13-WW1800-LCL-ptmin2000.dat.json.gz')
    bkgd   = Reader('gavin-json/lhc14-pythia8.230-Monash13-dijets{}.dat.json.gz'.format(runstring))
    signal = Reader('gavin-json/lhc14-pythia8.230-Monash13-WW{}.dat.json.gz'.format(runstring))
    #check_deltaR(bkgd)
    #exit(0)
    variable_class = eval(args.var)
    # these are some of the options
    #variable_class = MZ()
    #variable_class = mMDTZ(0.025)
    #variable_class = mMDTOnly(0.10)

    print ("# background:",     bkgd  .infile)
    print ("# signal:",         signal.infile)
    print ("# variable_class:", variable_class)

    component_class = eval(args.discrim+"DiscriminatorComponent")
    print ("# component_class = ", component_class.__name__)
    if (args.two_levels):
        input_discrim = PrimaryDiscriminator(variable_class, signal, bkgd, component_class)
        # # rewind and take the two inputs
        signal.reset()
        bkgd.reset()
        discrim = PrimaryDiscriminator(TwoLevels(input_discrim), signal, bkgd, PrimaryDiscriminatorComponent)
    else:
        discrim = PrimaryDiscriminator(variable_class, signal, bkgd, component_class)

    print (discrim)

    # now test the discriminant and output its characteristics
    nev = args.nev_ROC
    print ("Now constructing ROC curves with {} events".format(nev), file=sys.stderr)
    test_s = DiscriminantTester("signal", discrim, signal, nev)
    test_b = DiscriminantTester("bkgd"  , discrim, bkgd  , nev)
    print(test_b)
    print(test_s)
    
    print ("# ROC")
    ROC = test_s.ROC_v_bkgd(test_b)
    for p in ROC: print(p[0], p[1])

#----------------------------------------------------------------------
class PrimaryDiscriminator(object):
    def __init__(self, variable_class, signal_reader, bkgd_reader, component_class):
        self.variable_class = variable_class
        self.signal_reader = signal_reader
        self.bkgd_reader   = bkgd_reader

        nev_build = args.nev_train # 500000
        #nev_build = 500
        self.signal = component_class('2dsignal', nev_build, variable_class, signal_reader)
        self.bkgd   = component_class('2dbkgd'  , nev_build, variable_class, bkgd_reader)
        
    def discriminant(self, event):
        return self.signal.discriminant(event, self.bkgd)

    def discriminant1(self, event):
        return self.signal.discriminant1(event, self.bkgd)

    def discriminant2(self, event):
        return self.signal.discriminant2(event, self.bkgd)

    def __str__(self):
        return "# Primary discriminator based on background and signal components:\n"\
               + self.bkgd.__str__() + "\n"  + self.signal.__str__()
    
#----------------------------------------------------------------------        
class PrimaryDiscriminatorComponent(object):
    """A class to hold one component (either signal or background)
    of a PrimaryDiscriminator.

    It can be overloaded, in which case, several routines need to
    be worked on:

    - setup_structures
    - add_event
    - finalise
    - discriminant
    """
    def __init__(self, name, nev_build, variable_class, reader):
        self.nev_build      = nev_build
        self.variable_class = variable_class
        self.reader         = reader
        self.name           = name

        self._setup_structures()
        iev = 0
        while (iev < self.nev_build):
            iev += 1
            event = reader.__next__()
            if (len(event) == 0): continue
            self._add_event(event)

        self._finalise()
        
    def _setup_structures(self):
        self.hist = nh.Histogram2D(*(self.variable_class.bins()), name=self.name)
        
    def _add_event(self, event):
        # bin its coordinates
        coords = self.variable_class.primary_coordinates(event)
        self.hist.add(*coords)
        
    def _finalise(self):
        self.hist._contents += 0.5
        self.hist._contents /= self.hist._nentries
        self.hist.outflow   += 0.5
        self.hist.outflow   /= self.hist._nentries
        
    def discriminant(self, event, bkgd_component):
        """Returns the discriminant for the event from the reader, using itself
        as the signal component and bkgd_component as the background one
        """
        coords = self.variable_class.primary_coordinates(event)
        # watch out for NaNs... (I still don't understand why we get them: associated
        # with zero-length declustering, which should never happen for a 2 TeV jet...)
        if (coords[0] != coords[0]): return 1e6

        icoords = (self.hist.bins_x.ibin(coords[0]),
                   bkgd_component.hist.bins_y.ibin(coords[1]))

        try:
            if (icoords[0] < 0 or icoords[1] < 0): raise IndexError
            discrim = log( self.hist._contents[icoords] / bkgd_component.hist._contents[icoords])
        except IndexError:
            # we should be in the outflow bin...
            if (args.no_leading_outflow):
                discrim = -1000.0
            else:
                discrim = log( self.hist.outflow / bkgd_component.hist.outflow)
        return discrim

    def __str__(self):
        return self.hist.__str__()
    

#----------------------------------------------------------------------        
class DoubleDiscriminatorComponent(PrimaryDiscriminatorComponent):
    """A class to hold one component (either signal or background)
    of a DoubleDiscriminator"""
    def __init__(self, name, nev_build, variable_class, reader):
        # initialise self.super, which will be a PrimaryDiscriminatorComponent
        self.super = super(DoubleDiscriminatorComponent, self)
        self.super.__init__(name, nev_build, variable_class, reader)
        # for the signal case, this will hold a pointer to the backgroun
        # component 
        self._bkgd_component = None

    def _setup_structures(self):
        self.super._setup_structures()
        # now work on the variables that will go into the per emission log-likelihood
        if args.Lund_uses_var:
            self.lund_primary_selector = self.variable_class
        else:
            self.lund_primary_selector = MZ()
        self.lund_hist = nh.Histogram2D(nh.LinearBins( 0.0, 8.0, args.lund_spacing),  #<-- log(1/delta_R)
                                        nh.LinearBins(-2.0, 8.0, args.lund_spacing),  #<-- log(kt)
#                                        nh.LinearBins(-1.0, 8.0, 0.5),  #<-- log(kt)
                                        name="Lund-"+self.name)

    def _add_event(self, event):
        # let the super-class (primary discriminator) do its job
        self.super._add_event(event)
        
        # then fill the Lund plane:
        # first prepare the mass ordering class
        self.lund_primary_selector.primary_coordinates(event)
        # then access all branchings except the main one
        for d in self.lund_primary_selector.declusterings[1:]:
            self.lund_hist.add(-log(d["delta_R"]), log(d["kt"]))

    def _finalise(self):
        self.super._finalise()
        # there's nothing extra for us to do here...
        pass
        
    def discriminant1(self, event, bkgd_component):
        """returns the discriminant of the superclass (primary discriminator)
        """
        return self.super.discriminant(event, bkgd_component)
    
    def discriminant2(self, event, bkgd_component):
        # Cache info about signal v. bkgd and make sure the cached info remains valid
        # (we may want to restructure this later...)
        if   (self._bkgd_component is None): self._register_bkgd(bkgd_component)
        elif (self._bkgd_component is not bkgd_component): raise ValueError
        
        discrim = 0
        # now try to add in info from other emissions
        # prepare the lund_primary_selector class
        self.lund_primary_selector.primary_coordinates(event)
        # and then extract its declustering information
        for d in self.lund_primary_selector.declusterings[1:]:
            xval = -log(d["delta_R"])
            yval =  log(d["kt"])
            ibin_x = self.lund_hist.bins_x.ibin(xval)
            ibin_y = self.lund_hist.bins_y.ibin(yval)
            try:
                if (ibin_x < 0 or ibin_y < 0): raise IndexError
                discrim += self._lund_LL_table[ibin_x, ibin_y]
            except IndexError:
                discrim += self._lund_LL_outflow
            #print(ibin_x, ibin_y, discrim, file=sys.stderr)
        #print ("", file=sys.stderr)
        return discrim

    def _lund_LL_function(self, count_s, n_s, count_b, n_b):
        """
        Function that calculates the log-likelihood entry needed for
        the Lund plane LL check. 
        
        - count_s is the total number of signal emissions registered in a 
          given Lund-plane bin
        - n_s is the total number of signal _events_ (not emissions) used
          to generate the Lund place
        - similarly for count_b, n_b
        """
        rate_s = (count_s + 0.5)/n_s
        rate_b = (count_b + 0.5)/n_b
        return log(rate_s / rate_b)

    def _register_bkgd(self, bkgd_component):
        """
        it's easiest to finalise some parts of the signal / bkgd discrimination calc^n once
        you have the signal and background together. This function calculates and caches the 
        relevant info. 
        """
        self._bkgd_component = bkgd_component
        s = self
        b = bkgd_component
        print(s.hist._nentries, b.hist._nentries, file=sys.stderr)
        self._v_lund_LL_function = np.vectorize(self._lund_LL_function)
        self._lund_LL_table = self._v_lund_LL_function(s.lund_hist._contents, s.hist._nentries,
                                                       b.lund_hist._contents, b.hist._nentries)
        self._lund_LL_outflow = self._v_lund_LL_function(s.lund_hist.outflow, s.hist._nentries,
                                                         b.lund_hist.outflow, b.hist._nentries)
    
    def discriminant(self, event, bkgd_component):
        #return (self.discriminant1(event, bkgd_component))
        return (self.discriminant1(event, bkgd_component)
               + 1.0*self.discriminant2(event, bkgd_component))

    def __str__(self):
        result  = self.super.__str__() + "\n"
        result += self.lund_hist.__str__()
        return result
    

#----------------------------------------------------------------------        
class DoubleDiffDiscriminatorComponent(PrimaryDiscriminatorComponent):
    """A class to hold one component (either signal or background)
    of a DoubleDiscriminator"""
    def __init__(self, name, nev_build, variable_class, reader):
        self.super = super(DoubleDiffDiscriminatorComponent, self)
        self.super.__init__(name, nev_build, variable_class, reader)
        # for the signal case, this will hold a pointer to the backgroun
        # component 
        self._bkgd_component = None

    def _setup_structures(self):
        self.super._setup_structures()
        # now work on the variables that will go into the per emission log-likelihood 
        if args.Lund_uses_var:
            self.lund_primary_selector = self.variable_class
        else:
            self.lund_primary_selector = MZ()
        #self.lund_primary_selector = MZ()
        # set up some rapidity bins
        #self._rapbins = nh.CustomBins([-1e10, 1e10])
        self._rapbins = nh.CustomBins([-1e10, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 1e10])
        self._raphist = nh.Histogram(self._rapbins, name="raphist-{}".format(self.name))
        self._lund_hists = [nh.Histogram2D(nh.LinearBins( 0.0, 8.0, args.lund_spacing),  #<-- log(1/delta_R)
                                           nh.LinearBins(-2.0, 8.0, args.lund_spacing),  #<-- log(kt)
                                           name="Lund-{:02d}-".format(i)+self.name)
                            for i in range(self._rapbins.nbins)]
         
    def _add_event(self, event):
        # let the super-class (primary discriminator) do its job
        self.super._add_event(event)
        
        # then fill the Lund plane:
        # first prepare the lund_primary_selector class and decide which rapidity bin we are in
        self.lund_primary_selector.primary_coordinates(event)
        rap = -log(self.lund_primary_selector.declusterings[0]["delta_R"])            
        irap = self._rapbins.ibin(rap)
        # with our bins, there should never be anything in the outflow regions
        if (irap < 0): raise IndexError 
        self._raphist.add(rap)
        # then access and store all branchings except the main one. The
        # LundPlane 2d-histogram they get stored in is specific to the delta R
        for d in self.lund_primary_selector.declusterings[1:]:
            self._lund_hists[irap].add(-log(d["delta_R"]), log(d["kt"]))

    def _finalise(self):
        self.super._finalise()
        # there's nothing extra for us to do here...
        pass
        
    def discriminant1(self, event, bkgd_component):
        """returns the discriminant of the superclass (primary discriminator)
        """
        return self.super.discriminant(event, bkgd_component)
    
    def discriminant2(self, event, bkgd_component):
        # Cache info about signal v. bkgd and make sure the cached info remains valid
        # (we may want to restructure this later...)
        if   (self._bkgd_component is None): self._register_bkgd(bkgd_component)
        elif (self._bkgd_component is not bkgd_component): raise ValueError
        
        # now try to add in info from other emissions
        # prepare the lund_primary_selector class and decide which rapidity bin we are in
        self.lund_primary_selector.primary_coordinates(event)
        if not self.lund_primary_selector.declusterings:
            return 0.0
        rap = -log(self.lund_primary_selector.declusterings[0]["delta_R"])            
        # coords = self.XX.coordinates(event)
        # rap = coords[0]
        irap = self._rapbins.ibin(rap)
        
        # with our bins, there should never be anything in the outflow regions
        if (irap < 0): raise IndexError

        # first include the offset for the log-likelihood estimator, associated with
        # the ln(exp(-mu_s}) / exp(-mu_b)) = mu_b - mu_s term in the logarithm of the
        # ratios of Poisson distributions
        discrim = self._rapbin_LL_offset[irap]  ### GPS TMP
        
        # and then extract its declustering information
        for d in self.lund_primary_selector.declusterings[1:]:
            xval = -log(d["delta_R"])
            yval =  log(d["kt"])
            ibin_x = self._lund_hists[irap].bins_x.ibin(xval)
            ibin_y = self._lund_hists[irap].bins_y.ibin(yval)
            try:
                if (ibin_x < 0 or ibin_y < 0): raise IndexError
                discrim += self._rapbin_LL_table[irap][ibin_x, ibin_y]
                #print(self._rapbin_LL_table[irap][ibin_x, ibin_y], end="", file=sys.stderr)
            except IndexError:
                #print (len(self._rapbin_LL_table), end=" ", file=sys.stderr)
                #print (self._rapbin_LL_table.shape, end=" ", file=sys.stderr)
                discrim += self._rapbin_LL_outflow[irap]
            #print(ibin_x, ibin_y, discrim, irap, file=sys.stderr)
        #print ("", file=sys.stderr)
        return discrim

    def _lund_LL_function(self, count_s, n_s, count_b, n_b):
        """
        Function that calculates the log-likelihood entry needed for
        the Lund plane LL check. 
        
        - count_s is the total number of signal emissions registered in a 
          given Lund-plane bin
        - n_s is the total number of signal _events_ (not emissions) used
          to generate the Lund place
        - similarly for count_b, n_b

        The function returns a pair (offset, emsn_contrib):

        - offset is the amount that this bin contributes to the overall LL offset
        - emsn_contrib is the amount that an emission in this bin contributes to the
          LL result
        """

        # in constructing the offset and emsn_contrib, we need to
        # gracefully handle cases of bins where there are no
        # events.
        #
        # The basic principle that we'll use is that the true count of
        # something that came out as zero could be anywhere between 0
        # and 0.5: we should assume the value that minimises
        # |emsn_contrib = ln(mu_s/mu_b)| so as minimise the risk of
        # injecting strong erraneous information.
        #
        # cf also GPS-CCN-37-16
        if (count_s == 0):
            if (count_b == 0):
                return (0.0, 0.0)
            else:
                mu_s = 0.5 / n_s
                mu_b = count_b / n_b
                # if mu_s > mu_b with the assumtion of count_s = 0.5
                # then a smaller count_s could lead to mu_s < mu_b,
                # i.e. we don't know the sign of emsn_contrib -> set it
                # to zero by assigning mu_s = mu_b
                if (mu_s > mu_b): mu_s = mu_b
        elif (count_b == 0):
            # just swap s <=> b wrt case of count_s = 0, count_b > 0
            mu_b = 0.5 / n_b
            mu_s = count_s / n_s
            if (mu_b > mu_s): mu_b = mu_s
        else:
            # everything is non-zero -- yippee!
            mu_s = count_s/n_s
            mu_b = count_b/n_b

        # next two lines are GPS testing as is 0* offset
        #mu_b = (0.5 + count_b)/n_b
        #mu_s = (0.5 + count_s)/n_s
        #offset       = (mu_b - mu_s)*0
        offset       = mu_b - mu_s
        emsn_contrib = log(mu_s/mu_b)

        return (offset, emsn_contrib)

    def _register_bkgd(self, bkgd_component):
        """
        it's easiest to finalise some parts of the signal / bkgd discrimination calc^n once
        you have the signal and background together. This function calculates and caches the 
        relevant info. 
        """
        self._bkgd_component = bkgd_component
        s = self
        b = bkgd_component
        #print(s.hist._nentries, b.hist._nentries, file=sys.stderr)
        v = np.vectorize(self._lund_LL_function)

        self._rapbin_LL_offset   = []
        self._rapbin_LL_table    = []
        self._rapbin_LL_outflow  = []
        
        for irap in range(self._rapbins.nbins):
            table_res   = v(s._lund_hists[irap]._contents, s._raphist._contents[irap],
                            b._lund_hists[irap]._contents, b._raphist._contents[irap])
            outflow_res = v(s._lund_hists[irap].outflow,   s._raphist._contents[irap],
                            b._lund_hists[irap].outflow,   b._raphist._contents[irap])
            self._rapbin_LL_offset .append(table_res[0].sum() + outflow_res[0])
            self._rapbin_LL_table  .append(table_res[1])
            self._rapbin_LL_outflow.append(outflow_res[1])

        #print ("rapbin_LL_table: ", v(s._lund_hists[0]._contents, s._raphist._contents[0],
        #                    b._lund_hists[0]._contents, b._raphist._contents[0]), file=sys.stderr)
        print("# {} offsets: {}\n".format(self.name, self._rapbin_LL_offset), file=sys.stderr)
            
    def discriminant(self, event, bkgd_component):
        #return (self.discriminant1(event, bkgd_component))
        return (self.discriminant1(event, bkgd_component)
               + 1.0*self.discriminant2(event, bkgd_component))

    def __str__(self):
        result  = self.super.__str__() + "\n"
        for irap,lh in enumerate(self._lund_hists):
            result += '# {}-rapbin{:02d}: nev={}\n'.format(self.name, irap, self._raphist[irap])
            result += lh.__str__(rescale=1.0/self._raphist[irap]) + "\n"
        result += self._raphist.__str__() + "\n"
        #try:
        #    result += "# {} offsets: {}\n".format(self.name, self._rapbin_LL_offset)
        
        # should we also print out the offsets, etc.
        return result
    

            

#----------------------------------------------------------------------
class Reader(object):
    """
    Reader for files consisting of a sequence of json objects.
    Any pure string object is considered to be part of a header (even if it appears at the end!)
    """
    def __init__(self, infile, nmax = -1):
        self.infile = infile
        self.nmax = nmax
        self.reset()
        
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
        
    def __next__(self):
        ev = self.next_event()
        if (ev is None): raise StopIteration
        else           : return ev

    def next(self): return self.__next__()
        
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

    
#----------------------------------------------------------------------
class TwoLevels(object):
    """
    Class that is intended to take input from two other discriminators and build
    a combined discriminator from them...
    """
    def __init__(self, input_discrim):
        self.input_discrim = input_discrim
        
    def bins(self):
        return (nh.LinearBins(-6.0, 10.0, 0.5), nh.LinearBins(-16.0, 10.0, 0.5))
    
    def coordinates(self, event):
        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0): return (float('nan'),float('nan'))

        return (self.input_discrim.discriminant1(event), self.input_discrim.discriminant2(event))

    def __str__(self):
        return "a 2d discriminant based on the two inputs from the 'PrimaryDiscriminator'"
    
#----------------------------------------------------------------------
class MZ(object):
    def __init__(self, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lnzBins=nh.LinearBins(-10.0, log(0.5), 0.2)):
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins
        #return (nh.LinearBins( -2.0, 11.0, 0.025), nh.LinearBins(-10.0, log(0.5), 0.2))
        #return (nh.LinearBins( -2.0, 11.0, 0.05), nh.LinearBins(-10.0, log(0.5), 0.2))
        #return (nh.LinearBins( -2.0, 11.0, 0.2), nh.LinearBins(-10.0, log(0.5), 0.2))

    # order the declusterings according to the metric
    #    |log(pt1 pt2 dR^2/mW^2) log(1/z)|
    # the 1st has the smallest metric
    #
    # The mass and z values for the 1st declustering are returned
    def primary_coordinates(self, event):
        self.declusterings = event
        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0):
            return (float('nan'),float('nan'))
        
        # this is fairly symmetric between large v. small masses
        sort_fn = lambda d: abs(log(d["pt1"]*d["pt2"]*d["delta_R"]**2 / 80.4**2) * log(1/d["z"]))
        # this one prefers small masses
        # sort_fn = lambda d: abs((d["pt1"]*d["pt2"]*d["delta_R"]**2 - 80.4**2) * log(1/d["z"]))

        # sort the declusterings to pull out one we think is potentially interesting
        self.declusterings = sorted(event, key = sort_fn)
        d0 = self.declusterings[0]

        return (log(d0["m"]), log(d0["z"]))

    def __str__(self):
        return "MZ variable pair involving search for primary branching with pt1*pt2*deltaR^2 closest to MW^2, returning ln(m),ln(z); bins are "\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])

#----------------------------------------------------------------------
class DotZ(object):
    def __init__(self, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lnzBins=nh.LinearBins(-10.0, log(0.5), 0.2)):
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins

    # order the declusterings according to the metric
    #    |log(pt1 pt2 dR^2/ptjet^2) log(1/z)|
    # the 1st has the smallest metric
    #
    # The mass and z values for the 1st declustering are returned
    def primary_coordinates(self, event):
        self.declusterings = event
        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0):
            return (float('nan'),float('nan'))

        pt2 = event[0]["pt"]**2
        
        # this is fairly symmetric between large v. small masses
        sort_fn = lambda d: -d["pt1"]*d["pt2"]*d["delta_R"]**2

        # sort the declusterings to pull out one we think is potentially interesting
        self.declusterings = sorted(event, key = sort_fn)
        d0 = self.declusterings[0]

        return (log(d0["m"]), log(d0["z"]))

    def __str__(self):
        return "MZ variable pair involving search for primary branching with largest pt1*pt2*deltaR^2 and large z, returning ln(m),ln(z); bins are "\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])

#----------------------------------------------------------------------
class KTZ(object):
    def __init__(self, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lnzBins=nh.LinearBins(-10.0, log(0.5), 0.2)):
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins

    # order the declusterings according to the metric
    #    |log(pt1 pt2 dR^2/mW^2) log(1/z)|
    # the 1st has the smallest metric
    #
    # The mass and z values for the 1st declustering are returned
    def primary_coordinates(self, event):
        self.declusterings = event
        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0):
            return (float('nan'),float('nan'))
        
        # this is fairly symmetric between large v. small masses
        sort_fn = lambda d: -d["kt"]
        # this one prefers small masses
        # sort_fn = lambda d: abs((d["pt1"]*d["pt2"]*d["delta_R"]**2 - 80.4**2) * log(1/d["z"]))

        # sort the declusterings to pull out one we think is potentially interesting
        self.declusterings = sorted(event, key = sort_fn)
        d0 = self.declusterings[0]

        return (log(d0["m"]), log(d0["z"]))

    def __str__(self):
        return "MZ variable pair involving search for primary branching with pt1*pt2*deltaR^2 closest to MW^2, returning ln(m),ln(z); bins are "\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])
    
#----------------------------------------------------------------------
class mMDTZ(object):
    def __init__(self, zcut, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lnzBins=nh.LinearBins(-10.0, log(0.5), 0.2)):
        self.zcut = zcut
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins
    
    # put in front of the declusterings the first one that passes mMDT
    #
    # The mass and z values for the 1st declustering are returned
    def primary_coordinates(self, event):
        self.declusterings = list(event)

        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0): return (float('nan'),float('nan'))

        # we'rte just going to make sure that the 1st emission that we
        # have in the final record is put in front (w a simple swapping)
        for index,d in enumerate(self.declusterings):
            if d["z"] > self.zcut:
                self.declusterings[index],self.declusterings[0] = self.declusterings[0],self.declusterings[index]
                d0 = self.declusterings[0]
                return (log(d0["m"]), log(d0["z"]))

        # otherwise return nonsense
        return (-100.0, -100.0)
            
    def __str__(self):
        return "mMDTZ variable pair involving search for mMDT/SD threhold with z > {}, then returning ln(m),ln(z) for that splitting".format(self.zcut)\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])

#----------------------------------------------------------------------
class SDZ(object):
    def __init__(self, zcut, beta, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lnzBins=nh.LinearBins(-10.0, log(0.5), 0.2)):
        self.zcut = zcut
        self.beta = beta
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins
    
    # put in front of the declusterings the first one that passes SoftDrop
    #
    # The mass and z values for the 1st declustering are returned
    def primary_coordinates(self, event): 
        self.declusterings = list(event)
        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0): return (float('nan'),float('nan'))

        for index,d in enumerate(self.declusterings):
            # this assumes R = 1 (we also impose that at least half of the mass is carried by the current splitting
            if d["z"] > self.zcut * pow(d["delta_R"],self.beta):
                self.declusterings[index],self.declusterings[0] = self.declusterings[0],self.declusterings[index]
                d0 = self.declusterings[0]
                return (log(d0["m"]), log(d0["z"]))

        # otherwise return nonsense
        return (-100.0, -100.0)
            
    def __str__(self):
        return "SD variable pair involving search for mMDT/SD threshold with z > {} delta_R^{}, then returning ln(m),ln(z) for that splitting".format(self.zcut, self.beta)\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])

            
    def __str__(self):
        return "SDZ variable pair involving search for the largest angle emission satisfying z > {} delta_R^{}, then returning ln(m),ln(z) for that splitting".format(self.zcut, self.beta)\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])

#----------------------------------------------------------------------
class mMDTD(object):
    def __init__(self, zcut, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lndeltaRBins=nh.LinearBins(0.0, 8.0, 0.1)):
        self.zcut = zcut
        self._bins = (lnmBins, lndeltaRBins)

    def bins(self):
        return self._bins
    
    # put in front of the declusterings the first one that passes mMDT
    #
    # The log(mass) and log(1/deltaR) values for the 1st declustering are returned
    def primary_coordinates(self, event):
        self.declusterings = list(event)

        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0): return (float('nan'),float('nan'))

        # we'rte just going to make sure that the 1st emission that we
        # have in the final record is put in front (w a simple swapping)
        for index,d in enumerate(self.declusterings):
            if d["z"] > self.zcut:
                self.declusterings[index],self.declusterings[0] = self.declusterings[0],self.declusterings[index]
                d0 = self.declusterings[0]
                return (log(d0["m"]), -log(d0["delta_R"]))

        # otherwise return nonsense
        return (-100.0, -100.0)
            
    def __str__(self):
        return "mMDTD variable pair involving search for mMDT/SD threhold with z > {}, then returning ln(m),ln(1/DletaR) for that splitting".format(self.zcut)\
               + "lnm[{}], lndeltaR[{}]".format(self._bins[0], self._bins[1])
    
#----------------------------------------------------------------------
class KTmMDTZ(object):
    def __init__(self, zcut, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lnzBins=nh.LinearBins(-10.0, log(0.5), 0.2)):
        self.zcut = zcut
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins
    
    # put in front of the declusterings the first one that passes mMDT
    #
    # The mass and z values for the 1st declustering are returned
    def primary_coordinates(self, event):
        self.declusterings = event

        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0): return (float('nan'),float('nan'))

        jetpt=event[0]["pt"]

        # sort the selected emissions in dot product
        # only include the emissions above  the cut (in practice, the others are just given  some +ve value)
        sort_fn = lambda d: -d["kt"] if (d["z"] > self.zcut) else jetpt-d["kt"] 
        self.declusterings = sorted(event, key = sort_fn)
        d0 = self.declusterings[0]

        return (log(d0["m"]), log(d0["z"]))

            
    def __str__(self):
        return "KTmMDTZ variable pair involving search for the largest kt emission satisfying z > {}, then returning ln(m),ln(z) for that splitting".format(self.zcut)\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])
    

#----------------------------------------------------------------------
class KTSDZ(object):
    def __init__(self, zcut, beta, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lnzBins=nh.LinearBins(-10.0, log(0.5), 0.2)):
        self.zcut = zcut
        self.beta = beta
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins
    
    # put in front of the declusterings the first one that passes mMDT
    #
    # The mass and z values for the 1st declustering are returned
    def primary_coordinates(self, event):
        self.declusterings = list(event)

        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0): return (float('nan'),float('nan'))

        jetpt=event[0]["pt"]
        
        # sort the selected emissions in dot product
        # only include the emissions above  the cut (in practice, the others are just given  some +ve value)
        sort_fn = lambda d: -d["kt"] if (d["z"] > self.zcut * pow(d["delta_R"],self.beta)) else jetpt-d["kt"] 
        self.declusterings = sorted(event, key = sort_fn)
        d0 = self.declusterings[0]

        return (log(d0["m"]), log(d0["z"]))

            
    def __str__(self):
        return "KTSDZ variable pair involving search for the largest kt emission satisfying z > {}, then returning ln(m),ln(z) for that splitting".format(self.zcut)\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])
    

#----------------------------------------------------------------------
class DotmMDTZ(object):
    def __init__(self, zcut, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lnzBins=nh.LinearBins(-10.0, log(0.5), 0.2)):
        self.zcut = zcut
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins
    
    # put in front of the declusterings the first one that passes mMDT
    #
    # The mass and z values for the 1st declustering are returned
    def primary_coordinates(self, event):
        self.declusterings = event

        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0): return (float('nan'),float('nan'))

        jetpt2=event[0]["pt"]**2

        # sort the selected emissions in dot product
        # only include the emissions above  the cut (in practice, the others are just given  some +ve value)
        sort_fn = lambda d: -d["pt1"]*d["pt2"]*d["delta_R"]**2 if (d["z"] > self.zcut and d["m"]<125.0 ) else jetpt2-d["pt1"]*d["pt2"]*d["delta_R"]**2
        self.declusterings = sorted(event, key = sort_fn)
        d0 = self.declusterings[0]

        return (log(d0["m"]), log(d0["z"]))

            
    def __str__(self):
        return "DotmMDTZ variable pair involving search for the largest dot-product emission satisfying z > {}, then returning ln(m),ln(z) for that splitting".format(self.zcut)\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])
    

#----------------------------------------------------------------------
class DotSDZ(object):
    def __init__(self, zcut, beta, lnmBins=nh.LinearBins( -2.0, 11.0, 0.025), lnzBins=nh.LinearBins(-10.0, log(0.5), 0.2)):
        self.zcut = zcut
        self.beta = beta
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins
    
    # put in front of the declusterings the first one that passes mMDT
    #
    # The mass and z values for the 1st declustering are returned
    def primary_coordinates(self, event):
        self.declusterings = list(event)

        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0): return (float('nan'),float('nan'))

        jetpt2=event[0]["pt"]**2

        # sort the selected emissions in dot product
        # only include the emissions above  the cut (in practice, the others are just given  some +ve value)
        sort_fn = lambda d: -d["pt1"]*d["pt2"]*d["delta_R"]**2 if (d["z"] > self.zcut * pow(d["delta_R"],self.beta)) else jetpt2-d["pt1"]*d["pt2"]*d["delta_R"]**2
        self.declusterings = sorted(event, key = sort_fn)
        d0 = self.declusterings[0]

        return (log(d0["m"]), log(d0["z"]))

            
    def __str__(self):
        return "KTSDZ variable pair involving search for the largest dot-product emission satisfying z > {}, then returning ln(m),ln(z) for that splitting".format(self.zcut)\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])
    

#----------------------------------------------------------------------
class mMDTOnly(object):
    def __init__(self, zcut):
        self.zcut = zcut
        
    def bins(self):
        return (nh.LinearBins( -2.0, 11.0, 0.05), nh.LinearBins(0.0, 1.0, 1.0))
    
    def coordinates(self, event):
        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0): return (float('nan'),float('nan'))

        for d in event:
            if d["z"] > self.zcut: return (log(d["m"]), 0.5)

        # otherwise return nonsense
        return (-100.0, -100.0)
            
    def __str__(self):
        return "variable pair involving search for mMDT/SD threhold with z > {}, then returning ln(m),dummy for that splitting".format(self.zcut)

#----------------------------------------------------------------------
class NoLeading(object):
    def __init__(self, lnmBins=nh.LinearBins( -1.0, 1.0, 2.0), lnzBins=nh.LinearBins(-1.0, 1.0, 2.0)):
        self._bins = (lnmBins, lnzBins)

    def bins(self):
        return self._bins
        #return (nh.LinearBins( -2.0, 11.0, 0.025), nh.LinearBins(-10.0, log(0.5), 0.2))
        #return (nh.LinearBins( -2.0, 11.0, 0.05), nh.LinearBins(-10.0, log(0.5), 0.2))
        #return (nh.LinearBins( -2.0, 11.0, 0.2), nh.LinearBins(-10.0, log(0.5), 0.2))

    # just get the coordinates
    # the 1st one is added at the end so it is included in the Lund plane contribution
    def primary_coordinates(self, event):
        self.declusterings = list(event)

        # I'm not sure why we get empty declustering sequences...
        if (len(event) == 0):
            return (float('nan'),float('nan'))

        self.declusterings.append(event[0])
        return (0.0, 0.0)

    def __str__(self):
        return "No leading emission selection, returning a fixed (outflow) number for ln(m),ln(z); bins are "\
               + "lnm[{}], lnz[{}]".format(self._bins[0], self._bins[1])
    
#----------------------------------------------------------------------
def check_deltaR(reader):
    "small routine for checking the distribution of Delta R"
    print("# reading from", reader.infile)
    vars = MZ()
    h = nh.Histogram(nh.LinearBins(-1.0, 12.0, 0.2), name='ln(1/deltaR')
    for ev in reader:
        vars.coordinates(ev)
        primary = vars.declusterings[0]
        ln_1_deltaR = -log(primary["delta_R"])
        if (ln_1_deltaR != ln_1_deltaR): continue
        h.add(ln_1_deltaR)
    print(h)


#----------------------------------------------------------------------        
class DiscriminantTester(object):
    """
    Class to help test a discriminant. You create one with a signal stream, 
    a second one with a background stream. And then use the ROC 
    facility to compare the results
    """
    def __init__(self, name, discrim, reader, nev):
        """Create a Discriminant tester based on a discriminant
        (discrim), using nev events from the reader for the test. For
        the purpose of histograms, etc., assign name to the output.
        
        To get the final output, one needs to run two discriminant
        testers, one for the sigal, one for the background.

        """
        self.array = []
        self.name  = name

        self.hist = nh.Histogram(nh.LinearBins(-20.0, 10.0, 0.1), name='DT-'+name)
        for i in range(nev):
            event = reader.next()
            d = discrim.discriminant(event)
            self.array.append(d)
            self.hist.add(d)

            #print ("AA"+name, discrim.discriminant1(event), discrim.discriminant2(event))
            
        # it will be useful when creating the ROC curve later to have
        # this array sorted
        self.array.sort()

    def __str__(self):
        return "# Discriminant tester content ({})\n{}".format(self.name, self.hist.__str__())
        
    def ROC_v_bkgd(self, bkgd):
        """
        produce the ROC curve, assuming self is the signal, while bkgd is a separately
        created background discriminant tester
        """

        # convenient alises
        array_s = self.array
        array_b = bkgd.array

        result = []
        i_s = len(array_s) - 1
        i_b = len(array_b) - 1
        while i_s >= 0:
            # find background position that corresponds to a
            # discriminant that is just below the current value of the
            # signal discriminant
            while (i_b >= 0 and array_b[i_b] > array_s[i_s]): i_b -= 1
            # exit the loop if the searh was unsuccessful
            if (array_b[i_b] > array_s[i_s]): break

            # store the result
            result.append([1-(1.0 + i_s)/len(array_s), 1-(1.0 + i_b)/len(array_b), array_s[i_s]])
            i_s -= int(len(array_s)/100)

        return result

def double_train():
    "Attempt to try two layers of training"
    pass
    
#----------------------------------------------------------------------        
if __name__ == "__main__":
    main()

    # if you want to profile this, one way is
    # python3 -m cProfile -o outfile -s cumtime ./test.py -o a --discrim Double
    #
    # or else run with
    #cProfile.run("main()")
    
