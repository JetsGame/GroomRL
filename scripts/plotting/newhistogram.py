from __future__ import division, print_function
import numpy as np
from math import sqrt, floor, log, exp
import sys

# single-histogram binning & binning for x-axis in 2d-histograms
_default_bins = None
# binning for y-axis in 2d histograms
_default_bins_y = None
# output format for 2D case
_compact2D = False

def set_compact2D(value = True):
    """
    Sets whether 2D output is made compact (6 digits, just midpoints of x & y bins)
    """
    global _compact2D
    _compact2D = value

def set_default_bins(bins_x, bins_y = None):
    """
    Sets the default binning; if only one argument is provided the y binning
    (for 2d histograms) is set to the x binning (== single-axis histogram binning)
    """
    global _default_bins, _default_bins_y
    _default_bins   = bins_x
    if (bins_y is None): _default_bins_y = bins_x
    else:                _default_bins_y = bins_y

#----------------------------------------------------------------------
class Bins(object):
    def string_from_ibin(self, ibin, format_string = "{} {} {}"):
        """returns a string describing this bin using the given format, which
        can take positional arguments (in the order xlo, xmid, xhi) or
        named arguments (xlo, xmid, xhi).
        """
        if (ibin < 0): return "underflow"
        if (ibin >= self.nbins): return "underflow"
        xlo  = self.xlo (ibin)
        xmid = self.xmid(ibin)
        xhi  = self.xhi (ibin)
        return format_string.format(xlo, xmid, xhi, xlo=xlo, xmid=xmid, xhi=xhi)

    def string_from_x(self, x, number_format = "{} {} {}"):
        """Similar to string_from_ibin, but you supply the x value instead"""
        return self.string_from_ibin(self.ibin(x), number_format)

    def xedges_with_outflow(self):
        "Returns a numpy array with the bin edges, including outflow markers"
        edges = np.empty(self.nbins + 3)
        edges[1:-1] = self.xedges()
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges
    
#----------------------------------------------------------------------
class LinearBins(Bins):
    def __init__(self, lo, hi, dbin):
        # deal with case of explicit limits
        self.nbins = int(abs((hi-lo)/dbin) + 0.5)
        self.dbin = (hi-lo)/self.nbins
        self.lo = lo
        self.hi = hi
    
    def ibin(self,xvalue):
        # identify the bin in the case where we have uniform spacing
        return int(floor((xvalue - self.lo)/self.dbin))
    
    def xlo(self, _ibin):
        return self.lo + (_ibin)*self.dbin

    def xmid(self, _ibin):
        return self.xvalue(_ibin)

    def xhi(self, _ibin):
        return self.lo + (_ibin+1.0)*self.dbin
    
    def xvalue(self, _ibin):
        return self.lo + (_ibin+0.5)*self.dbin
    
    def xvalues(self):
        return self.lo + (np.arange(0, self.nbins)+0.5)*self.dbin

    def xedges(self):
        "Returns a numpy array with the bin edges"
        return np.array([self.lo + (ibin)*self.dbin for ibin in range(0,self.nbins+1)])

    def __str__(self):
        return "Linear bins from {} to {}, each of size {}".format(self.lo, self.hi, self.dbin)

#----------------------------------------------------------------------
class LogBins(Bins):
    def __init__(self, lo, hi, dbin):
        """
        Create a logarithmic binning between lo and hi, where dbin is the 
        bin size (in the natural logarithm of the variable)
        """
        # deal with case of explicit limits
        self.nbins = int(abs(log(hi/lo)/dbin) + 0.5)
        self.dbin = log(hi/lo)/self.nbins
        self.lo = lo
        self.hi = hi
    
    def ibin(self,xvalue):
        # identify the bin in the case where we have uniform spacing
        return int(floor(log(xvalue / self.lo)/self.dbin))
    
    def xlo(self, _ibin):
        return self.lo *exp(_ibin*self.dbin)

    def xmid(self, _ibin):
        return self.xvalue(_ibin)

    def xhi(self, _ibin):
        return self.lo * exp((_ibin+1.0)*self.dbin)
    
    def xvalue(self, _ibin):
        return self.lo * exp((_ibin+0.5)*self.dbin)
    
    def xvalues(self):
        return self.lo * np.exp((np.arange(0, self.nbins)+0.5)*self.dbin)

    def __str__(self):
        return "Logarithmic bins from {} to {}, each of logarithmic size {}".format(self.lo, self.hi, self.dbin)


#----------------------------------------------------------------------
class CustomBins(Bins):
    def __init__(self, bin_edges):
        self._bin_edges = np.array(bin_edges)
        self.nbins = len(self._bin_edges) - 1
        self.lo = self._bin_edges[ 0]
        self.hi = self._bin_edges[-1]
        
    def ibin(self,xvalue):
        # identify the bin in the case where we have explicit bin edges
        #
        # NB: programming this by hand gives a result that runs 10x
        # faster than using numpy's searchsorted.
        if (xvalue < self.lo): return -1
        if (xvalue > self.hi): return self.nbins
        # bisect to find the bin
        ilo = 0
        ihi = self.nbins
        while (ihi - ilo != 1):
            imid = (ilo+ihi)//2
            if (xvalue > self._bin_edges[imid]): ilo = imid
            else                               : ihi = imid
        return ilo
        # --- this version is slow... ---
        #u = np.searchsorted(self._bin_edges, [xvalue])
        #return u[0]-1
        
    def xlo(self, _ibin):
        return self._bin_edges[_ibin]

    def xmid(self, _ibin):
        return 0.5 * (self._bin_edges[_ibin] + self._bin_edges[_ibin+1])

    def xhi(self, _ibin):
        return self._bin_edges[_ibin+1]
    
    def xvalue(self, _ibin):
        return self.xmid(_ibin)
    
    def xvalues(self):
        return 0.5 * (self._bin_edges[:-1] + self._bin_edges[1:])
        
    def xedges(self):
        "Returns a numpy array with the bin edges"
        return np.array(self._bin_edges)

    def __str__(self):
        return "CustomBins with edges at {}".format(self._bin_edges)


#----------------------------------------------------------------------
class HistogramBase(object):
    def __init__(self,bins):
        self._bins_locked = False
        self.set_bins(bins, False)

    def set_bins(self, bins=None, lock = True):
        """
        Sets the bins and resets all data to zero; if the lock argument is True
        then subsequent calls to this function will not change the bins or reset
        the contents
        """
        if (self._bins_locked): return self
        self._bins_locked = lock

        if (bins is None): self.bins = _default_bins
        else             : self.bins = bins

        # one could end up in a situation (e.g. with a 
        # hists["someName"].set_bins(...,...).add(...) call)
        # where the bins are not defined at this stage
        # (e.g. if default bins are empty)
        #
        # In that case, just return an incomplete object,
        # knowing there's a chance it will be set up properly 
        # later...
        if (self.bins is None): return self

        self.xvalues = self.bins.xvalues
        self.xvalue = self.bins.xvalue
        self.xhi = self.bins.xhi
        self.xlo = self.bins.xlo
        self.xmid = self.bins.xmid
        self.ibin = self.bins.ibin

        # this will need to be implemented in the main class
        # (not the base)
        self._init_contents()
        return self    
    
#----------------------------------------------------------------------
class Histogram(HistogramBase):
    '''Object to contains a histogram
    '''
    def __init__(self, bins = None, name=None):
        '''Create a histogram with the binning as specified by bins (or the current default)'''
        super(Histogram,self).__init__(bins)
        self.name  = name
        
    def _init_contents(self):
        self.underflow = 0.0
        self.overflow  = 0.0
        self._contents = np.zeros(self.bins.nbins)
        self._nentries = 0.0
        self._sumwgt   = 0.0
        self._sumxwgt  = 0.0
        self._sumx2wgt = 0.0

    def add(self, xvalue, weight = 1):
        """
        Add an entry to the histogram. 
        """
        _ibin = self.bins.ibin(xvalue)
        self._add_ibin(_ibin, weight)
        self._sumxwgt  += xvalue * weight
        self._sumx2wgt += xvalue**2 * weight

    def add_series(self, series, weights = None, weight = 1.0):
        """
        Takes data (and optionally weights) in the form of an np array
        and add it to the histogram. This is (should be?) much faster than adding
        entries individually, because it makes use of the numpy's
        histogram routine.

        If a weights array is supplied, then weight must be 1
        """
        self._nentries += len(series)
        if (weights is None):
            count, division = np.histogram(series, bins = self.bins.xedges_with_outflow())
            self._contents += weight * count[1:-1]
            self.underflow += weight * count[0]
            self.overflow  += weight * count[-1]
            self._sumwgt   += weight * len(series)
            self._sumxwgt  += sum(series) * weight
            self._sumx2wgt += sum(series**2) * weight
        else:
            if (weight != 1.0): raise ValueError("weight was {} but should be 1.0 "
                                                 "when weights argument is supplied".format(weight))
            count, division = np.histogram(series, bins = self.bins.xedges_with_outflow(), weights=weights)
            self._contents += count[1:-1]
            self.underflow += count[0]
            self.overflow  += count[-1]
            self._sumwgt   += sum(weights)
            self._sumxwgt  += sum(series * weights)
            self._sumx2wgt += sum(series**2) * weight

    def _add_ibin(self, _ibin, weight):
        if   (_ibin < 0):                self.underflow += weight
        elif (_ibin >= self.bins.nbins): self.overflow += weight
        else:                            self._contents[_ibin] += weight
        self._nentries += 1
        self._sumwgt   += weight

    def average(self):
        if (self._sumwgt != 0.0): return self._sumxwgt/self._sumwgt
        else: return 0.0

    def yvalues(self):
        return self._contents
    
    def error(self):
        return self.stddev()/sqrt(max(1,self._nentries-1))
    
    def stddev(self):
        if (self._sumwgt != 0.0): 
            return sqrt(self._sumx2wgt/self._sumwgt - self.average()**2)
        else:
            return 0.0
        
    def __getitem__(self,i):
        return self._contents[i]
    
    def __str__(self, rescale=1.0):
        output = ""
        if (self.name): output += "# histogram:{}\n".format(self.name)
        output += "# nentries = {}, avg = {}+-{}, stddev = {}, underflow = {}, overflow = {}\n".format(
            self._nentries, self.average(), self.error(), self.stddev(), self.underflow, self.overflow)
        for i in range(len(self._contents)):
            output += "{} {} {} {}\n".format(self.bins.xlo(i),
                                             self.bins.xmid(i),
                                             self.bins.xhi(i),
                                             self[i]*rescale)
        output +="\n"
        return output


    
#----------------------------------------------------------------------
class ProfileHistogram(HistogramBase):
    def __init__(self, bins = None, name=None):
        '''Create a profile histogram with bins going from lo to hi with bin size dbin'''
        super(ProfileHistogram,self).__init__(bins)
        self.name             = name

    def _init_contents(self):
        self.weights          = Histogram(self.bins, self.name)
        self.weights_times_y  = Histogram(self.bins, self.name)
        self.weights_times_y2 = Histogram(self.bins, self.name)
        self.n_entries        = Histogram(self.bins, self.name)
        self._total_n_entries  = 0.0
        
    def add(self, xvalue, yvalue, weight = 1):
        """
        Add an entry to the profile histogram. 
        """
        _ibin = self.bins.ibin(xvalue)
        self._add_ibin(_ibin, yvalue, weight)

    def _add_ibin(self, ibin, yvalue, weight = 1):
        self.weights        . _add_ibin (ibin, weight)
        self.weights_times_y. _add_ibin (ibin, weight * yvalue)
        self.weights_times_y2._add_ibin (ibin, weight * yvalue**2)
        self.n_entries.       _add_ibin (ibin, 1.0)
        self._total_n_entries += 1.0
        
        
    def __str__(self):
        # prepare some shortcuts
        weights          = self.weights.yvalues()
        weights_times_y  = self.weights_times_y.yvalues()
        weights_times_y2 = self.weights_times_y2.yvalues()
        n_entries        = self.n_entries.yvalues()
        # then process them
        average  = weights_times_y  / np.where(weights == 0, 1.0, weights)
        average2 = weights_times_y2 / np.where(weights == 0, 1.0, weights)
        stddev  = np.sqrt(np.maximum(0, average2 - average**2))
        err     = stddev / np.sqrt(np.maximum(n_entries - 1, 1))
        # then generate the output
        output = ""
        if (self.name): output += "# profileHistogram:{}\n".format(self.name)
        output += "# xlo xmid xhi average stddev err n_entries\n"
        for i in range(len(weights)):
            output += "{} {} {} {} {} {} {}\n".format(self.bins.xlo(i),
                                                      self.bins.xmid(i),
                                                      self.bins.xhi(i),
                                                      average[i], stddev[i],
                                                      err[i], n_entries[i])
        output +="\n"
        return output
    

#----------------------------------------------------------------------
class Histogram2D(object):
    '''Object to contains a histogram
    '''
    def __init__(self, bins_x = None, bins_y = None, name=None):
        '''Create a 2d histogram with the binning as specified by bins_x and bins_y (or the current default)'''

        self.name  = name
        self._bins_locked = False
        self.set_bins(bins_x, bins_y, False)
        
    def set_bins(self, bins_x = None, bins_y = None, lock = True):
        """
        Sets the bins and resets all data to zero; if the lock argument is True
        then subsequent calls to this function will not change the bins or reset
        the contents
        """

        if (self._bins_locked): return self
        self._bins_locked = lock

        if (bins_x is None): self.bins_x = _default_bins
        else:                self.bins_x = bins_x

        if (bins_y is None): self.bins_y = _default_bins_y
        else:                self.bins_y = bins_y

        # one could end up in a situation (e.g. with a 
        # hists2D["someName"].set_bins(...,...).add(...) call)
        # where the bins are not defined at this stage. 
        #
        # In that case, just return an incomplete object,
        # knowing there's a chance it will be set up properly 
        # later...
        if (self.bins_x is None or self.bins_y is None): return self

        self.outflow = 0.0
        self._contents = np.zeros((self.bins_x.nbins, self.bins_y.nbins))
        self._nentries = 0.0
        self._sumwgt   = 0.0

        # by returning self, the user can chain the calls, e.g.
        # hists2D["someName"].set_bins(...,...).add(...)
        return self

    def add(self, xvalue, yvalue, weight = 1):
        _ibin_x = self.bins_x.ibin(xvalue)
        _ibin_y = self.bins_y.ibin(yvalue)
        self._add_ibin(_ibin_x, _ibin_y, weight)

    def _add_ibin(self, _ibin_x, _ibin_y, weight):

        try:
            # watch out: numpy wraps negative indices around...
            # so raise an error that will take us to the overflow bin
            if (_ibin_x < 0 or _ibin_y < 0): raise IndexError
            self._contents[_ibin_x, _ibin_y] += weight
        except IndexError:
            self.outflow += weight
        self._nentries += 1
        self._sumwgt   += weight

    def average(self):
        if (self._sumwgt != 0.0): return self._sumxwgt/self._sumwgt
        else: return 0.0

    def zvalues(self):
        return self._contents
    
    def __getitem__(self, pos):
        i, j = pos
        return self._contents[i, j]
    
    def __str__(self, rescale=1.0):
        output = ""
        if (self.name): output += "# histogram2d:{}\n".format(self.name)
        output += "# nentries = {}, sumwgt = {}, outflow = {}\n".format(
            self._nentries, self._sumwgt * rescale, self.outflow * rescale)

        if (_compact2D):
            for ix in range(self._contents.shape[0]):
                for iy in range(self._contents.shape[1]):
                    output += "{:.6g} {:.6g} {:.6g}\n".format(
                                                     self.bins_x.xmid(ix),
                                                     self.bins_y.xmid(iy),
                                                     self._contents[ix, iy]*rescale)
        else:
            for ix in range(self._contents.shape[0]):
                for iy in range(self._contents.shape[1]):
                    output += "{} {} {} {} {} {} {}\n".format(self.bins_x.xlo(ix),
                                                     self.bins_x.xmid(ix),
                                                     self.bins_x.xhi(ix),
                                                     self.bins_y.xlo(iy),
                                                     self.bins_y.xmid(iy),
                                                     self.bins_y.xhi(iy),
                                                     self._contents[ix, iy]*rescale)
                output +="\n"
        output +="\n"
        return output


#----------------------------------------------------------------------    
class HistogramCollection(object):
    """Contains a collection of histograms, accessed via a dictionary. If
    a histogram is absent, then it's created, using the current
    defaults (set_defaults) and its title is the key name.

    """
    def __init__(self, histogram_type = Histogram, bins = None):
        self._histogram_type = histogram_type
        self._default_bins = bins
        
    def __getitem__(self,item):
        if (item in self.__dict__): return self.__dict__[item]
        else:
            h = self._histogram_type(bins=self._default_bins, name=item)
            self.__dict__[item] = h
            return h

    def set_default_bins(self, bins):
        self._default_bins = bins
        
    def keys(self):
        return self.__dict__.keys()

    def __str__(self):
        """
        Returns all histograms from the collection, without any normalisation.
        
        They are in alphabetical order of the keys.
        """
        output = ""
        sorted_keys = sorted(self.keys())
        for k in sorted_keys:
            if (k == "_histogram_type" or k == "_default_bins"): continue
            output += str(self[k]) + "\n"
        return output
    
#----------------------------------------------------------------------    
class Histogram2DCollection(object):
    """Contains a collection of histograms, accessed via a dictionary. If
    a histogram is absent, then it's created, using the current
    defaults (set_defaults) and its title is the key name.

    """
    def __init__(self, histogram_type = Histogram2D, bins_x = None, bins_y = None):
        self._histogram_type = histogram_type
        self.set_default_bins(bins_x, bins_y)
        
    def __getitem__(self,item):
        if (item in self.__dict__): return self.__dict__[item]
        else:
            h = self._histogram_type(bins_x=self._default_bins_x, bins_y=self._default_bins_y, name=item)
            self.__dict__[item] = h
            return h

    def set_default_bins(self, bins_x = None, bins_y = None):
        self._default_bins_x = bins_x
        self._default_bins_y = bins_y
        
    def keys(self):
        return self.__dict__.keys()

    def __str__(self):
        """
        Returns all histograms from the collection, without any normalisation.
        
        They are in alphabetical order of the keys.
        """
        output = ""
        sorted_keys = sorted(self.keys())
        for k in sorted_keys:
            if (k == "_histogram_type" or k == "_default_bins_x" or  k == "_default_bins_y"): continue
            output += str(self[k]) + "\n"
        return output
    

hists = HistogramCollection()
profile_hists = HistogramCollection(ProfileHistogram)
hists2D = Histogram2DCollection()

#----------------------------------------------------------------------
# predefined objects


# for testing
def _run_tests():
    # set_default_bins(LinearBins(-2.0, -1.0, 0.5))
    x_bins = LinearBins(0.0, 1.0, 0.5)
    y_bins = LinearBins(0.0, 4.0, 0.5)
    
    hists["test"].set_bins(LinearBins(5.0, 10.0, 1.0)).add(7.2)
    hists["test"].set_bins(LinearBins(5.0, 10.0, 1.0)).add(8.2)
    print (hists)

    profile_hists["test"].set_bins(LogBins(5.0, 10.0, 0.2)).add(7.2, 2.0)
    profile_hists["test"].set_bins(LogBins(5.0, 10.0, 0.2)).add(7.2, 4.0)
    print (profile_hists)

    #hists2D.set_default_bins(bins_y = LinearBins(0.0, 4.0, 0.5))
    #set_default_bins()
    #h = Histogram2D(bins_y = LinearBins(0.0, 4.0, 0.5))
    hists2D["test"].set_bins(x_bins, y_bins).add(0.7,0.3)
    hists2D["test"].set_bins(x_bins, y_bins).add(6.7,0.3)
    print(hists2D["test"][1,0])
    print(hists2D)

if __name__ == "__main__":
    _run_tests()
