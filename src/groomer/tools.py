import numpy as np
import math                

#----------------------------------------------------------------------
def mass(constituents):
    """Given a list of jets, determine the masses for each one."""
    masses = []
    for jet in constituents:
        # # this is for an output containing a list of jet constituents
        # j = np.zeros(4)
        # for particle in jet:
        #     print(particle)
        #     j += np.array(particle)
        # msq = j[3]*j[3] - j[0]*j[0] - j[1]*j[1] - j[2]*j[2]
        msq = jet[3]*jet[3] - jet[0]*jet[0] - jet[1]*jet[1] - jet[2]*jet[2]
        masses.append(math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq))
    return masses

#----------------------------------------------------------------------
def get_window_width(masses, lower_frac=20, upper_frac=80):
    """Returns"""
    lower = np.nanpercentile(masses, lower_frac)
    upper = np.nanpercentile(masses, upper_frac)
    median = np.median(masses[(masses > lower) & (masses < upper)])
    return lower, upper, median
