import heapq as hq
import fastjet as fj
import numpy as np
import math
                

#----------------------------------------------------------------------
def pq_to_list(pq):
    """Take an input priority queue (heapq) and return the ordered list, without the delta R value"""
    l=[]
    # iterate until the priority queue is empty
    while(pq):
        # pop the largest Delta R element on the queue, and append it to the list
        l.append(hq.heappop(pq)[1:])

    return l

#----------------------------------------------------------------------
def fill_pq(pq, j, children=[], idty=0):
    """Fill the priority queue with all declustered pseudojets."""
    # create two empty pseudojets which will hold the parents
    j1 = fj.PseudoJet()
    j2 = fj.PseudoJet()
    # if j has two parents, label them j1 and j2
    if (j.has_parents(j1,j2)):
        # order the parents in pt
        if (j2.pt() > j1.pt()):
            j1,j2=j2,j1

        # set up the ID tags
        #  current node ID: idty+1
        #  hard subjet ID: tag_j+1 (or -1 if it is an endpoint)
        #  soft subjet ID: tag_j+len(j1.constits) (or -1 if it is an endpoint)
        tag_j  = idty+1
        if len(j1.constituents())>1:
            tag_j1 = tag_j+1
        else:
            tag_j1 = -1
        if len(j2.constituents())>1:
            tag_j2 = tag_j+len(j1.constituents())
        else:
            tag_j2 = -1
        
        # push the current node on the queue with the following info:
        # [delta_R, jet, [list_of_children], ID_tag, [list_of_parents]]
        # NB: children are the precessors in the clustering tree
        hq.heappush(pq, [-j1.squared_distance(j2), j, children, tag_j, [tag_j1, tag_j2]])
        
        # then continue to fill the priority queue with the two parents
        fill_pq(pq, j1, children+[idty+1], idty+1)
        fill_pq(pq, j2, children+[idty+1], idty+len(j1.constituents()))

#----------------------------------------------------------------------
def declusterings(jet):
    """Take a FastJet PseudoJet and return a list of declusterings."""
    # create a priority queue of ordered declusterings
    declusts = []
    fill_pq(declusts, jet)
    # transform the pq to a list, removing the angle component
    ldecl = pq_to_list(declusts)
    res = []
    # loop over the declusterings and save everything in four-vector
    # instead of pseudojets
    for jet,children,tag,parents in ldecl:
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        jet.has_parents(j1,j2)
        if (j2.pt() > j1.pt()):
            j1,j2=j2,j1
        res.append([[jet.px(),jet.py(),jet.pz(),jet.E()],
                    children, tag, parents,
                    [j1.px(),j1.py(),j1.pz(),j1.E()],
                    [j2.px(),j2.py(),j2.pz(),j2.E()]])
    return res

#---------------------------------------------------------------------- 
def coords(jet):
    """Get transverse momentum, rapidity and azimuth of a jet"""
    ptsq = jet[0]*jet[0] + jet[1]*jet[1]
    phi   = math.atan2(jet[1],jet[0]);
    if phi < 0.0:
        phi += 2*math.pi
    if phi >= 2*math.pi:
        phi -= 2*math.pi
    if (jet[3] == abs(jet[2]) and ptsq == 0):
        MaxRapHere = 1e5 + abs(jet[2]);
        if jet[2] >= 0.0:
            rap = MaxRapHere
        else:
            rap = -MaxRapHere
    else:
        effective_m2 = max(0.0,jet[3]*jet[3] - jet[0]*jet[0] - jet[1]*jet[1] - jet[2]*jet[2])
        E_plus_pz    = jet[3] + abs(jet[2])
        rap = 0.5*math.log((ptsq + effective_m2)/(E_plus_pz*E_plus_pz))
        if jet[2] > 0:
            rap = -rap
    return math.sqrt(ptsq), rap, phi

#----------------------------------------------------------------------
def kinematics_node(declust):
    """Get kinematics of the current node."""
    jet,children,tag,parents,j1,j2 = declust
    # calculate coordinates
    pt1, rap1, phi1 = coords(j1)
    pt2, rap2, phi2 = coords(j2)
    dphi = abs(phi1 - phi2);
    if dphi > math.pi:
        dphi = 2*math.pi - dphi
    drap = rap1 - rap2;
    deltaR = math.sqrt(dphi*dphi + drap*drap);
    # get ln kt / momentum fraction and ln Delta
    #lnkt    = math.log(deltaR*pt2)
    lnz     = math.log(pt2/(pt1+pt2))
    lnDelta = math.log(deltaR)

    return np.array([lnz,lnDelta])


def get_window_width(masses, lower_frac=20, upper_frac=80):
    """Returns"""
    lower = np.nanpercentile(masses, lower_frac)
    upper = np.nanpercentile(masses, upper_frac)
    median = np.median(masses[(masses > lower) & (masses < upper)])
    return lower, upper, median