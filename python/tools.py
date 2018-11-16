import heapq as hq
import fastjet as fj

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
