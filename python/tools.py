import heapq as hq
import fastjet as fj

def pq_to_list(pq):
    """Take an input priority queue (heapq) and return the ordered list, without the delta R value"""
    l=[]
    while(pq):
        l.append(hq.heappop(pq)[1:])

    for el in l:
        for i in range(len(l)):
            if l[i][2] in el[1]:
                if l[i][2]==el[1][-1]:
                    l[i][3].append(el[2])
    return l
    
def fill_pq(pq, j, parents=[], idty=0):
    """Fill the priority queue with all declustered pseudojets."""
    j1 = fj.PseudoJet()
    j2 = fj.PseudoJet()
    if (j.has_parents(j1,j2)):
        if (j2.pt() > j1.pt()):
            j1,j2=j2,j1
        hq.heappush(pq, [-j1.squared_distance(j2), j, parents, idty+1, []])
        fill_pq(pq, j1, parents+[idty+1], idty+1)
        fill_pq(pq, j2, parents+[idty+1], idty+len(j1.constituents()))
