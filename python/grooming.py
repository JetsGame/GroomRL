import fastjet as fj

#----------------------------------------------------------------------
def rsd_groom(declusts, beta, zcut, N=-1, R0=1.0):
    """Apply Recursive Soft Drop grooming to a declustering list"""
    groomed_branches = []
    res = []
    found_hard = 0
    for j, children, tag_j, parents in declusts:
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        # identify the parents
        if (j.has_parents(j1,j2) and not tag_j in groomed_branches):
            # order them in pt
            if (j2.pt() > j1.pt()):
                j1,j2=j2,j1
            # only groom if we have not found N hard prongs yet
            if found_hard < N or N < 0:
                tag_j1 = parents[0]
                tag_j2 = parents[1]
                deltaR = j1.delta_R(j2)
                z      = min(j1.pt(),j2.pt())/(j1.pt() + j2.pt())
                if (z < zcut*pow(deltaR/R0,beta) and \
                    set(children+[tag_j]).isdisjoint(groomed_branches)):
                    # groom away branch tag_j2 from the jet
                    if tag_j2>0:
                        groomed_branches.append(tag_j2)
                    for i in range(len(res)):
                        if res[i][2] in children:
                            res[i][0] = res[i][0] - j2
                    if len(res)==0:
                        res.append([j1,children,tag_j])
                elif set(children+[tag_j]).isdisjoint(groomed_branches):
                    # we found a hard splitting, increase counter
                    res.append([j,children,tag_j])
                    found_hard += 1
            elif set(children+[tag_j]).isdisjoint(groomed_branches):
                # once we found the N hard splittings, add all declusterings
                # that are part of the remaining branches
                res.append([j,children,tag_j])
        elif tag_j in groomed_branches:
            groomed_branches+=[i for i in parents if i>0]
    # for j, children, tag_j in res:
    #     print(j.px(),j.py(),j.pz(),j.E(), children, tag_j)
    return res
