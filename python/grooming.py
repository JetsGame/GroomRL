import fastjet as fj

def rsd_groom(declusts, beta, zcut, R0=1.0, N=1):
    """Apply Soft Drop grooming to a declustering list"""
    groomed_branches = []
    res = []
    found_hard = 0
    for j, par, i, chi in declusts:
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        j.has_parents(j1,j2)
        if (j2.pt() > j1.pt()):
            j1,j2=j2,j1
        if found_hard < N:
            i1 = i+1
            i2 = i+len(j1.constituents())
            deltaR = j1.delta_R(j2)
            z      = min(j1.pt(),j2.pt())/(j1.pt() + j2.pt())
            if (z < zcut*pow(deltaR/R0,beta) and set(par+[i]).isdisjoint(groomed_branches)):
                # groom away branch i2 from the jet
                groomed_branches.append(i2)
                for j in range(len(res)):
                    if res[j][2] in par:
                        res[j][0] = res[j][0] - j2
            elif set(par+[i]).isdisjoint(groomed_branches):
                # we found a hard splitting, increase counter
                res.append([j,par,i])
                found_hard += 1
        elif set(par+[i]).isdisjoint(groomed_branches):
            # once we found the hard splitting, add all declusterings
            # that are part of the remaining branches
            res.append([j,par,i])
    return res
