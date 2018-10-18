import random, math, gym, copy, os, pickle
from create_image import Jets
from tools import *
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import json

# class GroomSpace(Space):
#     def sample(self, seed=None):
#         return 1 - 2*random.randint(0, 1)

#     def contains(self, x):
#         return x in [-1,1]
    
# class GroomEnv(Env):
#     action_space=spaces.Discrete(2)
#     observation_space=
    

class GroomEnv(gym.Env):

    def __init__(self, fn, nev, outfn=None, low=np.array([0.0, -6.0]),
                 high=np.array([10.0, 8.0]), mass=80.385,
                 target_prec = 0.1, mass_width = 2):
        
        # read in the events
        self.fn      = fn
        self.nev     = nev
        self.outfn   = outfn
        reader       = Jets(fn, nev, pseudojets=False)
        self.jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.0)
        self.events  = np.array(reader.values())
            
        self.massgoal      = mass
        self.target_prec   = target_prec
        self.mass_width    = 20
        self.declust_index = 0
        self.current = self.get_random_declust()
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high)

        self.seed()
        self.viewer = None
        self.state = None

    def get_random_declust(self):
        event = random.choice(self.events)
        constits = []
        for p in event:
            constits.append(fj.PseudoJet(p[0],p[1],p[2],p[3]))
        jets = self.jet_def(constits)
        declusts = []
        if len(jets)>0:
            fill_pq(declusts, jets[0])
            ldecl = pq_to_list(declusts)
            res = []
            for jet,parents,tag,children in ldecl:
                j1 = fj.PseudoJet()
                j2 = fj.PseudoJet()
                jet.has_parents(j1,j2)
                if (j2.pt() > j1.pt()):
                    j1,j2=j2,j1
                res.append([[jet.px(),jet.py(),jet.pz(),jet.E()],
                            parents, tag, children,                
                            [j1.px(),j1.py(),j1.pz(),j1.E()],
                            [j2.px(),j2.py(),j2.pz(),j2.E()]])
            return res
        return []

    def get_state(self):
        jet,parents,tag,children,j1,j2 = self.current[self.declust_index]
        # calculate coordinates
        pt1, rap1, phi1 = self.coords(j1)
        pt2, rap2, phi2 = self.coords(j2)
        dphi = abs(phi1 - phi2);
        if dphi > math.pi:
            dphi = 2*math.pi - dphi
        drap = rap1 - rap2;
        deltaR = math.sqrt(dphi*dphi + drap*drap);
        # get ln kt and ln Delta
        lnkt     = math.log(deltaR*pt2)
        lnDelta  = math.log(deltaR)
        return [lnkt,lnDelta]

    def coords(self,jet):
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
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.state = self.get_state()
        declust = self.current
        jet, parents, tag, children, j1, j2 = declust[self.declust_index]
        self.declust_index+=1
        remove_soft = (action==1)
        if remove_soft:
            branch_torem = [min(children)] if len(children)>0 else []
            while branch_torem:
                i=self.declust_index
                while i < len(declust):
                    if declust[i][2] in branch_torem:
                        branch_torem+=declust[i][3]
                        del declust[i]
                    else:
                        i+=1
                del branch_torem[0]
                
            for i in range(self.declust_index):
                declust[i][0] = [a - b for a, b in zip(declust[i][0], j2)]

        # m^2 = declust[0].E()*declust[0].E() - declust[0].px()*declust[0].px() - declust[0].py()*declust[0].py() - declust[0].pz()*declust[0].pz()
        msq  = jet[3]*jet[3] - jet[0]*jet[0] - jet[1]*jet[1] - jet[2]*jet[2]
        mass = math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq)
        massdiff = abs(mass - self.massgoal)
        self.current = declust
        done = bool(self.declust_index >= len(declust))
        
        #reward = min(1.0,1/(self.mass_width*massdiff)) # inverse
        #reward = np.exp(-massdiff*massdiff/(2*self.mass_width*self.mass_width)) # gaussian
        reward = np.exp(-massdiff/self.mass_width) # exponential
        
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.current = self.get_random_declust()
        self.declust_index = 0
        self.state = self.get_state()
        return np.array(self.state)

    def render(self, mode='human'):
        # if True:
        #     jet = self.current[0][0]
        #     msq = jet[3]*jet[3] - jet[0]*jet[0] - jet[1]*jet[1] - jet[2]*jet[2]
        #     print(math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq))
        if (self.declust_index >= len(self.current) and self.outfn):
            masses = []
            if os.path.exists(self.outfn):
                with open(self.outfn,'rb') as rfp: 
                    masses = pickle.load(rfp)
            
            jet = self.current[0][0]
            msq = jet[3]*jet[3] - jet[0]*jet[0] - jet[1]*jet[1] - jet[2]*jet[2]
            masses.append(math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq))

            with open(self.outfn,'wb') as wfp:
                pickle.dump(masses, wfp)

                
    def close(self):
        if self.viewer: self.viewer.close()





class GroomEnvSD(GroomEnv):
    def __init__(self, fn, nev, outfn=None, low=np.array([0.0, -6.0]),
                 high=np.array([10.0, 8.0]), mass=80.385,
                 target_prec = 0.1, mass_width = 2):
        GroomEnv.__init__(self, fn, nev, outfn, low, high,
                          mass, target_prec, mass_width)
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.state = self.get_state()
        declust = self.current
        jet, parents, tag, children, j1, j2 = declust[self.declust_index]
        self.declust_index+=1
        pt1,rap1,phi1 = self.coords(j1)
        pt2,rap2,phi2 = self.coords(j2)
        dphi=phi1-phi2
        drap=rap1-rap2
        remove_soft = (pt2/(pt1+pt2) < 0.1 * math.pow(dphi*dphi + drap*drap,0.5))
        if remove_soft:
            branch_torem = [min(children)] if len(children)>0 else []
            while branch_torem:
                i=self.declust_index
                while i < len(declust):
                    if declust[i][2] in branch_torem:
                        branch_torem+=declust[i][3]
                        del declust[i]
                    else:
                        i+=1
                del branch_torem[0]
                
            for i in range(self.declust_index):
                declust[i][0] = [a - b for a, b in zip(declust[i][0], j2)]

        # m^2 = declust[0].E()*declust[0].E() - declust[0].px()*declust[0].px() - declust[0].py()*declust[0].py() - declust[0].pz()*declust[0].pz()
        msq  = jet[3]*jet[3] - jet[0]*jet[0] - jet[1]*jet[1] - jet[2]*jet[2]
        mass = math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq)
        massdiff = abs(mass - self.massgoal)
        self.current = declust
        done = bool(self.declust_index >= len(declust))
        
        #reward = min(1.0,1/(self.mass_width*massdiff)) # inverse
        #reward = np.exp(-massdiff*massdiff/(2*self.mass_width*self.mass_width)) # gaussian
        reward = np.exp(-massdiff/self.mass_width) # exponential
        
        return np.array(self.state), reward, done, {}
