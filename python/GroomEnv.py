import random, math, gym, copy, os, pickle
from create_image import Jets
from tools import *
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import json, warnings

    
#----------------------------------------------------------------------
class GroomEnv(gym.Env):
    """Class defining a gym environment for the groomer."""
    #---------------------------------------------------------------------- 
    def __init__(self, fn, mass=80.385, mass_width=1.0, nev=-1, target_prec=0.1, reward='cauchy',
                 low=np.array([-10.0, -8.0]), high=np.array([0.0, 0.0])):
        """Initialisation of the environment."""
        # read in the events
        self.fn      = fn
        self.outfn   = None
        reader       = Jets(fn, nev, pseudojets=False)
        self.jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.0)
        self.events  = np.array(reader.values())

        # set up the mass parameters and initial state
        self.massgoal      = mass
        self.target_prec   = target_prec
        self.mass_width    = mass_width
        self.declust_index = 0
        self.event_index   = -1
        self.current       = self.get_random_declust()

        # set up observation and action space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high)

        # set up some internal parameters
        self.seed()
        self.viewer = None
        self.state  = self.get_state()

        # set the reward function
        if reward=='cauchy':
            self.__reward=self.__reward_Cauchy
        elif reward=='gaussian':
            self.__reward=self.__reward_Gaussian
        elif reward=='exponential':
            self.__reward=self.__reward_Exponential
        elif reward=='inverse':
            self.__reward=self.__reward_Inverse
        else:
            raise ValueError('Invalid reward: %s'%reward)

        self.description= '%s with file=%s, target mass=%.3f, width=%.3f, using %s reward.'\
            % (self.__class__.__name__,fn,mass, mass_width, reward)
        print('Setting up %s' % self.description)

    #---------------------------------------------------------------------- 
    def get_random_declust(self):
        """Get a random declustering from the list of events"""
        # select a random event (or the next one if in testmode)
        if (self.event_index >= 0):
            # this is for test mode only: run sequentially through the
            # events to groom each one once.
            if (self.event_index >= len(self.events)):
                # check if we are beyond range, if so print warning
                # and loop back
                warnings.warn('Requested too many episodes, resetting to beginning of event file')
                self.event_index = 0
            event = self.events[self.event_index]
            self.event_index = self.event_index + 1
        else:
            # if in training mode, take a random event in the list
            event = random.choice(self.events)
        # append the particles to a constits list
        constits = []
        for p in event:
            constits.append(fj.PseudoJet(p[0],p[1],p[2],p[3]))
        # run jet clustering
        jets = self.jet_def(constits)
        # create a list of ordered declusterings
        declusts = []
        if len(jets)>0:
            fill_pq(declusts, jets[0])
            ldecl = pq_to_list(declusts)
            res = []
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
        return []

    #---------------------------------------------------------------------- 
    def get_state(self):
        """Get the state of the current declustering (i.e. Lund coordinates)"""
        curInd=self.declust_index
        if (curInd < 0) or (curInd >= len(self.current)):
            return np.array([0, 0])
        jet,children,tag,parents,j1,j2 = self.current[curInd]
        # calculate coordinates
        pt1, rap1, phi1 = self.coords(j1)
        pt2, rap2, phi2 = self.coords(j2)
        dphi = abs(phi1 - phi2);
        if dphi > math.pi:
            dphi = 2*math.pi - dphi
        drap = rap1 - rap2;
        deltaR = math.sqrt(dphi*dphi + drap*drap);
        # get ln kt / momentum fraction and ln Delta
        #lnkt    = math.log(deltaR*pt2)
        lnz     = math.log(pt2/(pt1+pt2))
        lnDelta = math.log(deltaR)
        # print ([lnz,lnDelta])
        return np.array([lnz,lnDelta])
        #return np.array([lnkt,lnDelta])

    #---------------------------------------------------------------------- 
    def coords(self,jet):
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
    def __reward_Cauchy(self, x):
        """A cauchy reward function."""
        return 1.0/(math.pi*(1.0 + (x*x)))

    #----------------------------------------------------------------------
    def __reward_Gaussian(self, x):
        """A gaussian reward function."""
        return np.exp(-x*x/2.0)

    #----------------------------------------------------------------------
    def __reward_Exponential(self, x):
        """A negative exponential reward function."""
        return  np.exp(-x)

    #----------------------------------------------------------------------
    def __reward_Inverse(self, x):
        """An inverse reward function."""
        return min(1.0, 1.0/x)
    
    #---------------------------------------------------------------------- 
    def reward(self,mass):
        """For a given jet mass, return the output of the reward function."""
        massdiff = abs(mass - self.massgoal)
        return self.__reward(massdiff/self.mass_width)    
    
    # #---------------------------------------------------------------------- 
    # def reward_SD(self, lnz, lnkt):
    #     """
    #     For a given jet mass, return the output of the Soft Drop component
    #     of the reward function.
    #     """
    #     reward = 
    #     return reward
    
    #---------------------------------------------------------------------- 
    def seed(self, seed=None):
        """Initialize the seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #---------------------------------------------------------------------- 
    def step(self, action):
        """Perform a step using the current declustering node, deciding whether to groom the soft branch or note and advancing to the next node."""
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        declust = self.current
        node, children, tag, parents, j1, j2 = declust[self.declust_index]
        self.declust_index+=1
        remove_soft = (action==1)
        # if action==1, then we remove the softer branch
        if remove_soft:
            # add tag of softer child to list of things to delete
            branch_torem = [parents[1]] if parents[1]>0 else []
            while branch_torem:
                # remove all declusterings whose ID is in our list of stuff to delete
                i=self.declust_index
                while i < len(declust):
                    if declust[i][2] in branch_torem:
                        # if we delete the branch, then add its parents (i.e. declust[i][3])
                        # to the list of things to remove (and make sure to only add valid IDs>0)
                        branch_torem+=[j for j in declust[i][3] if j>0]
                        del declust[i]
                    else:
                        i+=1
                del branch_torem[0]
                
            for i in range(self.declust_index):
                # loop over previous declusterings (children) and remove momentum
                # of soft emission if it is a parent of current node
                if declust[i][2] in children+[tag]:
                    declust[i][0] = [a - b for a, b in zip(declust[i][0], j2)]
                    # then remove it also along the j1 or j2 components of the node
                    # with which it is associated
                    # if j1 tag from node i (declust[i][3][0]) is in the list of children, groom it
                    if declust[i][3][0] in children+[tag]:
                        # remove soft emission from j1 of node i (declust[i][4])
                        declust[i][4] = [a - b for a, b in zip(declust[i][4], j2)]
                    # if j2 tag from node i (declust[i][3][1]) is in the list of children, groom it
                    if declust[i][3][1] in children+[tag]:
                        # remove soft emission from j2 of node i (declust[i][5])
                        declust[i][5] = [a - b for a, b in zip(declust[i][5], j2)]

        # calculate the mass
        # m^2 = declust[0].E()*declust[0].E() - declust[0].px()*declust[0].px() - declust[0].py()*declust[0].py() - declust[0].pz()*declust[0].pz()
        jet  = declust[0][0]
        msq  = jet[3]*jet[3] - jet[0]*jet[0] - jet[1]*jet[1] - jet[2]*jet[2]
        mass = math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq)
        
        # calculate a reward, normalised to total number of declusterings
        reward = self.reward(mass)


        # replace the internal declustering list with the current one
        self.current = declust
        self.state = self.get_state()
        # if we are at the end of the declustering list, then we are done for this event.
        done = bool(self.declust_index >= len(declust))

        #print(action," => ",reward,", ",done, "  .. ",self.state)
        # return the state, reward, and status
        return self.state, reward, done, {}

    #---------------------------------------------------------------------- 
    def reset(self):
        """Reset internal list of declusterings to a new random jet."""
        self.current = self.get_random_declust()
        self.declust_index = 0
        self.state = self.get_state()
        return self.state

    #---------------------------------------------------------------------- 
    def render(self, mode='human'):
        """Save masses in an output files"""
        # reached end of grooming and there is an output file
        if (self.declust_index >= len(self.current) and self.outfn):
            constituents = []
            if os.path.exists(self.outfn):
                with open(self.outfn,'rb') as rfp: 
                    constituents = pickle.load(rfp)

            # # doesn't work
            # jet = []
            # for j, children, tag, parents, j1, j2 in self.current:
            #     if parents[0] < 0:
            #         jet.append(j1)
            #     if parents[1] < 0:
            #         jet.append(j2)
            # constituents.append(jet)
            
            constituents.append(self.current[0][0])
            
            with open(self.outfn,'wb') as wfp:
                pickle.dump(constituents, wfp)

    #---------------------------------------------------------------------- 
    def close(self):
        if self.viewer: self.viewer.close()


    #---------------------------------------------------------------------- 
    def testmode(self, outfn, fn=None, nev=-1):
        """Switch the environment to test mode."""
        self.outfn         = outfn
        self.event_index   = 0
        self.declust_index = 0
        if fn:
            self.fn     = fn
            reader      = Jets(fn, nev, pseudojets=False)
            self.events = np.array(reader.values())

        
#======================================================================
class GroomEnvSD(GroomEnv):
    """Toy environment which should essentially recreate Recursive Soft Drop. For debugging purposes."""
    #----------------------------------------------------------------------
    def __init__(self, fn, zcut=0.05, beta=1, mass=80.385, mass_width=1.0,
                 nev=-1, target_prec=0.1, reward='cauchy',
                 low=np.array([0.0, 10.0]), high=np.array([1.0, 10.0])):
        self.zcut = zcut
        self.beta = beta
        GroomEnv.__init__(self, fn, mass, mass_width, nev, target_prec, reward, low, high)
        
    #---------------------------------------------------------------------- 
    def step(self, action):
        """Perform a grooming step, removing the soft branch if it fails the Soft Drop condition."""
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # get the current state, and move the intenral index forward
        self.state = self.get_state()
        declust = self.current
        node, children, tag, parents, j1, j2 = declust[self.declust_index]
        self.declust_index+=1
        # get subjet kinematics
        pt1,rap1,phi1 = self.coords(j1)
        pt2,rap2,phi2 = self.coords(j2)
        dphi=abs(phi1-phi2)
        if dphi>math.pi:
            dphi = 2*math.pi - dphi
        drap=rap1-rap2
        # check if soft drop condition is satisfied
        remove_soft = (pt2/(pt1+pt2) < self.zcut * math.pow(dphi*dphi + drap*drap, self.beta/2))
        # if soft drop condition is not verified, remove the soft branch.
        if remove_soft:
            # add tag of softer child to list of things to delete
            branch_torem = [parents[1]] if parents[1]>0 else []
            while branch_torem:
                # remove all declusterings whose ID is in our list of stuff to delete
                i=self.declust_index
                while i < len(declust):
                    if declust[i][2] in branch_torem:
                        # if we delete the branch, then add its parents (i.e. declust[i][3])
                        # to the list of things to remove (and make sure to only add valid IDs>0)
                        branch_torem+=[j for j in declust[i][3] if j>0]
                        del declust[i]
                    else:
                        i+=1
                del branch_torem[0]
                
            for i in range(self.declust_index):
                # loop over previous declusterings (children) and remove momentum
                # of soft emission if it is a parent of current node
                if declust[i][2] in children+[tag]:
                    declust[i][0] = [a - b for a, b in zip(declust[i][0], j2)]
                    # then remove it also along the j1 or j2 components of the node
                    # with which it is associated
                    # if j1 tag from node i (declust[i][3][0]) is in the list of children, groom it
                    if declust[i][3][0] in children+[tag]:
                        # remove soft emission from j1 of node i (declust[i][4])
                        declust[i][4] = [a - b for a, b in zip(declust[i][4], j2)]
                    # if j2 tag from node i (declust[i][3][1]) is in the list of children, groom it
                    if declust[i][3][1] in children+[tag]:
                        # remove soft emission from j2 of node i (declust[i][5])
                        declust[i][5] = [a - b for a, b in zip(declust[i][5], j2)]
            
        # calculate the mass
        # m^2 = declust[0].E()*declust[0].E() - declust[0].px()*declust[0].px() - declust[0].py()*declust[0].py() - declust[0].pz()*declust[0].pz()
        jet = declust[0][0]
        msq  = jet[3]*jet[3] - jet[0]*jet[0] - jet[1]*jet[1] - jet[2]*jet[2]
        mass = math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq)

        # calculate a reward, normalised to total number of declusterings
        reward = self.reward(mass)

        # replace the internal declustering list with the current one
        self.current = declust
        # if we are at the end of the declustering list, then we are done for this event.
        done = bool(self.declust_index >= len(declust))
                
        # return the state, reward, and status
        return np.array(self.state), reward, done, {}
