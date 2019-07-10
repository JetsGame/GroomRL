# This file is part of GroomRL by S. Carrazza and F. A. Dreyer

import random, math, pprint
from groomrl.read_data import Jets
from groomrl.JetTree import JetTree, LundCoordinates
from gym import spaces, Env
from gym.utils import seeding
import heapq as hq
import numpy as np

#======================================================================
class GroomEnv(Env):
    """Class defining a gym environment for the groomer."""
    #----------------------------------------------------------------------
    def __init__(self, hps, low, high):
        """
        Initialisation of the environment using a dictionary with hyperparameters
        and a lower and upper bound for the observable state.

        The hps dictionary should have the following entries:
        - fn: filename of data set
        - nev: number of events (-1 for all)
        - mass: target mass reference
        - mass_width: width to use in reward function
        - reward: type of reward function (cauchy, gaussian, ...)
        - SD_groom: type of SD reward for groomed subjets (exp_add or exp_mult)
        - SD_keep: type of SD reward for kept subjets (exp_add or exp_mult)
        - alpha1: parameter for groomed SD reward
        - beta1: parameter for groomed SD reward
        - alpha2: parameter for kept SD reward
        - beta2: parameter for kept SD reward
        - SDnorm: normalisation factor for SD reward
        - lnzRef1: parameter for groomed SD reward
        - lnzRef2: parameter for kept SD reward
        """
        # read in the events
        reader       = Jets(hps['fn'], hps['nev'])
        self.events  = reader.values()

        # set up the mass parameters and initial state
        self.massgoal      = hps['mass']
        self.mass_width    = hps['width']
        self.root          = None
        # self.event_index   = -1

        # set up observation and action space
        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        print('shape:',self.observation_space.shape)
        # set up some internal parameters
        self.seed()
        self.viewer = None

        # set the reward functions
        if hps['reward']=='cauchy':
            self._reward=self.__reward_Cauchy
        elif hps['reward']=='gaussian':
            self._reward=self.__reward_Gaussian
        elif hps['reward']=='exponential':
            self._reward=self.__reward_Exponential
        elif hps['reward']=='inverse':
            self._reward=self.__reward_Inverse
        else:
            raise ValueError('Invalid reward: %s'%hps['reward'])
        if hps['SD_groom']=='exp_add':
            self.__reward_Groom=self.__reward_Exp_add
        elif hps['SD_groom']=='exp_mult':
            self.__reward_Groom=self.__reward_Exp_mult
        else:
            raise ValueError('Invalid SD_groom: %s'%hps['SD_groom'])
        if hps['SD_keep']=='exp_add':
            self.__reward_Keep=self.__reward_Exp_add
        elif hps['SD_keep']=='exp_mult':
            self.__reward_Keep=self.__reward_Exp_mult
        else:
            raise ValueError('Invalid SD_keep: %s'%hps['SD_keep'])
        self.alpha1  = hps['alpha1']
        self.beta1   = hps['beta1']
        self.alpha2  = hps['alpha2']
        self.beta2   = hps['beta2']
        self.SDnorm  = hps['SD_norm']
        self.lnzRef1 = hps['lnzRef1']
        self.lnzRef2 = hps['lnzRef2']
        # the separate reward and reward_sig functions are required for
        # the GroomEnvDual class
        self.reward  = self.reward_sig

        self.description= '%s with\n%s' % (self.__class__.__name__,pprint.pformat(hps))
        print('Setting up %s' % self.description)

    #----------------------------------------------------------------------
    def get_random_tree(self):
        """Get a random jet tree from the list of events"""
        # get random event
        event = random.choice(self.events)
        return JetTree(event)

    #----------------------------------------------------------------------
    def reset_current_tree(self):
        """Set up the priority queue with the first node."""
        if self.root:
            del self.root
        self.root       = self.get_random_tree()
        self.current_pq = []
        hq.heappush(self.current_pq, self.root)
        self.set_next_node()

    #----------------------------------------------------------------------
    def set_next_node(self):
        """Set the current declustering node using the priority queue."""
        if not self.current_pq:
            # if priority queue is empty, set to none
            self.current=None
            self.state = np.zeros(LundCoordinates.dimension)
        else:
            # first get the tree node of branch with largest delta R separation
            self.current = hq.heappop(self.current_pq)
            # then set up the internal state to current values
            self.state = self.current.state()

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
        return min(1.0, 1.0/(x + 0.5))

    #----------------------------------------------------------------------
    def __reward_Exp_add(self, lnDelta, lnz, alpha, beta, lnzRef):
        """Exponential of addition of lnDelta and lnz."""
        return math.exp(alpha*lnDelta + beta*(lnzRef - lnz))

    #----------------------------------------------------------------------
    def __reward_Exp_mult(self, lnDelta, lnz, alpha, beta, lnzRef):
        """Exponential of multiplication of lnDelta and lnz."""
        return math.exp(-alpha*lnDelta*(lnzRef - lnz))

    #----------------------------------------------------------------------
    def reward_mass(self,mass):
        """For a given jet mass, return the output of the reward function."""
        massdiff = abs(mass - self.massgoal)
        return self._reward(massdiff/self.mass_width)

    #----------------------------------------------------------------------
    def reward_SD(self, lnz, lnDelta, is_groomed):
        """
        For a given jet mass, return the output of the Soft Drop component
        of the reward function.
        """# #
        if is_groomed:
            reward = min(1.0, self.__reward_Groom(lnDelta, lnz, self.alpha1, self.beta1, self.lnzRef1))
            # reward = min(1.0, math.exp(self.alpha1 * lnDelta + self.alpha1*(self.lnzRef1 - lnz)))
            # alternative implementation
            # reward = min(1.0, math.exp(-self.alpha1 * lnDelta * (self.lnzRef1 - lnz)))
        else:
            reward = max(0.0, 1.0 - self.__reward_Keep(lnDelta, lnz, self.alpha2, self.beta2, self.lnzRef2))
            # reward = max(0.0, 1.0 - math.exp(self.alpha2 * lnDelta + self.alpha2*(self.lnzRef2 - lnz)))
            # alternative implementation
            # reward = max(0.0, 1.0 - math.exp(-self.alpha2 * lnDelta * (self.lnzRef2 - lnz)))
        return self.SDnorm*reward

    #----------------------------------------------------------------------
    def reward_sig(self, mass, lnz, lnDelta, is_groomed):
        """Full reward function."""
        return self.reward_mass(mass) + self.reward_SD(lnz, lnDelta, is_groomed)

    #----------------------------------------------------------------------
    def seed(self, seed=None):
        """Initialize the seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #----------------------------------------------------------------------
    def step(self, action):
        """Perform a step using the current declustering node, deciding whether to groom the soft branch or note and advancing to the next node."""
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        tree = self.current
        lnz, lnDelta = self.state[:2]

        remove_soft = (action==1)
        # if action==1, then we remove the softer branch
        if remove_soft:
            tree.remove_soft()

        # then add the subjets to the priority_queue
        if tree.harder and tree.harder.delta2 > 0.0:
            hq.heappush(self.current_pq, tree.harder)
        if tree.softer and tree.softer.delta2 > 0.0:
            hq.heappush(self.current_pq, tree.softer)

        # calculate the mass
        # m^2 = declust[0].E()*declust[0].E() - declust[0].px()*declust[0].px() - declust[0].py()*declust[0].py() - declust[0].pz()*declust[0].pz()
        jet  = self.root.jet()
        msq  = jet[3]*jet[3] - jet[0]*jet[0] - jet[1]*jet[1] - jet[2]*jet[2]
        mass = math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq)

        # calculate a reward
        reward = self.reward(mass, lnz, lnDelta, action==1)

        # move to the next node in the clustering sequence
        self.set_next_node()

        # if we are at the end of the declustering list, then we are done for this event.
        done = bool(not self.current)

        # return the state, reward, and status
        return self.state, reward, done, {}

    #----------------------------------------------------------------------
    def reset(self):
        """Reset internal list of declusterings to a new random jet."""
        self.reset_current_tree()
        return self.state

    #----------------------------------------------------------------------
    def render(self, mode='human'):
        """Save masses in an output files"""
        pass
    #----------------------------------------------------------------------
    def close(self):
        if self.viewer: self.viewer.close()


#======================================================================
class GroomEnvDual(GroomEnv):
    """Class defining a gym environment for the groomer with both signal and background events."""

    #----------------------------------------------------------------------
    def __init__(self, hps, *args, **kwargs):
        """
        Initialisation of the environment. The dictionary for GroomEnvDual requires
        two additional entries:
        - fn_bkg: file with background events
        - width_bkg: parameter for the background mass reward
        """
        super(GroomEnvDual, self).__init__(hps, *args, **kwargs)
        reader_bkg      = Jets(hps['fn_bkg'], hps['nev'])
        self.events_bkg = reader_bkg.values()
        self.mass_width_bkg = hps['width_bkg']
        self.reward   = self.reward_total
        self.norm_bkg = hps['reward_bkg_norm']
        self.frac_bkg = hps['frac_bkg']

    #----------------------------------------------------------------------
    def get_random_tree(self):
        """Get a random jet tree from either the signal or the background list of events."""
        if random.uniform(0,1) > self.frac_bkg:
            self.signal = True
            event = random.choice(self.events)
        else:
            self.signal = False
            event = random.choice(self.events_bkg)
        return JetTree(event)

    #----------------------------------------------------------------------
    def reward_mass_bkg(self, mass):
        """Reward for current jet mass, for background events."""
        #massdiff = abs(mass - self.massgoal)
        #x = abs(massdiff/self.mass_width_bkg)
        #return x*x/(math.pi*(1.0 + (x*x)))
        x = abs(mass/self.mass_width_bkg)
        return x*np.exp(-x)


    #----------------------------------------------------------------------
    def reward_bkg(self, mass, lnz, lnDelta, is_groomed):
        """Reward function for background events."""
        return self.reward_mass_bkg(mass) + self.reward_SD(lnz, lnDelta, is_groomed)

    #----------------------------------------------------------------------
    def reward_total(self, mass, lnz, lnDelta, is_groomed):
        """Full reward function."""
        if self.signal:
            return self.reward_sig(mass, lnz, lnDelta, is_groomed)
        else:
            return self.norm_bkg*self.reward_bkg(mass, lnz, lnDelta, is_groomed)

#======================================================================
class GroomEnvTriple(GroomEnvDual):
    """Class defining a gym environment for the groomer with two signal samples and one background sample."""

    #----------------------------------------------------------------------
    def __init__(self, hps, *args, **kwargs):
        """
        Initialisation of the environment. The dictionary for GroomEnvDual requires
        two additional entries:
        - fn_bkg: file with background events
        - width_bkg: parameter for the background mass reward
        """
        super(GroomEnvTriple, self).__init__(hps, *args, **kwargs)
        reader2 = Jets(hps['fn2'], hps['nev'])
        self.events2     = reader2.values()
        self.massgoal2   = hps['mass2']
        self.mass_width2 = hps['width2']
        self.reward = self.reward_triple
        self.frac2  = hps['frac2']

    #----------------------------------------------------------------------
    def get_random_tree(self):
        """Get a random jet tree from either the signals or the background list of events."""
        if random.uniform(0,1) > self.frac_bkg:
            if random.uniform(0,1) > self.frac2:
                self.signal = 1
                event = random.choice(self.events)
            else:
                self.signal = 2
                event = random.choice(self.events2)
        else:
            self.signal = 0
            event = random.choice(self.events_bkg)
        return JetTree(event)

    #----------------------------------------------------------------------
    def reward_mass2(self,mass):
        """For a given jet mass, return the output of the reward function."""
        massdiff = abs(mass - self.massgoal2)
        return self._reward(massdiff/self.mass_width2)

    #----------------------------------------------------------------------
    def reward_triple(self, mass, lnz, lnDelta, is_groomed):
        """Full reward function."""
        if self.signal==1:
            return self.reward_mass(mass) + self.reward_SD(lnz, lnDelta, is_groomed)
        elif self.signal==2:
            return self.reward_mass2(mass) + self.reward_SD(lnz, lnDelta, is_groomed)
        else:
            return self.norm_bkg*self.reward_bkg(mass, lnz, lnDelta, is_groomed)

#======================================================================
class GroomEnvSD(GroomEnv):
    """Toy environment which should essentially recreate Recursive Soft Drop. For debugging purposes."""
    #----------------------------------------------------------------------
    def __init__(self, hps, zcut=0.05, beta=1, low=LundCoordinates.low, high=LundCoordinates.high):
        self.zcut = zcut
        self.beta = beta
        GroomEnv.__init__(self, hps, low, high)

    #----------------------------------------------------------------------
    def step(self, action):
        """Perform a grooming step, removing the soft branch if it fails the Soft Drop condition."""
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # get the current state
        tree = self.current
        lnz, lnDelta = self.state[:2]

        # check if soft drop condition is satisfied
        remove_soft = (math.exp(lnz) < self.zcut * math.pow(math.exp(lnDelta), self.beta))
        # if soft drop condition is not verified, remove the soft branch.
        if remove_soft:
            tree.remove_soft()

        # then add the subjets to the priority_queue
        if tree.harder and tree.harder.delta2 > 0.0:
            hq.heappush(self.current_pq, tree.harder)
        if tree.softer and tree.softer.delta2 > 0.0:
            hq.heappush(self.current_pq, tree.softer)

        # move to the next node in clustering sequence
        self.set_next_node()

        # if we are at the end of the declustering list, then we are done for this event.
        done = bool(not self.current)

        # return the state, reward, and status
        return self.state, 0.0, done, {}
