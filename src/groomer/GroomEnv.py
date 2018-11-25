import random, math, gym, copy, os, pickle
from groomer.read_clustseq_json import Jets
#from tools import declusterings, kinematics_node, coords
from groomer.JetTree import JetTree, LundCoordinates
from gym import spaces, logger
from gym.utils import seeding
import heapq as hq
import fastjet as fj
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
        self.nev     = nev
        reader       = Jets(fn, nev)
        self.events  = reader.values()

        # set up the mass parameters and initial state
        self.massgoal      = mass
        self.target_prec   = target_prec
        self.mass_width    = mass_width
        self.root          = None
        # self.event_index   = -1

        # set up observation and action space
        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high)

        # set up some internal parameters
        self.seed()
        self.viewer = None

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
        # set variables needed for the SD reward
        self.alpha1 = 0.5
        # self.alpha2 = 0.4
        # for alternative implementation
        #self.alpha1 = 1.0
        self.alpha2 = 0.1
        self.SDnorm = 0.05
        # lnzRef is the reference value below which radiation is
        # considered soft and to be groomed
        self.lnzRef1 = -4
        # self.lnzRef2 = -6
        # for alternative implementation
        # self.lnzRef1 = -8
        self.lnzRef2 = -8
        
        self.description= '%s with file=%s, target mass=%.3f, width=%.3f, using %s reward.'\
            % (self.__class__.__name__,fn,mass, mass_width, reward)
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
    def reward_mass(self,mass):
        """For a given jet mass, return the output of the reward function."""
        massdiff = abs(mass - self.massgoal)
        return self.__reward(massdiff/self.mass_width)    
    
    #---------------------------------------------------------------------- 
    def reward_SD(self, lnz, lnDelta, is_groomed):
        """
        For a given jet mass, return the output of the Soft Drop component
        of the reward function.
        """# # 
        if is_groomed:
            reward = min(1.0, math.exp(self.alpha1 * lnDelta + self.alpha1*(self.lnzRef1 - lnz)))
            # alternative implementation
            # reward = min(1.0, math.exp(-self.alpha1 * lnDelta * (self.lnzRef1 - lnz)))
        else:
            #reward = max(0.0, 1.0 - math.exp(self.alpha2 * lnDelta + self.alpha2*(self.lnzRef2 - lnz)))
            # alternative implementation
            reward = max(0.0, 1.0 - math.exp(-self.alpha2 * lnDelta * (self.lnzRef2 - lnz)))
        return self.SDnorm*reward
    
    #---------------------------------------------------------------------- 
    def reward(self, mass, lnz, lnDelta, is_groomed):
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
        lnz, lnDelta = self.state

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
    
        # get the current state
        tree = self.current
        lnz, lnDelta = self.state
        
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
