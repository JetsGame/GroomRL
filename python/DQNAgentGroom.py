from Groomer import Groomer
from rl.agents.dqn import DQNAgent


#----------------------------------------------------------------------
class DQNAgentGroom(DQNAgent):
    """DQN Agent for jet grooming"""
    def __init__(self, model, policy=None, test_policy=None, run_optimization=False
                 enable_double_dqn=False, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(DQNAgent,self).__init__(model, policy, test_policy, enable_double_dqn,
                                      enable_dueling_network, dueling_type='avg',
                                      *args, **kwargs)
        self.groomer = Groomer(self.model, self.test_policy)
        if self.run_optimization:
            do_scan()
    
        
    def save_model():
        # save to json

    def load_mode(fn):
        # load model from json

    
