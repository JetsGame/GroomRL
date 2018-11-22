from Groomer import Groomer
from rl.agents.dqn import DQNAgent


#----------------------------------------------------------------------
class DQNAgentGroom(DQNAgent):
    """DQN Agent for jet grooming"""
    def __init__(self, *args, **kwargs):
        """Initialize the DQN agent."""
        super(DQNAgentGroom, self).__init__(*args, **kwargs)

    def groomer(self):
        """Return the current groomer."""
        return Groomer(self.model, self.test_policy)
        
    # def save_model():
    #     # save to json

    # def load_mode(fn):
    #     # load model from json

    
