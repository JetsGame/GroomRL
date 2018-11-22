from Groomer import Groomer
from rl.agents.dqn import DQNAgent


#----------------------------------------------------------------------
class DQNAgentGroom(DQNAgent):
    """DQN Agent for jet grooming"""
    def __init__(self, *args, **kwargs):
        """Initialize the DQN agent."""
        super(DQNAgentGroom, self).__init__(*args, **kwargs)

    def save(self, filepath, overwrite=False, include_optimizer=True):
        """Save the model to file."""
        self.model.save(filepath, overwrite=overwrite,
                        include_optimizer=include_optimizer)

    def load_model(self, filepath, custom_objects=None, compile=True):
        """Load model from file"""
        self.model = load_model(filepath, custom_objects=custom_objects,
                                compile=compile)
        self.update_target_model_hard()
        
    def groomer(self):
        """Return the current groomer."""
        return Groomer(self.model, self.test_policy)
        
    # def save_model():
    #     # save to json

    # def load_mode(fn):
    #     # load model from json

    
