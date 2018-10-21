from GroomEnv import GroomEnv
import numpy as np
from matplotlib import pyplot as plt

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

# construct a DQN network to be used on lund input
# https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
def dqn_construct(hps, env):
    
    # we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + hps['input_dim']))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(hps['nb_actions']))
    model.add(Activation('linear'))
    print(model.summary())
    
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    agent = DQNAgent(model=model, nb_actions=hps['nb_actions'],
                     memory=memory, nb_steps_warmup=10,
                     target_model_update=1e-2, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    
    return agent

def run_model(fn = '../constit-long.json.gz', nev=400000, outfn=None):
    """Run a test model"""

    # set up environment
    env = GroomEnv(fn, nev, outfn=outfn, low=np.array([0.0, -6.0]),
                   high=np.array([10.0, 8.0]), mass=80.385,
                   target_prec = 0.1, mass_width = 2)

    # hyperparameters
    dqn_hps = {
        'input_dim': env.observation_space.shape,
        'nb_actions': 2,
        'description': 'test model',
        'model_name': 'DQN_Test_Model'
    }

    print('Constructing DQN agent...')
    dqn = dqn_construct(dqn_hps, env)

    print('Fitting DQN agent...')
    dqn.fit(env, nb_steps=300000, visualize=False, verbose=2)

    print('Saving weights...')
    # After training is done, we save the final weights.
    dqn.save_weights('../models/'+dqn_hps['model_name']+'.h5', overwrite=True)

    return dqn
    
if __name__ == "__main__":
    dqn = run_model()
    env = GroomEnv('../constit.json.gz', 500, outfn='test.pickle', low=np.array([0.0, -6.0]),
                   high=np.array([10.0, 8.0]), mass=80.385,
                   target_prec = 0.1, mass_width = 2)
    dqn.test(env, nb_episodes=500, visualize=True)
