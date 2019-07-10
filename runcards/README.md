Runcard setup
=============

Parameters are defined through a runcard which is stored in json format.

It is made of a dictionary with several entries

## groomer_env
The groomer_env entry is itself a dictionary containing all the
environment parameters for the groomer.
- fn: the data set used for the fit
- fn_bkg: training file of background events for the fit
- val: the validation data for the fit
- mass: the target mass
- nev: number of events to load
- nev_val: number of events for the validation data
- width: width parameter used in the reward
- reward: functional form of the mass reward (cauchy, gaussian, exponential, inverse)
- SD_groom: form of the groomed SD reward (exp_add, exp_mult)
- SD_keep: form of the ungroomed SD reward (exp_add, exp_mult)
- alpha1: parameter for groomed SD reward
- beta1: parameter for groomed SD reward
- alpha2: parameter for kept SD reward
- beta2: parameter for kept SD reward
- SDnorm: normalisation factor for SD reward
- lnzRef1: parameter for groomed SD reward
- lnzRef2: parameter for kept SD reward
- state_dim: dimensionality of the observable state
- dual_groomer_env: use GroomEnvDual instead of GroomEnv (true or false)
- fn_bkg: data set used for the background (GroomEnvDual/GroomEnvTriple only)
- frac_bkg: fraction of background data (GroomEnvDual/GroomEnvTriple only)
- width_bkg: parameter used for background reward (GroomEnvDual/GroomEnvTriple only)
- triple_groomer_env: use GroomEnvTriple instead of GroomEnvDual (true or false)
- fn2: the data set used for the second signal sample (GroomEnvTriple only)
- frac2: fraction of second sample used (GroomEnvTriple only)
- mass2: target mass of second sample (GroomEnvTriple only)
- width2: width parameter for second sample (GroomEnvTriple only)

## groomer_agent
The groomer_agent contains all the parameters for the DQN and NN
- learning_rate: learning rate for Adam
- policy: policy used (boltzmann or epsgreedyq)
- enable_dueling_network: enable dueling network option (true or false)
- enable_double_dqn: enable double dqn option (true or false)
- nstep: number of steps in the fit
- architecture: model architecture (Dense, LSTM)
- dropout: value for dropout layer (ignored if 0)
- nb_units: number of units in NN
- nb_layers: number of layers for Dense architecture

## cluster
The parameters for cluster runs
- enable: enable cluster mode (true or false)
- url: url for MongoTrials
- exp_key: exp_key parameter
- max_evals: maximum number of evaluations for the grid search

## test
Contains information about the test data
- fn: test file for diagnostic plots
- fn_bkg: test file of background events for diagnostic plots