from grooming import *
from tools import *
from create_image import Jets
import numpy as np
import pickle
from matplotlib import pyplot as plt


reader = Jets('../sample_WW_2TeV_CA.json.gz',5000)
events_jet = reader.values()
events = []
for jet in events_jet:
    declusts = []
    fill_pq(declusts, jet)
    events.append(pq_to_list(declusts))

mplain=[]
mrsd=[]
msd=[]
for ev in events:
    mplain.append(ev[0][0].m())
    sdev=rsd_groom(ev,0.0,0.1)
    rsdev=rsd_groom(ev,1.0,0.05,N=100000)
    if sdev:
        msd.append(sdev[0][0].m())
    if rsdev:
        mrsd.append(rsdev[0][0].m())

# now read the DQN masses
with open('test_Dense.pickle','rb') as rfp:
    mdqn = pickle.load(rfp)

with open('test_LSTM.pickle','rb') as rfp:
    mdqnLSTM = pickle.load(rfp)

# from GroomEnv import GroomEnvSD
# from MLGroomer import run_model
# env = GroomEnvSD('../sample_WW_2TeV_CA.json.gz', 10000, outfn='test_RSD.pickle',
#                  low=np.array([0.0, -6.0]),
#                  high=np.array([10.0, 8.0]), mass=80.385,
#                  target_prec = 0.1, mass_width = 1.0)
# dqn = run_model('Dense','../sample_WW_2TeV_CA.json.gz',1,80.385,1.0,1)
# dqn.test(env, nb_episodes=5000, visualize=True)
# with open('test_RSD.pickle','rb') as rfp:
#     mdqnRSD = pickle.load(rfp)
    
bins = np.arange(0, 401, 4)
plt.hist(mplain, bins=bins, alpha=0.5, label='plain')
plt.hist(msd, bins=bins, alpha=0.5, label='SD $(\\beta=0,z_\\mathrm{cut}=0.1)$')
plt.hist(mrsd, bins=bins, alpha=0.5, label='RSD $(\\beta=1,z_\\mathrm{cut}=0.1)$')
plt.hist(mdqn, bins=bins, alpha=0.5, label='DQN-Grooming-Dense')
plt.hist(mdqnLSTM, bins=bins, alpha=0.5, label='DQN-Grooming-LSTM')
# plt.hist(mdqnRSD, bins=bins, alpha=0.5, label='DQN-RSD $(\\beta=0,z_\\mathrm{cut}=0.1)$')
plt.xlim((0,300))
plt.legend()
plt.savefig('test.png',bbox_inches='tight')

# printing some stats
def print_stats(name, data, refmass=80.385):
    r_plain = np.array(data)-refmass
    m = np.median(r_plain)
    a = np.mean(r_plain)
    s = np.std(r_plain)
    print('%s:\tmedian-diff %.2f\tavg-diff %.2f\tstd-diff %.2f' % (name, m, a, s))

print_stats('plain   ', mplain)
print_stats('msd     ', msd)
print_stats('mrsd    ', mrsd)
print_stats('mdqn    ', mdqn)
print_stats('mdqnLSTM', mdqnLSTM)

