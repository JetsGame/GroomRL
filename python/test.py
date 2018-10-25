from grooming import *
from tools import *
from create_image import Jets
import numpy as np
import pickle
from matplotlib import pyplot as plt

# def debug_rsd():
#     from GroomEnv import GroomEnvSD
#     from MLGroomer import run_model
#     import os
#     fnres = 'test_RSD.pickle'
#     if os.path.exists(fnres):
#         os.remove(fnres)
#     env = GroomEnvSD('../sample_WW_2TeV_CA.json.gz', mass=80.385,
#                      mass_width = 1.0, nev=10000, target_prec = 0.05)
#     dqn, _ = run_model('Dense','../sample_WW_2TeV_CA.json.gz',1,80.385,1.0,1)
#     env.testmode(fnres)
#     dqn.test(env, nb_episodes=10000, visualize=True)

reader = Jets('../sample_WW_2TeV_CA.json.gz',10000)
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

## debug_rsd()
# with open('test_RSD.pickle','rb') as rfp:
#         mdqnRSD = pickle.load(rfp)

bins = np.arange(0, 401, 2)
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(18,14))
plt.hist(mplain, bins=bins, color='C0', alpha=0.3, label='plain')
plt.hist(mrsd, bins=bins, alpha=0.4, color='C2', label='RSD $(\\beta=1,z_\\mathrm{cut}=0.05)$')
plt.hist(msd,  bins=bins, alpha=0.4, color='C1', label='SD $(\\beta=0,z_\\mathrm{cut}=0.1)$')
plt.hist(mdqn,     bins=bins, facecolor='none', edgecolor='C3', lw=2,
         label='DQN-Grooming-Dense', hatch="\\")
plt.hist(mdqnLSTM, bins=bins, facecolor='none', edgecolor='C4', lw=2,
         label='DQN-Grooming-LSTM', hatch="/")
#plt.hist(mdqnRSD, bins=bins, color='C5',alpha=0.5, label='DQN-RSD $(\\beta=1,z_\\mathrm{cut}=0.1)$')
plt.xlim((0,150))
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

