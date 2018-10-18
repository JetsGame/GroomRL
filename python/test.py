from grooming import *
from tools import *
from create_image import Jets
import numpy as np
import pickle
from matplotlib import pyplot as plt


reader = Jets('../constit.json.gz',500)
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
with open('test.pickle','rb') as rfp:
    mdqn = pickle.load(rfp)
    
bins = np.arange(0, 401, 10)
plt.hist(mplain, bins=bins, alpha=0.5, label='plain')
plt.hist(msd, bins=bins, alpha=0.5, label='SD $(\\beta=0,z_\\mathrm{cut}=0.1)$')
plt.hist(mrsd, bins=bins, alpha=0.5, label='RSD $(\\beta=1,z_\\mathrm{cut}=0.1)$')
plt.hist(mdqn, bins=bins, alpha=0.5, label='DQN-Grooming')
plt.xlim((0,350))
plt.legend()
plt.show()
