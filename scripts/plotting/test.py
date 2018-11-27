from groomer.Groomer import Groomer, RSD
from groomer.JetTree import *
from groomer.observables import *
from groomer.read_clustseq_json import Jets
from groomer.models import build_model
import numpy as np
import pickle, json
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


# printing some stats
def print_stats(name, data, refmass=80.385):
    r_plain = np.array(data)-refmass
    m = np.median(r_plain)
    a = np.mean(r_plain)
    s = np.std(r_plain)
    print('%s:\tmedian-diff %.2f\tavg-diff %.2f\tstd-diff %.2f' % (name, m, a, s))

# plot masses and output stats for a given input file and compare to sample file with RSD
def plot_mass_from_file(params_fn, modelwgts_fn, sample_fn, zcut=0.05, beta=1.0):
    params = json.loads(open(params_fn).read())
    model = build_model(params['groomer_agent'], (LundCoordinates.dimension,))
    groomer = Groomer(model)
    groomer.load_weights(modelwgts_fn)
    plot_mass(groomer, sample_fn, zcut=zcut, beta=beta)
    
def plot_mass(groomer, sample_fn, zcut=0.05, beta=1.0):
    reader  = Jets(sample_fn,10000)
    rsd = RSD(zcut=zcut, beta=beta)
    events  = reader.values()
    jets    = []
    groomed_jets = []
    rsd_jets = []
    for jet in events:
        groomed_jet = groomer(jet)
        rsd_jet = rsd(jet)
        jets.append(np.array([jet.px(),jet.py(),jet.pz(),jet.E()]))
        groomed_jets.append(groomed_jet)
        rsd_jets.append(rsd_jet)
    
    mplain = mass(jets)
    mdqn   = mass(groomed_jets)
    mrsd   = mass(rsd_jets)
    
    # debug_rsd()
    # with open('test_RSD.pickle','rb') as rfp:
    #     constits_dqnRSD = pickle.load(rfp)
    # mdqnRSD = mass(constits_dqnRSD)
    
    bins = np.arange(0, 401, 2)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    plt.hist(mplain, bins=bins, color='C0', alpha=0.3, label='plain')
    plt.hist(mrsd, bins=bins, alpha=0.4, color='C2',
             label='RSD $(z_\\mathrm{cut}='+'{},\\beta={})$'.format(zcut,beta))
    plt.hist(mdqn,     bins=bins, facecolor='none', edgecolor='C3', lw=2,
             label='DQN-Grooming', hatch="\\")

    plt.xlim((0,150))
    plt.legend()
    plt.savefig('test.pdf',bbox_inches='tight')
        
    print_stats('plain   ', mplain)
    print_stats('mrsd    ', mrsd)
    print_stats('mdqn    ', mdqn)

# plot the Lund plane for a given input file, and compare to sample file with RSD
def plot_lund_from_file(params_fn, modelwgts_fn, sample_fn, zcut=0.05, beta=1.0):
    params = json.loads(open(params_fn).read())
    model = build_model(params['groomer_agent'], (LundCoordinates.dimension,))
    groomer = Groomer(model)
    groomer.load_weights(modelwgts_fn)
    plot_lund(groomer, sample_fn, zcut=0.05, beta=1.0)
    
def plot_lund(groomer, sample_fn, zcut=0.05, beta=1.0):
    # set up the reader and get array from file
    xval   = [0.0, 7.0]
    yval   = [-3.0, 7.0]
    reader  = Jets(sample_fn,10000)
    rsd = RSD(zcut=zcut, beta=beta)
    lundImg = LundImage(xval, yval)
    events  = reader.values()
    plain_imgs   = []
    groomed_imgs = []
    rsd_imgs     = []
    for jet in events:
        groomed_tree = groomer(jet, returnTree=True)
        rsd_tree = rsd(jet, returnTree=True)
        groomed_imgs.append(lundImg(groomed_tree))
        plain_imgs.append(lundImg(JetTree(jet)))
        rsd_imgs.append(lundImg(rsd_tree))

    avg_plain   = np.average(plain_imgs, axis=0)
    avg_groomed = np.average(groomed_imgs, axis=0)
    avg_rsd     = np.average(rsd_imgs, axis=0)

    # Plot the result
    with PdfPages('test_lund.pdf') as pdf:

        plt.rcParams.update({'font.size': 20})
        fig=plt.figure(figsize=(12, 9))
        plt.title('Averaged Lund image DQN-Grooming')
        plt.xlabel('$\ln(R / \Delta)$')
        plt.ylabel('$\ln(k_t / \mathrm{GeV})$')
        plt.imshow(avg_groomed.transpose(), origin='lower', aspect='auto',
                   extent=xval+yval, cmap=plt.get_cmap('BuPu'))
        plt.colorbar()
        pdf.savefig()
        plt.close()

        fig=plt.figure(figsize=(12, 9))
        plt.title('Averaged Lund image plain')
        plt.xlabel('$\ln(R / \Delta)$')
        plt.ylabel('$\ln(k_t / \mathrm{GeV})$')
        plt.imshow(avg_plain.transpose(), origin='lower', aspect='auto',
                   extent=xval+yval, cmap=plt.get_cmap('BuPu'))
        plt.colorbar()
        pdf.savefig()
        plt.close()

        fig=plt.figure(figsize=(12, 9))
        plt.title('Averaged Lund image RSD $(z_\\mathrm{cut}='+'{},\\beta={})$'.format(zcut,beta))
        plt.xlabel('$\ln(R / \Delta)$')
        plt.ylabel('$\ln(k_t / \mathrm{GeV})$')
        plt.imshow(avg_rsd.transpose(), origin='lower', aspect='auto',
                   extent=xval+yval, cmap=plt.get_cmap('BuPu'))
        plt.colorbar()
        pdf.savefig()
        plt.close()

#----------------------------------------------------------------------
if __name__ == "__main__":
    plot_mass('../../output/default_dense.json','../../output/weights.h5',
              '../../../data/sample_WW_2TeV_CA.json.gz') 
