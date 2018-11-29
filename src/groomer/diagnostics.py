from groomer.Groomer import RSD
from groomer.JetTree import *
from groomer.read_clustseq_json import Jets
from groomer.observables import mass
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

#----------------------------------------------------------------------
def print_stats(name, data, mass_ref=80.385, output_folder='./'):
    """Print statistics on the mass distribution."""
    r_plain = np.array(data)-mass_ref
    m = np.median(r_plain)
    a = np.mean(r_plain)
    s = np.std(r_plain)
    with open('%s/diagnostics.txt'%output_folder,'a+') as f:
        print('%s:\tmedian-diff %.2f\tavg-diff %.2f\tstd-diff %.2f' % (name, m, a, s),
              file=f)

#----------------------------------------------------------------------
def plot_mass(groomer, sample_fn, mass_ref=80.385, output_folder='./', zcut=0.05, beta=1.0):
    """Plot the mass distribution and output some statistics."""
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
    plt.savefig('%s/mass.pdf' % output_folder, bbox_inches='tight')
        
    print_stats('plain   ', mplain, mass_ref=mass_ref, output_folder=output_folder)
    print_stats('mrsd    ', mrsd  , mass_ref=mass_ref, output_folder=output_folder)
    print_stats('mdqn    ', mdqn  , mass_ref=mass_ref, output_folder=output_folder)

#----------------------------------------------------------------------
def plot_lund(groomer, sample_fn, zcut=0.05, beta=1.0, output_folder="./"):
    """Plot the lund plane."""
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
    with PdfPages('%s/lund.pdf' % output_folder) as pdf:

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(12, 9))
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
