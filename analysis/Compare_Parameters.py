from Experiment import *
from GIF_K import *
from iGIF_NP import *
from GIF import *
from AEC_Badel import *
from Tools import *
from Filter_Rect_LogSpaced import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

def compare_params(model_type='GIF'):

    # For all experiments, extract the cell names
    folder_path = './Results/'
    CellNames = [name for name in os.listdir(folder_path) if
                                        os.path.isdir(folder_path + name) and '_5HT' in name]


    # Gather all models
    models = []
    for cell_name in CellNames:
        file_path = folder_path + cell_name + '/' + cell_name + '_' + model_type + '_ModelParams.pck'
        print file_path
        if os.path.exists(file_path):
            models.append(GIF.load(file_path))


    params = {'El': [], 'taum': [], 'C': [], 'DV': []}


    for model in models:
        params['El'].append(model.El)
        params['taum'].append(model.C/model.gl)
        params['C'].append(model.C)
        params['DV'].append(model.DV)

    fig = plt.figure(1, figsize=(8, 3))
    fig.suptitle(model_type + ' model parameters for 5-HT neurons', y=1)
    ax1 = fig.add_subplot(131)
    ax1.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    ax1.boxplot(params['El'], showmeans=True)
    plt.ylabel(r'$E_L$ [mV]')
    ax2 = fig.add_subplot(132)
    ax2.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are of
    plt.ylabel(r'$\tau_m$ [ms]')
    ax2.boxplot(params['taum'], showmeans=True)
    ax3 = fig.add_subplot(133)
    ax3.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    plt.ylabel(r'$\Delta V$ [mV]')
    ax3.boxplot(params['DV'], showmeans=True)
    fig.tight_layout()
    plt.savefig('EIFparams5HT'+model_type+'.png', format='png')
    plt.close(fig)

    # Plot kernel
    K_all = []
    eta_all = []
    gamma_all = []
    plt.figure(2, figsize=(8,3))
    K_support = np.linspace(0, 150.0, 300)
    for model in models:
        K = 1. / model.C * np.exp(-K_support / (model.C / model.gl))
        K_all.append(K)
        plt.subplot(1, 3, 1)
        plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
        (p_eta_support, p_eta) = model.eta.getInterpolatedFilter(model.dt)
        eta_all.append(p_eta)
        plt.subplot(1, 3, 2)
        plt.plot(p_eta_support, p_eta, color='0.3', lw=1, zorder=5)
        plt.subplot(1, 3, 3)
        (p_gamma_support, p_gamma) = model.gamma.getInterpolatedFilter(model.dt)
        gamma_all.append(p_gamma)
        plt.plot(p_gamma_support, p_gamma, color='0.3', lw=1, zorder=5)
    K_mean = np.mean(K_all, axis=0)
    K_std = np.std(K_all, axis=0)
    eta_mean = np.mean(eta_all, axis=0)
    eta_std = np.std(eta_all, axis=0)
    gamma_mean = np.mean(gamma_all, axis=0)
    gamma_std = np.std(gamma_all, axis=0)


    plt.subplot(1, 3, 1)
    plt.fill_between(K_support, K_mean + K_std, y2=K_mean - K_std, color='gray', zorder=0)
    plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)
    plt.xlabel('Time [ms]')
    plt.ylabel('Membrane filter, $\kappa$ [MOhm/ms]')

    plt.subplot(1, 3, 2)
    plt.fill_between(p_eta_support, eta_mean + eta_std, y2=eta_mean - eta_std, color='gray', zorder=0)
    plt.plot(p_eta_support, eta_mean, color='red', lw=2, zorder=10)
    plt.xlim(0, 500)
    plt.ylim(0, 0.1)
    plt.xlabel('Time [ms]')
    plt.ylabel('Spike-triggered\nadaptation current, $\eta$ [nA]')

    plt.subplot(1, 3, 3)
    plt.fill_between(p_gamma_support, gamma_mean + gamma_std, y2=gamma_mean - gamma_std, color='gray', zorder=0)
    plt.plot(p_gamma_support, gamma_mean, color='red', lw=2, zorder=10)
    plt.xlim(0,250)
    plt.ylim(0, 20)
    plt.xlabel('Time [ms]')
    plt.ylabel('Spike-triggered\nmoving threshold, $\gamma$ [mV]')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    compare_params(model_type='GIF')