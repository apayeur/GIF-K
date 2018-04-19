from Experiment import *
from GIF import *
from AEC_Badel import *
from Tools import *
from Filter_Rect_LogSpaced import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

path_results = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/'


def min_max_mean(I):
    min = np.min(I)
    max = np.max(I)
    mean = np.mean(I)
    return (min, max, mean)

# List separate experiments in separate folder
data_folders_for_separate_experiments = ['seventh_set', 'eighth_set', 'ninth_set']

# For all experiments, extract the cell names
cell_count = 0
CellNames = {}
for experiment_folder in data_folders_for_separate_experiments:
    folder_path = './' + experiment_folder + '/'
    CellNames[experiment_folder] = [name for name in os.listdir(folder_path) if os.path.isdir(folder_path + name) and '_5HT' in name]
    cell_count += len(CellNames[experiment_folder])


number_of_rows = int(np.sqrt(cell_count))
number_of_cols = number_of_rows + 1
fig, ax = plt.subplots(number_of_rows, number_of_cols, sharex=True, sharey=True, figsize=(10,10./1.618))
subplot_row = 0
subplot_col = 0

counter=0
for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:

        #################################################################################################
        # Load data
        #################################################################################################

        path_data = './' + experiment_folder + '/' + cell_name + '/'
        #path_results = './Results/' + cell_name + '/'

        # Find extension of data files
        file_names = os.listdir(path_data)
        for file_name in file_names:
            if '.abf' in file_name:
                ext = '.abf'
                break
            elif '.mat' in file_name:
                ext = '.mat'
                break

        # Load AEC data
        filename_AEC = path_data + cell_name + '_aec' + ext
        (sampling_timeAEC, voltage_traceAEC, current_traceAEC) = load_AEC_data(filename_AEC)

        # Create experiment
        experiment = Experiment('Experiment 1', sampling_timeAEC)
        experiment.setAECTrace(voltage_traceAEC, 10.**-3, current_traceAEC, 10.**-12, len(voltage_traceAEC)*sampling_timeAEC, FILETYPE='Array')

        # Load training set data and add to experiment object
        filename_training = path_data + cell_name + '_training' + ext
        (sampling_time, voltage_trace, current_trace, time) = load_training_data(filename_training)
        experiment.addTrainingSetTrace(voltage_trace, 10**-3, current_trace, 10**-12, len(voltage_trace)*sampling_time, FILETYPE='Array')
        #Note: once added to experiment, current is converted to nA.

        # Compute spike shapes
        tr = experiment.trainingset_traces[0]
        (support, all_spikes, spike_nb, prev_spike, spike_width) = tr.returnSpikeShapes()

        print min_max_mean(tr.I)


        # Plot
        subplot_row = counter / number_of_cols
        subplot_col = counter % number_of_cols
        #ax[subplot_row, subplot_col].set_title(cell_name[3:6], fontsize=11)
        for k in range(spike_nb):
            ax[subplot_row, subplot_col].plot(support, all_spikes[k], lw=0.1, color='blue', alpha=0.2)
        (support, spike_avg, spike_nb) = tr.computeAverageSpikeShape()
        ax[subplot_row, subplot_col].plot(support, spike_avg, lw=1, color='black')
        #ax[subplot_row, subplot_col].set_xlabel('Time (ms)', fontsize=12)
        #ax[subplot_row, subplot_col].set_ylabel('V (mV)', fontsize=12)
        ax[subplot_row, subplot_col].set_xlim(-5, 10)
        ax[subplot_row, subplot_col].set_ylim(-60, 60)
        ax[subplot_row, subplot_col].text(0.9, 0.9, cell_name[3:6], transform=ax[subplot_row, subplot_col].transAxes,
                fontsize=11, va='top', ha='right')
        counter += 1
fig.set_tight_layout(True)
#plt.show()
plt.savefig(path_results + 'SpikeShape.png')
plt.close()