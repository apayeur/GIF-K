from Experiment import *
from GIF_K import *
from AEC_Badel import *
from Tools import *
from Filter_Rect_LogSpaced import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import neo


# Dictionaries for voltage and current traces
all_voltage_traces = {}
all_current_traces = {}
all_time_traces = {}
experiment_counter = 0

# List separate experiments in separate folder
data_folders_for_separate_experiments = ['seventh_set', 'eighth_set', 'ninth_set', 'tenth_set']

# For all experiments, extract the cell names
CellNames = {}
for experiment_folder in data_folders_for_separate_experiments:
    folder_path = './' + experiment_folder + '/'
    CellNames[experiment_folder] = [name for name in os.listdir(folder_path) if os.path.isdir(folder_path + name) and '_5HT' in name]



for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:
        print '\n\n#############################################'
        print '##########     process cell %s    ###' %cell_name
        print '#############################################'


        #################################################################################################
        # Load data
        #################################################################################################

        path_data = './' + experiment_folder + '/' + cell_name + '/'
        path_results = './Results/' + cell_name + '/'

        if not os.path.exists(path_results):
            os.makedirs(path_results)

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

        # Create new object to perform AEC
        myAEC = AEC_Badel(experiment.dt)

        # Define metaparametres
        myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=experiment.dt, binsize_ub=2.0, slope=30.0,
                                      clamp_period=1.0)
        myAEC.p_expFitRange = [3.0, 150.0]
        myAEC.p_nbRep = 15

        # Assign myAEC to experiment and compensate the voltage recordings
        experiment.setAEC(myAEC)
        experiment.performAEC()

        # Put voltage and current traces in container
        all_voltage_traces[cell_name] = experiment.trainingset_traces[0].V
        all_current_traces[cell_name] = experiment.trainingset_traces[0].I
        all_time_traces[cell_name] = time
        experiment_counter+=1

# Group 1 = 8 cells
Group1 = ['DRN157_5HT', 'DRN159_5HT', 'DRN160_5HT', 'DRN162_5HT',
          'DRN163_5HT', 'DRN164_5HT', 'DRN165_5HT', 'DRN156_5HT']
#Group2 = 10 cells
Group2 = ['DRN652_5HT', 'DRN651_5HT', 'DRN655_5HT', 'DRN660_5HT', 'DRN656_5HT',
          'DRN653_5HT', 'DRN654_5HT', 'DRN659_5HT', 'DRN657_5HT', 'DRN539_5HT']
Group3 = ['DRN544_5HT', 'DRN543_5HT']



# Plotting group-wise traces
fig1 = plt.figure(1, figsize=(5,8))
ax1 = fig1.add_subplot(len(Group1)+1, 1, 1)
ax1.set_xlim(20, 30)
ax1.axis('off')
for i in range(len(Group1)):
    ax1.plot(all_time_traces[Group1[i]] / 1000., 1000*all_current_traces[Group1[i]], lw=0.2, label=Group1[i])
    ax = fig1.add_subplot(len(Group1) + 1, 1, i+2)
    ax.plot(all_time_traces[Group1[i]] / 1000., all_voltage_traces[Group1[i]], lw=0.5)
    #ax.text(0.5, 0.95, cell_name, transform=ax.transAxes, fontsize=10)
    ax.set_xlim(20, 30)
    ax.set_ylim(-100, 0)
    ax.axis('off')
ax1.legend()
plt.tight_layout()
plt.show()


# Plotting traces
fig1 = plt.figure(1, figsize=(9,8))
i=0
for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:
        i+=1
        ax = fig1.add_subplot(experiment_counter/2+1,2,i)
        ax.plot(all_time_traces[cell_name]/1000., all_voltage_traces[cell_name], lw=0.5)
        ax.text(0.5, 0.95, cell_name, transform=ax.transAxes, fontsize=10)
        ax.set_xlim(20, 30)
        ax.set_ylim(-100,0)
        ax.axis('off')
plt.tight_layout()

fig2 = plt.figure(2, figsize=(9,8))
i=0
for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:
        i+=1
        ax = fig2.add_subplot(experiment_counter/2+1,2,i)
        ax.plot(all_time_traces[cell_name]/1000., all_current_traces[cell_name], lw=0.5)
        ax.text(0.5, 0.95, cell_name, transform=ax.transAxes, fontsize=10)
        ax.set_xlim(20, 30)
        ax.axis('off')
plt.tight_layout()

plt.show()