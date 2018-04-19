import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import neo


experiment_folder = 'eighth_set'

# For all experiments, extract the cell names
CellNames = {}
folder_path = './' + experiment_folder + '/'
CellNames = [name for name in os.listdir(folder_path) if os.path.isdir(folder_path + name) and '_5HT' in name]


for cell_name in CellNames:
    path_data = './' + experiment_folder + '/' + cell_name + '/'
    path_results = './Results/' + cell_name + '/'

    filename = path_data + 'current_steps.abf'

    if os.path.exists(filename):
        r = neo.io.AxonIO(filename=filename)
        bl = r.read_block()


        for i in range(len(bl.segments)):
            voltage_trace = bl.segments[i].analogsignals[0].magnitude
            current_trace = bl.segments[i].analogsignals[1].magnitude
            times = bl.segments[i].analogsignals[0].times.rescale('s').magnitude

            plt.figure(i, figsize=(5, 5./1.618))
            plt.subplot(2, 1, 1)
            plt.plot(times, current_trace)
            plt.xlabel('Time [s]')
            plt.ylabel('Current [pA]')
            plt.subplot(2, 1, 2)
            plt.plot(times, voltage_trace)
            plt.xlabel('Time [s]')
            plt.ylabel('Voltage [mV]')
            plt.tight_layout()
            plt.savefig(path_results + cell_name + '_CurrentSteps_' + str(i) + '.png', format='png')
            plt.close()