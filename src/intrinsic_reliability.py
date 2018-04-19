from Experiment import *
from iGIF_NP import *
from AEC_Badel import *
from Tools import *
from Filter_Rect_LogSpaced import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import os









R_X = {}

# List separate experiments in separate folder
data_folders_for_separate_experiments = ['seventh_set', 'eighth_set', 'ninth_set']

# For all experiments, extract the cell names
CellNames = {}
for experiment_folder in data_folders_for_separate_experiments:
    folder_path = './' + experiment_folder + '/'
    CellNames[experiment_folder] = [name for name in os.listdir(folder_path) if os.path.isdir(folder_path + name) and '_5HT' in name]


for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:
        print '####################################################'
        print '##########     process cell %s    #########' %cell_name
        print '####################################################'


        #################################################################################################
        # Load data
        #################################################################################################

        path_data = './' + experiment_folder + '/' + cell_name + '/'

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

        # Load test set data
        filename_test = path_data + cell_name + '_test' + ext
        if filename_test.find('.mat') > 0:
            mat_contents = sio.loadmat(filename_test)
            analogSignals = mat_contents['analogSignals']
            times_test = mat_contents['times'];
            times_test = times_test.reshape(times_test.size)
            times_test = times_test*10.**3
            sampling_time_test = times_test[1] - times_test[0]
            for testnum in range(analogSignals.shape[1]):
                voltage_test = analogSignals[0, testnum, :]
                current_test = analogSignals[1, testnum, :] - 5.
                experiment.addTestSetTrace(voltage_test, 10. ** -3, current_test, 10. ** -12,
                                           len(voltage_test) * sampling_time_test, FILETYPE='Array')
        elif filename_test.find('.abf') > 0:
            r = neo.io.AxonIO(filename=filename_test)
            bl = r.read_block()
            times_test = bl.segments[0].analogsignals[0].times.rescale('ms').magnitude
            sampling_time_test = times_test[1] - times_test[0]
            for i in xrange(len(bl.segments)):
                voltage_test = bl.segments[i].analogsignals[0].magnitude
                current_test = bl.segments[i].analogsignals[1].magnitude - 5.
                experiment.addTestSetTrace(voltage_test, 10. ** -3, current_test, 10. ** -12,
                                           len(voltage_test) * sampling_time_test, FILETYPE='Array')


        #################################################################################################
        # PERFORM ACTIVE ELECTRODE COMPENSATION
        #################################################################################################

        # Create new object to perform AEC
        myAEC = AEC_Badel(experiment.dt)

        # Define metaparametres
        myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=experiment.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
        myAEC.p_expFitRange = [3.0,150.0]
        myAEC.p_nbRep = 15

        # Assign myAEC to experiment and compensate the voltage recordings
        experiment.setAEC(myAEC)
        experiment.performAEC()

        ################################################################################
        # Compute intrinsic reliability
        # We use the intrinsic reliability R_X defined in Eq. (2.15) of
        #   Naud et al., Improved Similarity Measures for Small Sets of Spike Trains, NECO 2011
        ################################################################################
        test_spike_trains = []
        test_spike_number = []
        T = []
        for tr in experiment.testset_traces :
            test_spike_trains.append(tr.getSpikeTrain())
            test_spike_number.append(tr.getSpikeNb())
            T.append(tr.T)

        test_spike_trains = np.array(test_spike_trains)
        test_spike_number = np.array(test_spike_number)

        number_of_tests = test_spike_trains.shape[0]

        delta = 200.
        KistlerDotProduct_args = {'delta' : delta }
        Ncoinc = 0.
        sum_of_norms = 0. # = \sum_i || S_i ||^2 where i = 1, ..., number of test sets and S_i = spike train i
                          # || S_i ||^2 = <S_i, S_i>, where <.,.> is hte Kistler dot product used below

        for i in range(number_of_tests) :
            sum_of_norms += SpikeTrainComparator.Md_dotProduct_Kistler(test_spike_trains[i], test_spike_trains[i], KistlerDotProduct_args, experiment.dt)
            for j in range(i+1,number_of_tests) :
                Ncoinc += SpikeTrainComparator.Md_dotProduct_Kistler(test_spike_trains[i], test_spike_trains[j], KistlerDotProduct_args, experiment.dt)
        R_X[cell_name] = 2.*Ncoinc/(sum_of_norms*(number_of_tests-1))

output_file = open('./Results/intrinsic_reliability_window_'+str(delta) + '.dat','w')
output_file.write('#Cell name\tR_X(' + str(delta) + ' ms)\n')

for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:
        output_file.write(cell_name + '\t' + str(R_X[cell_name]) + '\n')
output_file.close()

