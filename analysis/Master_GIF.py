#####################################################
#   Master_GIF.py
#   Process all neurons and extract GIF parameters.
#   Output files:
#       (1) CellName_GIF_Raster.png : Raster of GIF model vs experiments
#       (2) CellName_GIF_FitPerformance.dat : Md* Epsilon_V(test) PVar for cell CellName
#       (3) GIF_FitPerformance.dat : Md* Epsilon_V(test) for all cells
#       (4) CellName_GIF_ModelParams.pck : Model parameters and filters
#   Output folders for files 1,2 and 4 : './Results/CellName/'
#   Output folder for file 3 : './Results/'
#####################################################
from Experiment import *
from GIF import *
from AEC_Badel import *
from Tools import *
from Filter_Rect_LogSpaced import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import os


Md_star = {}
epsilon_V_test = {}
PVar = {}

# List separate experiments in separate folder
data_folders_for_separate_experiments = ['seventh_set', 'eighth_set', 'ninth_set']

# For all experiments, extract the cell names
CellNames = {}
for experiment_folder in data_folders_for_separate_experiments:
    folder_path = './' + experiment_folder + '/'
    CellNames[experiment_folder] = [name for name in os.listdir(folder_path) if os.path.isdir(folder_path + name) and '_5HT' in name]


for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:
        print '#############################################'
        print '##########     process cell %s    ###' %cell_name
        print '#############################################'


        #################################################################################################
        # Load data
        #################################################################################################

        path_data = './' + experiment_folder + '/' + cell_name + '/'
        path_results = './Results/' + cell_name + '/'

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

        # Determine the refractory period


        #################################################################################################
        # FIT STANDARD GIF
        #################################################################################################

        # Create a new object GIF
        GIF_fit = GIF(sampling_time)

        # Define parameters
        GIF_fit.Tref = 6.0

        GIF_fit.eta = Filter_Rect_LogSpaced()
        GIF_fit.eta.setMetaParameters(length=2000.0, binsize_lb=0.5, binsize_ub=500.0, slope=10.0)

        GIF_fit.gamma = Filter_Rect_LogSpaced()
        GIF_fit.gamma.setMetaParameters(length=2000.0, binsize_lb=2.0, binsize_ub=500.0, slope=5.0)

        for tr in experiment.trainingset_traces:
            tr.setROI([[2000., sampling_time * (len(voltage_trace) - 1) - 2000.]])

        (var_explained_dV, var_explained_V_GIF_train) = GIF_fit.fit(experiment, DT_beforeSpike=5.0)
        GIF_fit.save(path_results + cell_name + '_GIF_ModelParams' + '.pck')


        ###################################################################################################
        # EVALUATE MODEL PERFORMANCES ON THE TEST SET DATA
        ###################################################################################################

        # predict spike times in test set
        prediction = experiment.predictSpikes(GIF_fit, nb_rep=500)

        # Compute epsilon_V
        epsilon_V = 0.
        local_counter = 0.
        for tr in experiment.testset_traces:
            SSE = 0.
            VAR = 0.
            # tr.detectSpikesWithDerivative(threshold=10)
            (time, V_est, eta_sum_est) = GIF_fit.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
            indices_tmp = tr.getROI_FarFromSpikes(5., GIF_fit.Tref)

            SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp]) ** 2)
            VAR += len(indices_tmp) * np.var(tr.V[indices_tmp])
            epsilon_V += 1.0 - SSE / VAR
            local_counter += 1
        epsilon_V = epsilon_V / local_counter
        epsilon_V_test[cell_name] = epsilon_V

        # Compute Md*
        Md_star[cell_name] = prediction.computeMD_Kistler(8.0, GIF_fit.dt*2.)
        fname = path_results  + cell_name  + '_GIF_Raster.png'
        kernelForPSTH = 50.0
        PVar[cell_name] = prediction.plotRaster(fname, delta=kernelForPSTH)


        #################################################################################################
        #  PLOT TRAINING AND TEST TRACES, MODEL VS EXPERIMENT
        #################################################################################################

        #Comparison for training and test sets w/o inactivation
        V_training = experiment.trainingset_traces[0].V
        I_training = experiment.trainingset_traces[0].I
        (time, V, eta_sum, V_t, S) = GIF_fit.simulate(I_training, V_training[0])
        fig = plt.figure(figsize=(10,6), facecolor='white')
        plt.subplot(2,1,1)
        plt.plot(time/1000, V,'--r', lw=0.5, label='GIF')
        plt.plot(time/1000, V_training,'black', lw=0.5, label='Data')
        plt.xlim(18,20)
        plt.ylim(-80,20)
        plt.ylabel('Voltage [mV]')
        plt.title('Training')

        V_test = experiment.testset_traces[0].V
        I_test = experiment.testset_traces[0].I
        (time, V, eta_sum, V_t, S) = GIF_fit.simulate(I_test, V_test[0])
        plt.subplot(2,1,2)
        plt.plot(time/1000, V,'--r', lw=0.5, label='GIF')
        plt.plot(time/1000, V_test,'black', lw=0.5, label='Data')
        plt.xlim(5,7)
        plt.ylim(-80,20)
        plt.xlabel('Times [s]')
        plt.ylabel('Voltage [mV]')
        plt.title('Test')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_results  + cell_name + '_GIF_simulate.png', format='png')
        plt.close()

        # Figure comparing V_model, V_data and I during training with forced spikes
        (time, V, eta_sum) = GIF_fit.simulateDeterministic_forceSpikes(I_training, V_training[0], experiment.trainingset_traces[0].getSpikeTimes())
        fig = plt.figure(figsize=(10,6), facecolor='white')
        plt.subplot(2,1,1)
        plt.plot(time/1000, I_training,'-b', lw=0.5, label='$I$')
        plt.xlim(17,20)
        plt.ylabel('Injected current [nA]')
        plt.title('Training')
        plt.subplot(2,1,2)
        plt.plot(time/1000, V,'-b', lw=0.5, label='GIF')
        plt.plot(time/1000, V_training,'black', lw=0.5, label='Data')
        plt.xlim(17,20)
        plt.ylim(-75,0)
        plt.ylabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        plt.legend(loc='best')
        plt.savefig(path_results  + cell_name + '_GIF_simulateForcedSpikes_Training.png', format='png')
        plt.close(fig)

output_file = open('./Results/' + 'GIF_FitPerformance.dat','w')
output_file.write('#Cell name\tMd*\tEpsilonV\tPVar\n')

for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:
        output_file.write(cell_name + '\t' + str(Md_star[cell_name]) + '\t' + str(epsilon_V_test[cell_name]) + '\t' + str(PVar[cell_name]) + '\n')
output_file.close()
