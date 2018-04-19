from Experiment import *

from GIF import *
from GIF_Ca import *
from AEC_Badel import *
from Tools import *


from Filter_Rect_LinSpaced import *
from Filter_Rect_LogSpaced import *
from Filter_Exps import *

import matplotlib.pyplot as plt
import numpy as np
import copy
import json

import scipy
from scipy import io
import cPickle as pkl
import time


"""
This script load some experimental data set acquired according to the experimental protocol
discussed in Pozzorini et al. PLOS Comp. Biol. 2015 and fit the GIF-Ca model.

The performance of the models is assessed on a test set (as described in Pozzorini et al. 2015) by computing
the spike train similarity measure Md*. The test dataset consists of 9 injections of a frozen noise signal
generated according to an Ornstein-Uhlenbeck process whose standard deviation was modulated with a sin function.
"""

CELL_NAME = 'DRN539_5HT'
PATH_DATA = '/Users/alexp/Dropbox/Recherches/Raphe/GIF/Data/seventh_set/'+CELL_NAME+'/'
PATH_RESULTS = '/Users/alexp/Dropbox/Recherches/Raphe/GIF/Results/'+CELL_NAME+'/'
SPECIFICATION = ''
ADDITIONAL_SPECIFIER = ''

############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################
# Load AEC data
#Convert .mat file containing voltage and current traces into numpy arrays
filename_AEC = PATH_DATA + CELL_NAME + ADDITIONAL_SPECIFIER + '_AEC.mat'
(sampling_timeAEC, voltage_traceAEC, current_traceAEC) = load_AEC_data(filename_AEC)

# Create experiment
experiment = Experiment('Experiment 1', sampling_timeAEC)
experiment.setAECTrace(voltage_traceAEC, 10.**-3, current_traceAEC, 10.**-12, len(voltage_traceAEC)*sampling_timeAEC, FILETYPE='Array')

# Load training set data and add to experiement object
filename_training = PATH_DATA + CELL_NAME +  ADDITIONAL_SPECIFIER + '_training' + SPECIFICATION + '.mat'
(sampling_time, voltage_trace, current_trace) = load_training_data(filename_training)
experiment.addTrainingSetTrace(voltage_trace, 10**-3, current_trace, 10**-12, len(voltage_trace)*sampling_time, FILETYPE='Array')
#Note: once added to experiment, current is converted to nA.


# Load test set data
filename_test = PATH_DATA + CELL_NAME +  ADDITIONAL_SPECIFIER + '_test' + SPECIFICATION + '.mat'
mat_contents = sio.loadmat(filename_test)
analogSignals = mat_contents['analogSignals']
times_test = mat_contents['times'];
times_test = times_test.reshape(times_test.size)
times_test = times_test*10.**3
sampling_time_test = times_test[1] - times_test[0]

for testnum in range(analogSignals.shape[1]):
    voltage_test = analogSignals[0,testnum,:]
    current_test = analogSignals[1,testnum,:] - 5.
    experiment.addTestSetTrace(voltage_test, 10.**-3, current_test, 10.**-12, len(voltage_test)*sampling_time_test, FILETYPE='Array')

# Plot data
#experiment.plotTrainingSet()
#experiment.plotTestSet()


#################################################################################################
# STEP 2: PERFORM ACTIVE ELECTRODE COMPENSATION
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

# Plot AEC filters (Kopt and Ke)
#myAEC.plotKopt()
#myAEC.plotKe()


################################################################################
#
# Compute intrinsic reliability
# We use the intrinsic reliability R_X defined in Eq. (2.15) of
#   Naud et al., Improved Similarity Measures for Small Sets of Spike Trains, NECO 2011
#
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
delta = 8.
intrinsic_reliability = []
#for delta in delta_array :
KistlerDotProduct_args =  {'delta':delta}
local_counter=0
Ncoinc = 0.
sum_of_norms = 0. # = \sum_i || S_i ||^2 where i = 1, ..., number of test sets and S_i = spike train i
                  # || S_i ||^2 = <S_i, S_i>, where <.,.> is hte Kistler dot product used below

for i in range(number_of_tests) :
    sum_of_norms += SpikeTrainComparator.Md_dotProduct_Kistler(test_spike_trains[i], test_spike_trains[i], KistlerDotProduct_args, experiment.dt)
    for j in range(i+1,number_of_tests) :
        Ncoinc += SpikeTrainComparator.Md_dotProduct_Kistler(test_spike_trains[i], test_spike_trains[j], KistlerDotProduct_args, experiment.dt)
R_X = 2.*Ncoinc/(sum_of_norms*(number_of_tests-1))
print 'Intrinsic reliability = %f' %R_X


#################################################################################################
# STEP 3: FIT GIF-Ca MODEL
#################################################################################################

# Create a new object GIF
GIF_Ca_fit = GIF_Ca(sampling_time)

# Define parameters
GIF_Ca_fit.Tref = 6.0

GIF_Ca_fit.eta = Filter_Rect_LogSpaced()
GIF_Ca_fit.eta.setMetaParameters(length=2000.0, binsize_lb=0.5, binsize_ub=500.0, slope=10.0)
#5HT GIF_Ca_fit.eta.setMetaParameters(length=5000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

GIF_Ca_fit.gamma = Filter_Rect_LogSpaced()
GIF_Ca_fit.gamma.setMetaParameters(length=2000.0, binsize_lb=2.0, binsize_ub=500.0, slope=5.0)
#5HT GIF_Ca_fit.gamma.setMetaParameters(length=5000.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

# Define the ROI of the training set to be used for the fit
#experiment.trainingset_traces[0].setROI([[1000,times[length_training-1]-1000]])
for tr in experiment.trainingset_traces :
    tr.setROI([[2000., sampling_time*(len(voltage_trace)-1)-2000.]])
    '''time_start = time.clock()
    selection = tr.getROI_FarFromSpikes(5., 6.)
    selection_l = len(selection)

    # Build X matrix for linear regression
    X = np.zeros((selection_l, 5))

    # Fill first two columns of X matrix
    X[:, 0] = tr.V[selection]
    X[:, 1] = tr.I[selection]
    X[:, 2] = np.ones(selection_l)

    m = GIF_Ca_fit.simulate_m(tr.V)
    print 'm shape '
    print m.shape
    m = m[selection]
    h = GIF_Ca_fit.simulate_h(tr.V)
    h = h[selection]
    #m = m.reshape((selection_l, 1))
    #h = h.reshape((selection_l, 1))
    print 'Done computing m and h.'

    #X_Ca = np.zeros((selection_l, 2))
    #X_Ca[:, 0] = m * h * X[:, 0]
    #X_Ca[:, 1] = m * h
    X[:, 3] = m * h * X[:, 0]
    X[:, 4] = m * h

    #X = np.concatenate((X, X_Ca), axis=1)

    time_elapsed = time.clock() - time_start
    print 'Time to simulate = %f' %time_elapsed'''

# To visualize the training set and the ROI call again
#experiment.plotTrainingSet()


# Perform the fit
is_E_Ca_fixed = False
shifts = np.arange(0., 40., 10.)
Vm = np.arange(-100., 40., 5)
window_current_no_shift = GIF_Ca.m_inf(Vm)**4*GIF_Ca.h_inf(Vm)
for shift in shifts:
    GIF_Ca_fit.shift = shift
    plt.plot(Vm, window_current_no_shift)
    plt.plot(Vm, GIF_Ca.m_inf(Vm-shift)**4*GIF_Ca.h_inf(Vm-shift))
    plt.xlabel('Voltage [mV]')
    plt.show()
    (var_explained_dV, var_explained_V) = GIF_Ca_fit.fit(experiment, DT_beforeSpike=5.0, is_E_Ca_fixed=is_E_Ca_fixed)

temp = GIF_Ca_fit.fit(experiment, DT_beforeSpike=5.0, is_E_Ca_fixed=is_E_Ca_fixed)

# Plot the model parameters
GIF_Ca_fit.plotParameters()

# Save the model
GIF_TYPE = 'GIF_Ca_'
#GIF_Ca_fit.save(PATH_RESULTS + GIF_TYPE + CELL_NAME + SPECIFICATION + ADDITIONAL_SPECIFIER + '.pck')


#################################################################################################
# FIT standard GIF
#################################################################################################

# Create a new object GIF
GIF_fit = GIF(sampling_time)

# Define parameters
GIF_fit.Tref = 6.0

GIF_fit.eta = Filter_Rect_LogSpaced()
GIF_fit.eta.setMetaParameters(length=2000.0, binsize_lb=0.5, binsize_ub=500.0, slope=10.0)
#5HT GIF_fit.eta.setMetaParameters(length=5000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

GIF_fit.gamma = Filter_Rect_LogSpaced()
GIF_fit.gamma.setMetaParameters(length=2000.0, binsize_lb=2.0, binsize_ub=500.0, slope=5.0)
#5HT GIF_fit.gamma.setMetaParameters(length=5000.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

GIF_fit.fit(experiment, DT_beforeSpike=5.0)

# Plot the model parameters
GIF_fit.plotParameters()



'''
#################################################################################################
# STEP 3B: FIT iGIF_NP (Mensi et al. 2016 with current-based spike-triggered adaptation)
#################################################################################################

# Note that in the iGIF_NP model introduced in Mensi et al. 2016, the adaptation current is
# conductance-based (i.e., eta is a spike-triggered conductance).

# Define metaparameters used during the fit

theta_inf_nbbins  = 10                      # Number of rect functions used to define the nonlinear coupling between
                                            # membrane potential and firing threshold (note that the positioning of
                                            # the rect function
                                            # is computed automatically based on the voltage distribution).

theta_tau_all     = np.linspace(15.,25., 10)  # tau_theta is the timescale of the threshold-voltage coupling


# Create the new model used for the fit

iGIF_NP_fit = iGIF_NP(experiment.dt)

iGIF_NP_fit.Tref  = GIF_fit.Tref                 # use the same absolute refractory period as in GIF_fit
iGIF_NP_fit.eta   = copy.deepcopy(GIF_fit.eta)   # use the same basis function as in GIF_fit for eta (filer coeff will be refitted)
iGIF_NP_fit.gamma = copy.deepcopy(GIF_fit.gamma) # use the same basis function as in GIF_fit for gamma (filer coeff will be refitted)


# Perform the fit

iGIF_NP_fit.fit(experiment, theta_inf_nbbins=theta_inf_nbbins, theta_tau_all=theta_tau_all, DT_beforeSpike=5.0)


# Plot optimal parameters

iGIF_NP_fit.plotParameters()

# Save the model
GIF_TYPE = 'iGIF_NP_'
iGIF_NP_fit.save(PATH_RESULTS + GIF_TYPE + CELL_NAME + SPECIFICATION + ADDITIONAL_SPECIFIER + '.pck')


###################################################################################################
# STEP 3C: FIT iGIF_Na (Mensi et al. 2016 with current-based spike-triggered adaptation)
###################################################################################################
# Note that in the iGIF_Na model introduced in Mensi et al. 2016, the adaptation current is
# conductance-based (i.e., eta is a spike-triggered conductance).

# Define metaparameters used during the fit
ki_bounds_Na      = [1., 5.]                  # mV, interval over which the optimal parameter k_i (ie, Na inactivation slope) is searched
ki_BRUTEFORCE_RESOLUTION = 5                    # number of parameters k_i considered in the fit (lin-spaced over ki_bounds_Na)

Vi_bounds_Na      = [-55., -45.]              # mV, interval over which the optimal parameter V_i (ie, Na half inactivation voltage) is searched
Vi_BRUTEFORCE_RESOLUTION = 10                    # number of parameters V_i considered in the fit (lin-spaced over Vi_bounds_Na)


# Create new iGIF_Na model that will be used for the fit

iGIF_Na_fit       = iGIF_Na(experiment.dt)
iGIF_Na_fit.Tref  = GIF_fit.Tref                 # use the same absolute refractory period as in GIF_fit
iGIF_Na_fit.eta   = copy.deepcopy(GIF_fit.eta)   # use the same basis function as in GIF_fit for eta (filer coeff will be refitted)
iGIF_Na_fit.gamma = copy.deepcopy(GIF_fit.gamma) # use the same basis function as in GIF_fit for gamma (filer coeff will be refitted)

# Compute set of values that will be tested for ki and Vi (these parameters are extracted using a brute force approach, as described in Mensi et al. 2016 PLOS Comp. Biol. 2016)

ki_all = np.linspace(ki_bounds_Na[0], ki_bounds_Na[-1], ki_BRUTEFORCE_RESOLUTION)
Vi_all = np.linspace(Vi_bounds_Na[0], Vi_bounds_Na[-1], Vi_BRUTEFORCE_RESOLUTION)

# Fit the model

# Note that the second parameter provided by the input is the timescale theta_tau, ie the timescale of the threshold-votlage coupling.
# This parameter is not extracted form the data, but is assumed)
iGIF_Na_fit.fit(experiment, iGIF_NP_fit.theta_tau, ki_all, Vi_all, DT_beforeSpike=5.0, do_plot=True)

# Plot optimal parameters
iGIF_Na_fit.printParameters()

# Save the model
GIF_TYPE = 'iGIF_Na_'
iGIF_Na_fit.save(PATH_RESULTS + GIF_TYPE + CELL_NAME + SPECIFICATION + ADDITIONAL_SPECIFIER + '.pck')

###################################################################################################
# STEP 4: EVALUATE MODEL PERFORMANCES ON THE TEST SET DATA
###################################################################################################
'''
models = [GIF_fit, GIF_Ca_fit]
labels = ['GIF', 'GIF-Ca']

#models = [GIF_fit, iGIF_NP_fit]
#labels = ['GIF', 'iGIF_NP']
#output_file_md = open(PATH_RESULTS + CELL_NAME + ADDITIONAL_SPECIFIER  + SPECIFICATION + '_Md.dat','w')
#output_file_md.write('#' + CELL_NAME + ADDITIONAL_SPECIFIER + SPECIFICATION + '\n')
#output_file_md.write('#Cell name\tSig\tIntrinsic reliability\tMd*(GIF)\tMd*(iGIF_NP)\tMd*(iGIF_Na)\n')

Md_all = []

for i in np.arange(len(models)) :

    model = models[i]

    # predict spike times in test set
    prediction = experiment.predictSpikes(model, nb_rep=500)

    print "\n Model: ", labels[i]

    # Compute epsilon_V
    epsilon_V = 0.
    local_counter = 0.
    for tr in experiment.testset_traces:
        SSE = 0.
        VAR = 0.
        # tr.detectSpikesWithDerivative(threshold=10)
        r = model.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
        indices_tmp = tr.getROI_FarFromSpikes(5., model.Tref)

        SSE += sum((r[1][indices_tmp] - tr.V[indices_tmp]) ** 2)
        VAR += len(indices_tmp) * np.var(tr.V[indices_tmp])
        epsilon_V += 1.0 - SSE / VAR
        local_counter += 1
    epsilon_V = epsilon_V / local_counter
    print 'epsilon_V = %f' % epsilon_V

    # Compute Md*
    Md = prediction.computeMD_Kistler(8.0, model.dt*2.)
    Md_all.append(Md)
    fname = PATH_RESULTS + CELL_NAME + ADDITIONAL_SPECIFIER  + SPECIFICATION +'_'+ labels[i] + '_ExpVsModel.png'
    kernelForPSTH = 100.0
    Percent_of_explained_var = prediction.plotRaster(fname, delta=kernelForPSTH)
    print "Explained var by %s = %0.2f" % (labels[i], Percent_of_explained_var)
#output_file_md.write(CELL_NAME[3:6] + '\t' + SPECIFICATION[4:] + '\t' + str(R_X) + '\t' + str(Md_all[0]) + '\t' + str(Md_all[1]) + '\t' + str(Md_all[2]) + '\n')
#output_file_md.close()





###################################################################################################
# STEP 5: COMPARE model parameters
###################################################################################################
GIF_Ca_fit.printParameters()
GIF_fit.printParameters()

#iGIF.compareModels([GIF_Ca_fit, GIF_fit], labels=['GIF_Ca', 'GIF'])


#################################################################################################
#    Compare training and test for GIF
#################################################################################################
V_training = experiment.trainingset_traces[0].V
I_training = experiment.trainingset_traces[0].I
(time, V, eta_sum, V_t, S) = GIF_Ca_fit.simulate(I_training, V_training[0])
(time, V_GIF, eta_sum, V_t, S) = GIF_fit.simulate(I_training, V_training[0])
fig = plt.figure(figsize=(14,5), facecolor='white')
plt.subplot(2,1,1)
plt.plot(time/1000, V,'-b', lw=0.5, label='GIF-Ca')
plt.plot(time/1000, V_GIF,'--r', lw=0.5, label='GIF')
plt.plot(time/1000, V_training,'black', lw=0.5, label='Data')
plt.xlim(17,20)
plt.ylim(-75,0)
plt.ylabel('Voltage [mV]')
plt.title('Training')

V_test = experiment.testset_traces[0].V
I_test = experiment.testset_traces[0].I
(time, V, eta_sum, V_t, S) = GIF_Ca_fit.simulate(I_test, V_test[0])
(time, V_GIF, eta_sum, V_t, S) = GIF_fit.simulate(I_test, V_test[0])
plt.subplot(2,1,2)
plt.plot(time/1000, V,'-b', lw=0.5, label='GIF-Ca')
plt.plot(time/1000, V_GIF,'--r', lw=0.5, label='GIF')
plt.plot(time/1000, V_test,'black', lw=0.5, label='Data')
plt.xlim(4,7)
plt.ylim(-80,0)
plt.xlabel('Times [s]')
plt.ylabel('Voltage [mV]')
plt.title('Test')
plt.legend()
if not is_E_Ca_fixed:
    plt.savefig(PATH_RESULTS + CELL_NAME + ADDITIONAL_SPECIFIER  + SPECIFICATION + '_E_Cafree.png', format='png')
else:
    plt.savefig(PATH_RESULTS + CELL_NAME + ADDITIONAL_SPECIFIER  + SPECIFICATION + '_E_Cafixed.png', format='png')
plt.close(fig)
#plt.show()


# Figure comparing V_model, V_data and I during training with forced spikes
(time, V, eta_sum) = GIF_Ca_fit.simulateDeterministic_forceSpikes(I_training, V_training[0], experiment.trainingset_traces[0].getSpikeTimes())
(time, V_GIF, eta_sum) = GIF_fit.simulateDeterministic_forceSpikes(I_training, V_training[0], experiment.trainingset_traces[0].getSpikeTimes())
fig = plt.figure(figsize=(14,5), facecolor='white')
plt.subplot(2,1,1)
plt.plot(time/1000, I_training,'-b', lw=0.5, label='$I$')
#plt.plot(time/1000, I_Ca,'--r', lw=0.5, label='$I_\mathrm{Ca}$')
plt.xlim(17,20)
plt.ylabel('Current [nA]')
plt.title('Training')
plt.subplot(2,1,2)
plt.plot(time/1000, V,'-b', lw=0.5, label='GIF-Ca')
plt.plot(time/1000, V_GIF,'--r', lw=0.5, label='GIF')
plt.plot(time/1000, V_training,'black', lw=0.5, label='Data')
plt.xlim(17,20)
plt.ylim(-75,0)
plt.ylabel('Time [s]')
plt.ylabel('Voltage [mV]')

if not is_E_Ca_fixed:
    plt.savefig(PATH_RESULTS + CELL_NAME + ADDITIONAL_SPECIFIER  + SPECIFICATION + 'V_I_E_Cafree.png', format='png')
else:
    plt.savefig(PATH_RESULTS + CELL_NAME + ADDITIONAL_SPECIFIER  + SPECIFICATION + 'V_I_E_Cafixed.png', format='png')
plt.close(fig)
#plt.show()
