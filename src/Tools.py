import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import leastsq
import weave

import sys
import scipy.io as sio
import neo

#import GIF
#from Experiment import *
#from AEC_Badel import *

###########################################################
# Remove axis
###########################################################

def removeAxis(ax, which_ax=['top', 'right']):

    for loc, spine in ax.spines.iteritems():
        if loc in which_ax:
            spine.set_color('none') # don't draw spine

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


###########################################################
# Reprint
###########################################################
def reprint(str):
    sys.stdout.write('%s\r' % (str))
    sys.stdout.flush()


###########################################################
# Generate Ornstein-Uhlenbeck process
###########################################################

def generateOUprocess(T=10000.0, tau=3.0, mu=0.0, sigma=1.0, dt=0.1):

    """
    Generate an Ornstein-Uhlenbeck (stationnary) process with:
    - mean mu
    - intensity parameter sigma
    - temporal correlation tau (ms)
    The duration of the signal is specified by the input parameter T (in ms).
    The process is generated in discrete time with temporal resolution dt (in ms)
    """

    T_ind = int(T/dt)

    white_noise = np.random.randn(T_ind)
    white_noise = white_noise.astype("double")

    OU_process = np.zeros(T_ind)
    OU_process = OU_process.astype("double")

    '''OU_k1 = dt / tau
    OU_k2 = np.sqrt(dt / tau)
    for t in xrange(T_ind-1) :
        OU_process[t + 1] = OU_process[t] + (mu - OU_process[t]) * OU_k1 + sigma * OU_k2 * white_noise[t]
    '''

    code =  """

            #include <math.h>

            int cT_ind    = int(T_ind);
            float cdt     = float(dt);
            float ctau    = float(tau);
            float cmu     = float(mu);
            float csigma  = float(sigma);

            float OU_k1 = cdt / ctau ;
            float OU_k2 = sqrt(cdt/ctau) ;

            for (int t=0; t < cT_ind-1; t++) {
                OU_process[t+1] = OU_process[t] + (cmu - OU_process[t])*OU_k1 +  csigma*OU_k2*white_noise[t] ;
            }

            """

    vars = ['T_ind', 'dt', 'tau', 'sigma','mu', 'OU_process', 'white_noise']
    v = weave.inline(code,vars)

    return OU_process


def generateOUprocess_sinSigma(f=1.0, T=10000.0, tau=3.0, mu=0.0, sigma=1.0, delta_sigma=0.5, dt=0.1):

    """
    Generate an Ornstein-Uhlenbeck process with time dependent standard deviation:
    - mean mu
    - sigma(t) = sigma*(1+delta_sigma*sin(2pift)), f in Hz
    - temporal correlation tau (ms)
    The duration of the signal is specified by the input parameter T (in ms).
    The process is generated in discrete time with temporal resolution dt (in ms)
    """

    OU_process = generateOUprocess(T=T, tau=tau, mu=0.0, sigma=1.0, dt=dt)
    t          = np.arange(len(OU_process))*dt

    sin_sigma = sigma*(1+delta_sigma*np.sin(2*np.pi*f*t*10**-3))

    I = OU_process*sin_sigma + mu

    return I


def generateOUprocess_sinMean(f=1.0, T=10000.0, tau=3.0, mu=0.2, delta_mu=0.5, sigma=1.0, dt=0.1):

    """
    Generate an Ornstein-Uhlenbeck process with time dependent mean:
    - sigma
    - mu(t) = mu*(1+delta_mu*sin(2pift)), f in Hz
    - temporal correlation tau (ms)
    The duration of the signal is specified by the input parameter T (in ms).
    The process is generated in discrete time with temporal resolution dt (in ms)
    """

    OU_process = generateOUprocess(T=T, tau=tau, mu=0.0, sigma=sigma, dt=dt)
    t          = np.arange(len(OU_process))*dt

    sin_mu = mu*(1+delta_mu*np.sin(2*np.pi*f*t*10**-3))

    I = OU_process + sin_mu

    return I


###########################################################
# Functin to convert spike times in spike indices
###########################################################
def timeToIndex(x_t, dt):

    x_t = np.array(x_t)
    x_i = np.array( [ int(np.round(s/dt)) for s in x_t ] )
    x_i = x_i.astype('int')

    return x_i


###########################################################
# Functions to perform exponential fit
###########################################################

def multiExpEval(x, bs, taus):

    result = np.zeros(len(x))
    L = len(bs)

    for i in range(L) :
        result = result + bs[i] *np.exp(-x/taus[i])

    return result


def multiExpResiduals(p, x, y, d):
    bs = p[0:d]
    taus = p[d:2*d]

    return (y - multiExpEval(x, bs, taus))



def fitMultiExpResiduals(bs, taus, x, y) :
    x = np.array(x)
    y = np.array(y)
    d = len(bs)
    p0 = np.concatenate((bs,taus))
    plsq = leastsq(multiExpResiduals, p0, args=(x,y,d), maxfev=100000,ftol=0.00000001)
    p_opt = plsq[0]
    bs_opt = p_opt[0:d]
    taus_opt = p_opt[d:2*d]

    fitted_data = multiExpEval(x, bs_opt, taus_opt)

    ind = np.argsort(taus_opt)

    taus_opt = taus_opt[ind]
    bs_opt = bs_opt[ind]

    return (bs_opt, taus_opt, fitted_data)


###########################################################
# Get indices far from spikes
###########################################################

def getIndicesFarFromSpikes(T, spikes_i, dt_before, dt_after, initial_cutoff, dt) :

    T_i = int(T/dt)
    flag = np.zeros(T_i)
    flag[:int(initial_cutoff/dt)] = 1
    flag[-1] = 1

    dt_before_i = int(dt_before/dt)
    dt_after_i = int(dt_after/dt)

    for s in spikes_i :
        flag[ max(s-dt_before_i,0) : min(s+dt_after_i, T_i) ] = 1

    selection = np.where(flag==0)[0]

    return selection


def getIndicesDuringSpikes(T, spikes_i, dt_after, initial_cutoff, dt) :

    T_i = int(T/dt)
    flag = np.zeros(T_i)
    flag[:int(initial_cutoff/dt)] = 1
    flag[-1] = 1

    dt_after_i = int(dt_after/dt)

    for s in spikes_i :
        flag[ max(s,0) : min(s+dt_after_i, T_i) ] = 1

    selection = np.where(flag>0.1)[0]

    return selection


###########################################################
# Load AEC data
###########################################################
def load_AEC_data(filename):
    if filename.find('.mat') > 0:
        mat_contents = sio.loadmat(filename)
        analogSignalsAEC = mat_contents['analogSignals']
        timesAEC = mat_contents['times']
        timesAEC = timesAEC.reshape(timesAEC.size)
        lengthAEC = timesAEC[-1]
        voltage_traceAEC = analogSignalsAEC[0, 0, :]
        current_traceAEC = analogSignalsAEC[1, 0, :] - 5.  # Correction for exp bias
        maxT = int((lengthAEC - 0.5) / (timesAEC[1] - timesAEC[0]))
        minT = int(0.5 / (timesAEC[1] - timesAEC[0]))
        current_traceAEC = current_traceAEC[minT:maxT]
        voltage_traceAEC = voltage_traceAEC[minT:maxT]
        timesAEC = timesAEC[minT:maxT] * 10. ** 3
        sampling_time = timesAEC[1] - timesAEC[0]
    elif filename.find('.abf') > 0:
        r = neo.io.AxonIO(filename=filename)
        bl = r.read_block()
        voltage_traceAEC = bl.segments[0].analogsignals[0].magnitude
        current_traceAEC = bl.segments[0].analogsignals[1].magnitude - 5.
        timesAEC = bl.segments[0].analogsignals[0].times.rescale('ms').magnitude
        lengthAEC = timesAEC[-1]
        sampling_time = timesAEC[1] - timesAEC[0]
        maxT = int((lengthAEC - 0.5) / sampling_time)
        minT = int(0.5 / sampling_time)
        current_traceAEC = current_traceAEC[minT:maxT]
        voltage_traceAEC = voltage_traceAEC[minT:maxT]
    return (sampling_time, voltage_traceAEC, current_traceAEC)


def load_training_data(filename):
    if filename.find('.mat') > 0:
        mat_contents = sio.loadmat(filename)
        analogSignals = mat_contents['analogSignals']
        voltage_trace = analogSignals[0,0,:]
        current_trace = analogSignals[1,0,:] - 5.   #offset due to uncalibrated current
        times = mat_contents['times'];
        times = times.reshape(times.size)
        times = times*10.**3                        #convert to ms
        sampling_time = times[1] - times[0]
    elif filename.find('.abf') > 0:
        r = neo.io.AxonIO(filename=filename)
        bl = r.read_block()
        voltage_trace = bl.segments[0].analogsignals[0].magnitude
        current_trace = bl.segments[0].analogsignals[1].magnitude - 5.
        times = bl.segments[0].analogsignals[0].times.rescale('ms').magnitude
        sampling_time = times[1] - times[0]
    return (sampling_time, voltage_trace, current_trace, times)


##########################################################
# Plotting training trace with forced spikes
##########################################################
'''
def plot_training_forced(filename_AEC, filename_training, filename_model):
    """
    :param filename_training: training data file name  (string)
    :param filename_model: name of the model (ending in .pck)
    :return: none
    """
    # Load AEC data
    (sampling_timeAEC, voltage_traceAEC, current_traceAEC) = load_AEC_data(filename_AEC)

    experiment = Experiment('Experiment 1', sampling_timeAEC)
    experiment.setAECTrace(voltage_traceAEC, 10. ** -3, current_traceAEC, 10. ** -12,
                           len(voltage_traceAEC) * sampling_timeAEC, FILETYPE='Array')

    # Load preprocessed training data set
    (sampling_time, voltage_trace, current_trace) = load_training_data(filename_training)
    experiment.addTrainingSetTrace(voltage_trace, 10 ** -3, current_trace, 10 ** -12,
                                   len(voltage_trace) * sampling_time, FILETYPE='Array')

    # Compensate traces
    myAEC = AEC_Badel(experiment.dt)

    # Define metaparametres
    myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=experiment.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
    myAEC.p_expFitRange = [3.0, 150.0]
    myAEC.p_nbRep = 15

    # Assign myAEC to experiment and compensate the voltage recordings
    experiment.setAEC(myAEC)
    experiment.performAEC()

    # Load model
    model = GIF.load(filename_model)

    # Simulate model
    (time, V_est, eta_sum_est) = model.simulateDeterministic_forceSpikes(current_trace, voltage_trace[0], experiment.trainingset_traces[0].spks)

    #Plot
    plt.plot(time, V_est, 'r-', label='V_$\mathrm{est}$', lw=0.5)
    plt.plot(time, voltage_trace, 'k-', label='V__$\mathrm{training}$', lw=0.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    plt.xlim(4000, 4500)
    plt.savefig(filename_model[:-4] + '_training_forced.png')
    plt.close()
'''
