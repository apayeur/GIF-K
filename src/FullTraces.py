import matplotlib.pyplot as plt
import scipy.io as sio

def disp(filename):
	'''
	Display raw experimental traces (current versus time and voltage vs time)
	'''
	mat_contents = sio.loadmat(filename)
	analogSignals = mat_contents['analogSignals']
	times = mat_contents['times']
	times = times.reshape(times.size)
	times = times*10**3
	voltage_trace = analogSignals[0,0,:]
	current_trace = analogSignals[0,1,:]
	plt.subplot(211)
	plt.plot(times, current_trace)
	plt.ylabel('Input current [pA]')
	
	plt.subplot(212)
	plt.plot(times, voltage_trace)
	plt.xlabel('Time [ms]')
	plt.ylabel('Membrane voltage [mV]')
	plt.show()