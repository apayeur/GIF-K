import matplotlib.pyplot as plt
import cPickle as pkl

from SpikeTrainComparator import *
from SpikingModel import *
from Trace import *


class Experiment :
    
    """
    Objects of this class contains the experimental data and an "AEC object" that takes care of Active Electrode Compensation
    """
    
    
    def __init__(self, name, dt):
        
        print "Create new Experiment"

        self.name               = name          # Experiment name
        
        self.dt                 = dt            # Sampling (all traces in same experiment must have same sampling)  
        
        self.AEC_trace          = 0             # Trace object containing voltage and input current used for AEC  
        
        self.trainingset_traces = []            # Traces for training
        
        self.testset_traces     = []            # Traces of test set (typically multiple experiments conducted with frozen noise)
        
        self.AEC                = 0             # Object that knows how to perform AEC 

        self.spikeDetection_threshold    = 0.0  # mV, voltage threshold used to detect spikes
        
        self.spikeDetection_ref          = 3.0  # ms, absolute refractory period used for spike detection to avoid double counting of spikes



    ############################################################################################
    # FUNCTIONS TO ADD TRACES TO THE EXPERIMENT
    ############################################################################################   
    
    def setAECTrace(self, V, V_units, I, I_units, T, FILETYPE='Igor'):
    
        print "Set AEC trace..."
        trace_tmp = Trace( V, V_units, I, I_units, T, self.dt, FILETYPE=FILETYPE)
        self.AEC_trace = trace_tmp

        return trace_tmp
    
    
    def addTrainingSetTrace(self, V, V_units, I, I_units, T, FILETYPE='Igor'):
    
        print "Add Training Set trace..."
        trace_tmp = Trace( V, V_units, I, I_units, T, self.dt, FILETYPE=FILETYPE)
        self.trainingset_traces.append( trace_tmp )

        return trace_tmp


    def addTestSetTrace(self, V, V_units, I, I_units, T, FILETYPE='Igor'):
    
        print "Add Test Set trace..."
        trace_tmp = Trace( V, V_units, I, I_units, T, self.dt, FILETYPE=FILETYPE)    
        self.testset_traces.append( trace_tmp )

        return trace_tmp
    
    

    ############################################################################################
    # FUNCTIONS ASSOCIATED WITH ACTIVE ELECTRODE COMPENSATION
    ############################################################################################    
    def setAEC(self, AEC):
        
        self.AEC = AEC


    def getAEC(self):
        
        return self.AEC    
             
             
    def performAEC(self):

        self.AEC.performAEC(self)
    
    
    ############################################################################################
    # FUNCTIONS FOR SAVING AND LOADING AN EXPERIMENT
    ############################################################################################
    def save(self, path):
        
        filename = path + "/Experiment_" + self.name + '.pkl'
        
        print "Saving: " + filename + "..."        
        f = open(filename,'w')
        pkl.dump(self, f)
        print "Done!"
        
        
    @classmethod
    def load(cls, filename):
        
        print "Load experiment: " + filename + "..."        
      
        f = open(filename,'r')
        expr = pkl.load(f)
    
        print "Done!" 
           
        return expr      
      
      
    ############################################################################################
    # EVALUATE PERFORMANCES OF A MODEL
    ############################################################################################         
    def predictSpikes(self, spiking_model, nb_rep=500):

        # Collect spike times in test set

        all_spks_times_testset = []

        for tr in self.testset_traces:
            
            if tr.useTrace :
                
                spks_times = tr.getSpikeTimes()
                all_spks_times_testset.append(spks_times)
    
    
        # Predict spike times using model
        T_test = self.testset_traces[0].T
        I_test = self.testset_traces[0].I
        
        all_spks_times_prediction = []
        
        print "Predict spike times..."
        
        for rep in np.arange(nb_rep) :
            print "Progress: %2.1f %% \r" % (100*(rep+1)/nb_rep),
            spks_times = spiking_model.simulateSpikingResponse(I_test, self.dt)
            all_spks_times_prediction.append(spks_times)
        
        print
                
        prediction = SpikeTrainComparator(T_test, all_spks_times_testset, all_spks_times_prediction)
        
        return prediction
        

        
    ############################################################################################
    # AUXILIARY FUNCTIONS
    ############################################################################################            
    def detectSpikes_python(self, threshold=0.0, ref=3.0):

        # implement here a function that detects all the spkes in all the traces...
        # parameters of spike detection should be set in this class and not in trace

        print "Detect spikes!"
                
        self.spikeDetection_threshold = threshold   
        self.spikeDetection_ref = ref         

        if self.AEC_trace != 0 :
            self.AEC_trace.detectSpikes_python(self.spikeDetection_threshold, self.spikeDetection_ref)
        
        for tr in self.trainingset_traces :
            tr.detectSpikes_python(self.spikeDetection_threshold, self.spikeDetection_ref)           
            
        for tr in self.testset_traces :
            tr.detectSpikes_python(self.spikeDetection_threshold, self.spikeDetection_ref)         
        
        print "Done!"
        
        
    def detectSpikes(self, threshold=0.0, ref=3.0):

        # implement here a function that detects all the spkes in all the traces...
        # parameters of spike detection should be set in this class and not in trace

        print "Detect spikes!"
                
        self.spikeDetection_threshold = threshold   
        self.spikeDetection_ref = ref         

        if self.AEC_trace != 0 :
            self.AEC_trace.detectSpikes(self.spikeDetection_threshold, self.spikeDetection_ref)
        
        for tr in self.trainingset_traces :
            tr.detectSpikes(self.spikeDetection_threshold, self.spikeDetection_ref)           
            
        for tr in self.testset_traces :
            tr.detectSpikes(self.spikeDetection_threshold, self.spikeDetection_ref)         
        
        print "Done!"
    
    
    def getTrainingSetNb(self):
        
        return len(self.trainingset_traces) 
      


      
    ############################################################################################
    # FUNCTIONS FOR PLOTTING
    ############################################################################################
    def plotTrainingSet(self):
        
        plt.figure(figsize=(12,8), facecolor='white')
        
        cnt = 0
        
        for tr in self.trainingset_traces :
            
            # Plot input current
            plt.subplot(2*self.getTrainingSetNb(),1,cnt*2+1)
            plt.plot(tr.getTime(), tr.I, 'gray')

            # Plot ROI
            ROI_vector = -10.0*np.ones(int(tr.T/tr.dt)) 
            if tr.useTrace :
                ROI_vector[tr.getROI()] = 10.0
            
            plt.fill_between(tr.getTime(), ROI_vector, 10.0, color='0.2')
            
            plt.ylim([min(tr.I)-0.1, max(tr.I)+0.1])
            plt.ylabel("I (nA)")
            plt.xticks([])
            
            # Plot membrane potential    
            plt.subplot(2*self.getTrainingSetNb(),1,cnt*2+2)  
            plt.plot(tr.getTime(), tr.V_rec, 'black')    
            
            if tr.AEC_flag :
                plt.plot(tr.getTime(), tr.V, 'blue')    
                
                
            if tr.spks_flag :
                plt.plot(tr.getSpikeTimes(), np.zeros(tr.getSpikeNb()), '.', color='red')
            
            # Plot ROI
            ROI_vector = -100.0*np.ones(int(tr.T/tr.dt)) 
            if tr.useTrace :
                ROI_vector[tr.getROI()] = 100.0
            
            plt.fill_between(tr.getTime(), ROI_vector, 100.0, color='0.2')
            
            plt.ylim([min(tr.V)-5.0, max(tr.V)+5.0])
            plt.ylabel("Voltage (mV)")   
                  
            cnt +=1
        
        plt.xlabel("Time (ms)")
        
        plt.subplot(2*self.getTrainingSetNb(),1,1)
        plt.title('Experiment ' + self.name + " - Training Set (dark region not selected)")
        plt.subplots_adjust(left=0.10, bottom=0.07, right=0.95, top=0.92, wspace=0.25, hspace=0.25)

        plt.show()

        
    def plotTestSet(self):
        
        plt.figure(figsize=(12,6), facecolor='white')
        
        # Plot  test set currents 
        plt.subplot(3,1,1)
       
        for tr in self.testset_traces :         
            plt.plot(tr.getTime(), tr.I, 'gray')
        plt.ylabel("I (nA)")
        plt.title('Experiment ' + self.name + " - Test Set")
        # Plot  test set voltage        
        plt.subplot(3,1,2)
        for tr in self.testset_traces :          
            plt.plot(tr.getTime(), tr.V, 'black')
        plt.ylabel("Voltage (mV)")

        # Plot test set raster
        plt.subplot(3,1,3)
        
        cnt = 0
        for tr in self.testset_traces :
            cnt += 1      
            if tr.spks_flag :
                plt.plot(tr.getSpikeTimes(), cnt*np.ones(tr.getSpikeNb()), '|', color='black', ms=5, mew=2)
        
        plt.yticks([])
        plt.ylim([0, cnt+1])
        plt.xlabel("Time (ms)")
        
        plt.subplots_adjust(left=0.10, bottom=0.07, right=0.95, top=0.92, wspace=0.25, hspace=0.25)

        plt.show()