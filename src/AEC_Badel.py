import numpy as np
import matplotlib.pyplot as plt
import copy

from AEC import *
from Experiment import *
from Filter_Rect_LinSpaced import *
from Filter_Rect_LogSpaced_AEC import *

from numpy.linalg import *
from random import sample
from time import time


class AEC_Badel(AEC) :
        
        
    def __init__(self, dt):
        
        # Define variables for optimal linear filter K_opt   
        self.K_opt      = Filter_Rect_LogSpaced_AEC(length=150.0, binsize_lb=dt, binsize_ub=5.0, slope=10.0, clamp_period=0.5)
        self.K_opt_all  = []                # List of K_opt, store bootstrap repetitions                         

        # Define variables for electrode filter    
        self.K_e        = Filter_Rect_LinSpaced()
        self.K_e_all    = []                # List of K_e, store bootstrap repetitions 

        # Meta parameters used in AEC = Step 1 (compute optimal linear filter)
        self.p_nbRep       = 15             # nb of times the filer is estimated (each time resampling from available data)
        self.p_pctPoints   = 0.8            # between 0 and 1, fraction of datapoints in subthreshold recording used in bootstrap at each repetition
   
        # Meta parameters used in AEC = Step 2 (estimation of Ke given Kopt)
        self.p_Ke_l        = 7.0            # ms, length of the electrode filter Ke
        self.p_b0          = [15.0]         # MOhm/ms, initial condition for exp fit on the tail of Kopt (amplitude)
        self.p_tau0        = [30.0]         # ms, initial condition for exp fit on the tail of Kopt (timescale)
        self.p_expFitRange = [3.0, 50.0]    # ms, range were to perform exp fit on the tail of K_opt
              
        self.p_derivative_flag = False
        
    ##############################################################################################    
    # ABSTRACT METHODS FROM AEC THAT HAVE TO BE IMPLEMENTED
    ##############################################################################################
    
    def performAEC(self, experiment):

        print "\nPERFORM ACTIVE ELECTRODE COMPENSATION (Badel method)..."

        # Estimate electrode filter
        self.computeElectrodeFilter(experiment)

        # Estimate electrode filter        
        self.compensateAllTraces(experiment)
        

    ##############################################################################################    
    # ESTIMATE ELECTRODE FILTER
    # This function implements two steps:
    # Step 1: compute optimal linear filter
    # Step 2: compute electrode filter
    ##############################################################################################
    def computeElectrodeFilter(self, expr) :
    
        """
        Extract the optimal linter filter between I and V_rec.
        The regression is performed using the tempral derivative of the signals (see Badel et al 2008).
        To speed up, the optimal linear filter is expanded in rectangular basis functions.
        """
        
        print "\nEstimate electrode properties..."
        
        dt = expr.dt       
        
        # estimate optimal linear filter on I_dot - V_dot
        if self.p_derivative_flag :
        
            # Compute temporal derivative of the signal
            V_dot = np.diff(expr.AEC_trace.V_rec)/dt
            I_dot = np.diff(expr.AEC_trace.I)/dt
            
        # estimate optimal linear filter on I - V
        else :
            
            # Just remove mean from signals (do not use derivative) 
                                       
            V_dot = expr.AEC_trace.V_rec - np.mean(expr.AEC_trace.V_rec) 
            I_dot = expr.AEC_trace.I - np.mean(expr.AEC_trace.I) 
            
        # Get ROI indices and remove initial part
        ROI_selection = expr.AEC_trace.getROI_cutInitialSegments(self.K_opt.getLength())
        ROI_selection = ROI_selection[:-1]
        ROI_selection_l = len(ROI_selection)


        # Build full X matrix for linear regression
        
        X = self.K_opt.convolution_ContinuousSignal_basisfunctions(I_dot, dt)
        nbPoints = int(self.p_pctPoints*ROI_selection_l)
        
        # Estimate electrode filter on multiple repetitions by bootstrap      
        for rep in np.arange(self.p_nbRep) :
              
            ############################################
            # ESTIMATE OPTIMAL LINEAR FILETR K_opt
            ############################################
    
            # Sample datapoints from ROI  
            ROI_selection_sampled = sample(ROI_selection, nbPoints)
            Y = np.array(V_dot[ROI_selection_sampled])
            X_tmp = X[ROI_selection_sampled, :]    
                    
            # Compute optimal linear filter   
            XTX = np.dot(np.transpose(X_tmp), X_tmp)
            XTX_inv = inv(XTX)
            XTY = np.dot(np.transpose(X_tmp), Y)
            K_opt_coeff = np.dot(XTX_inv, XTY)
            K_opt_coeff = K_opt_coeff.flatten()

            
            ############################################
            # ESTIMATE ELECTRODE FILETR K_e
            ############################################
            
            # Define K_opt
            K_opt_tmp = copy.deepcopy(self.K_opt)
            K_opt_tmp.setFilter_Coefficients(K_opt_coeff)
            self.K_opt_all.append(K_opt_tmp)
            
            # Fit exponential on tail of K_opt            
            (t,K_opt_tmp_interpol) = K_opt_tmp.getInterpolatedFilter(dt)
            (K_opt_tmp_expfit_t, K_opt_tmp_expfit) = K_opt_tmp.fitSumOfExponentials(len(self.p_b0), self.p_b0, self.p_tau0, ROI=self.p_expFitRange, dt=dt)

            # Generate electrode filter  
            Ke_coeff_tmp = (K_opt_tmp_interpol - K_opt_tmp_expfit)[ : int(self.p_Ke_l/dt) ]        
            Ke_tmp = Filter_Rect_LinSpaced(length=self.p_Ke_l, nbBins=len(Ke_coeff_tmp))
            Ke_tmp.setFilter_Coefficients(Ke_coeff_tmp)
            
            (Ke_tmp_expfit_t, Ke_tmp_expfit) = Ke_tmp.fitSumOfExponentials(1, [60.0], [0.5], ROI=[0.0,7.0], dt=dt)

            self.K_e_all.append(Ke_tmp)

            print "Repetition ", (rep+1), " R_e (MOhm) = %0.2f, " % (Ke_tmp.computeIntegral(dt))

        # Average filters obtained through bootstrap
        self.K_opt = Filter.averageFilters(self.K_opt_all)
        self.K_e = Filter.averageFilters(self.K_e_all)   
        
        print "Done!"      


    ##############################################################################################    
    # FUCTIONS TO APPLY AEC TO ALL TRACES IN THE EXPERIMENT
    ##############################################################################################    
    def compensateAllTraces(self, expr) :
        
        print "\nCompensate experiment"
        
        print "AEC trace..."
        self.deconvolveTrace(expr.AEC_trace)

        print "Training set..."        
        for tr in expr.trainingset_traces :
            self.deconvolveTrace(tr)
         
        print "Test set..."     
        for tr in expr.testset_traces :
            self.deconvolveTrace(tr)         
        
        print "Done!"
         
         
         
    def deconvolveTrace(self, trace):
        
        V_e = self.K_e.convolution_ContinuousSignal(trace.I, trace.dt)
        V_aec = trace.V_rec - V_e
        
        trace.V = V_aec
        trace.AEC_flag = True
        trace.detectSpikesWithDerivative(threshold=15)
        #trace.detectSpikes()
   
    

    #####################################################################################
    # FUNCTIONS FOR PLOTTING
    #####################################################################################
    def plot(self):
           
        # Plot optimal linear filter K_opt
        Filter.plotAverageFilter(self.K_opt_all, 0.05, loglog=False, label_x='Time (ms)', label_y='Optimal linear filter (MOhm/ms)')
 
        # Plot optimal linear filter K_e        
        Filter.plotAverageFilter(self.K_e_all, 0.05, label_x='Time (ms)', label_y='Electrode filter (MOhm/ms)')
  
        plt.show()
  
    def plotKopt(self):
           
        # Plot optimal linear filter K_opt
        Filter.plotAverageFilter(self.K_opt_all, 0.05, loglog=False, label_x='Time (ms)', label_y='Optimal linear filter (MOhm/ms)')
 
        plt.show()
 
    def plotKe(self):
           
        # Plot optimal linear filter K_e        
        Filter.plotAverageFilter(self.K_e_all, 0.05, label_x='Time (ms)', label_y='Electrode filter (MOhm/ms)', plot_expfit=False)
       
        plt.show()
