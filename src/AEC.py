import abc
from Experiment import *


class AEC :

    """
    Abstract class defining an interface for performing active electrode compensation.
    """
    
    __metaclass__  = abc.ABCMeta
    
    @abc.abstractmethod
    def performAEC(self, experiment):
        
        """
        This method should preprocess all the traces in the experiment and compute V given V_rec.
        """
   
    @abc.abstractmethod
    def plot(self):
        
        """
        Plot the filter of AEC.
        """    
        
    
            
