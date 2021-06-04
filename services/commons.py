import numpy 
import time

import ConfigSpace as CS 
from hpbandster.core.worker import Worker 

class BOHBWorker(Worker):
    
    def __init__(self, *args,config,sleep_interval=0,**kwargs):
        super().__init__(*args,*kwargs)

        self.sleep_interval=sleep_interval
        self.config=config
    
    def compute(self,config,budget,**kwargs):
        """
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        
        res=0
        return({
            'loss':float(res),
            'info'
        })
        

    @staticmethod
    def get_configspace(self):
        return self.config
        

