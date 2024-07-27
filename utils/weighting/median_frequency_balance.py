import numpy as np 
from .base import Weighting
from utils import SkinCancerDataset
import numpy as np 

class MedianFrequencyBalance(Weighting):
     
     def compute_class_weights(self, dataset: SkinCancerDataset):
         
         class_distribution = self.compute_class_distribution(dataset)
         median = np.median(list(class_distribution.values()))
         return {
              key: median / value for key, value in class_distribution.items()
              }
     

