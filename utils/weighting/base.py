from abc import abstractmethod
from typing import List 
from utils import SkinCancerDataset

class Weighting(object):

    
    def compute_class_distribution(self, dataset: SkinCancerDataset):
        metadata = dataset.metadata
        class_labels = metadata["dx"].to_list()
        class_distributions = {
            key: class_labels.count(key) for key in class_labels 
        }
        return class_distributions
    
    @abstractmethod
    def compute_class_weights(self, dataset: SkinCancerDataset):
         return NotImplemented
    


