
from .base import Weighting
from utils import SkinCancerDataset

class InverseFrequencyMethod(Weighting):
     def compute_class_weights(self, dataset: SkinCancerDataset):
         
         class_distribution = self.compute_class_distribution(dataset)
         return {
              key: len(dataset) / (value * len(class_distribution)) for key, value in class_distribution.items()
              }
     
