from torchmetrics.metric import Metric
from typing import List
from collections import Counter

import torch 

class Metrics(Metric):

    def __init__(self, metrics: List[Metric], 
                 device: torch.device, 
                 per_epoch_averaging: bool = True):
        
        if per_epoch_averaging:
            self._per_epoch_averaging_result = {}
            self._per_epoch_averaging = per_epoch_averaging
            for metric in metrics:
                self._per_epoch_averaging_result.update({
                    metric.__class__.__name__:0
                })

        self._metrics = [metric.to(device) for metric in metrics] 
        self._result = {metric.__class__.__name__:0 for metric in self._metrics}
    
    def reset(self):
        if self._per_epoch_averaging:
            self._reset_averaging_result()

        for metrics in self._metrics:
            metrics.reset()
    
    def _reset_averaging_result(self):
        for metric in self._metrics:
            self._per_epoch_averaging_result.update({
                    metric.__class__.__name__:0
                })

    def update(self, predicted_labels, target_labels):
        for metrics in self._metrics:
            metrics.update(predicted_labels, target_labels)

    def compute(self):
        for metrics in self._metrics:
            self._result.update(
                {
                    metrics.__class__.__name__: metrics.compute().item()
                }
            )

        if self._per_epoch_averaging:
            self._aggregate_result()

        return self._result
    
    def _aggregate_result(self):
        current_result = Counter(self._result)
        per_epoch_averaging_result = Counter(self._per_epoch_averaging_result)
        self._per_epoch_averaging_result = dict(current_result + per_epoch_averaging_result)
    
    def aggregate(self, num_batches:int):
            return {
                key: value/ num_batches for key, value in self._per_epoch_averaging_result.items()
            } if self._per_epoch_averaging else None 
    
    def get_device(self):
        for metric in self._metrics:
            print(metric.device)
        

if __name__ == "__main__":
    from torchmetrics import Recall, F1Score, Accuracy, MeanSquaredError, ExactMatch
    
    metric = Metrics(
        [
            MeanSquaredError(), 
            F1Score(task="multiclass", num_classes=5), 
            Recall(task="multiclass", num_classes=5), 
            ExactMatch(task="multiclass", num_classes=5), 
            Accuracy(task="multiclass", num_classes=5)
        ], 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        per_epoch_averaging=True
        )
    print(metric.get_device())

    
    
