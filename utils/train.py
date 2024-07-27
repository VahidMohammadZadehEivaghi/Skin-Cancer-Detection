from src import SkinCancerClassifier
from utils import SkinCancerDataset, Metrics, log_artifacts
from utils.weighting import MedianFrequencyBalance, ModeFrequencyBalance, InverseFrequencyMethod

import torch 
from torch.utils.data import DataLoader

from torchmetrics import Accuracy, Precision, Recall


import tqdm 
import os 
import json 

from typing import List 


def train(
            model: SkinCancerClassifier, 
            train_dataset: SkinCancerDataset, 
            test_dataset: SkinCancerDataset,
            batch_size: int,
            epochs: int, 
            lr: float, 
            device: torch.device,
            class_weights_computation_method: str="median_frequency_method", 
            artifacts_path: str=None, 
            per_epoch_averaging: bool = True, 
            best_critreion: str="accuracy"
          ):
        
        _freeze_backbone = model.__dict__.get("freeze_backbone")
        
        initial_criterion_value = 0
        start_epoch = 1

        if class_weights_computation_method == "median_frequency_method":
              class_weights = MedianFrequencyBalance().compute_class_weights(train_dataset)

        else:
              class_weights = InverseFrequencyMethod().compute_class_weights(train_dataset)

        

        weights = list(class_weights.values())

        model.to(device)

        if not _freeze_backbone:
            optimizer = torch.optim.Adam([
                  {"params": model.feature_maps.parameters(), "lr": 1e-2 * lr}, 
                  {"params": model.classifier.parameters()},
                  
            ],  lr=lr
                  )
        else:
              optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr
              )

        checkpoints_path = os.path.join(artifacts_path, "checkpoints.pt")
        history_path = os.path.join(artifacts_path, "history.json")
        if os.path.isfile(checkpoints_path):
             checkpoints = torch.load(checkpoints_path)
             model.load_state_dict(checkpoints['last_state_dict'])
             optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
             start_epoch = checkpoints['epoch'] + 1
             with open(history_path) as outfile:
                  history = json.load(outfile)
             log_artifacts.history = history

        cross_entropy_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights)).to(device)

        metrics = Metrics(
             metrics=[
                  Accuracy(task="multiclass", num_classes=7),
                  Precision(task="multiclass", num_classes=7, average="macro"), 
                  Recall(task="multiclass", num_classes=7, average="macro")
             ],
             device=device, 
             per_epoch_averaging=per_epoch_averaging
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)


        # Model training 

        for epoch in range(start_epoch, epochs + 1):
              
              model.train()
              metrics.reset()

              train_loss = 0

              pbar_train = tqdm.tqdm(train_loader, total=len(train_loader), desc=f"Model training: {epoch}/{epochs}")
        
              for inputs, labels in pbar_train:
                   inputs, labels = inputs.to(device), labels.to(device)
                   optimizer.zero_grad()  
                   outputs = model(inputs) 

                   loss = cross_entropy_loss_fn(outputs, labels)  
                   loss.backward()  
                   optimizer.step() 
                    
                   predicted_label = torch.argmax(outputs, dim=1)
                   target_label = torch.argmax(labels, dim=1)

                   metrics.update(predicted_label, target_label)

                   train_loss += loss.item()
                   criterion = metrics.compute()
                   criterion.update({"Loss": loss.item()})
            
                   pbar_train.set_postfix({**criterion})
            
              train_metrics_for_one_epoch = metrics.aggregate(num_batches=len(train_loader))
              train_metrics_for_one_epoch.update({
                    "Loss": train_loss / len(train_loader)
                    })
    
              pbar_train.close()

              # Model evaluation 
              
              test_metrics_for_one_epoch = evaluation(
                    model=model, test_loader=test_loader, metrics=metrics, weights=weights, device=device
              )

              print("-"*150)
              
              best_critreion_value = [value for key, value in test_metrics_for_one_epoch.items() if best_critreion in key.lower()][0]

              if best_critreion_value > initial_criterion_value:
                    initial_criterion_value = best_critreion_value
                    if not artifacts_path:
                        artifacts_path = os.path.join(os.getcwd(), "best.pt")
                    torch.save(model.state_dict(), os.path.join(artifacts_path, "best.pt"))
           
              checkpoints = {
                    "last_state_dict": model.state_dict(), 
                    "optimizer_state_dict": optimizer.state_dict(), 
                    "epoch": epoch
                    }
              log_artifacts(
                    checkpoints=checkpoints,
                    train_metrics_for_one_epoch=train_metrics_for_one_epoch, 
                    test_metrics_for_one_epoch=test_metrics_for_one_epoch
                    )
        
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluation(
      model: SkinCancerClassifier, 
      test_loader: DataLoader, 
      metrics: Metrics, 
      weights: List,
      device:torch.device
):
        
        cross_entropy_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights)).to(device)
        model.eval()
        
        pbar_test = tqdm.tqdm(test_loader, total=len(test_loader), desc="Model evaluation")
        
        test_loss = 0
        metrics.reset()         
        
        for inputs, labels in pbar_test:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) 

            predicted_label = torch.argmax(outputs, dim=1)
            target_label = torch.argmax(labels, dim=1)

            metrics.update(predicted_label, target_label)
                        
            criterion = metrics.compute()
            loss = cross_entropy_loss_fn(outputs, labels) 
            criterion.update({"Loss": loss.item()}) 

            test_loss += loss.item()

            pbar_test.set_postfix({**criterion})

    
        test_metrics_for_one_epoch = metrics.aggregate(num_batches=len(test_loader))
        test_metrics_for_one_epoch.update({"Loss": test_loss / len(test_loader)})
        return test_metrics_for_one_epoch
