from typing import Dict
import os 
import json 
import torch 

def log_artifacts(
        checkpoints: Dict,
        train_metrics_for_one_epoch: Dict, 
        test_metrics_for_one_epoch: Dict, 
        artifacts_path: str=None,
        ):
    if not artifacts_path:
        artifacts_path = os.path.join(os.getcwd(), "artifacts")

    os.makedirs(artifacts_path, exist_ok=True)

    # Initialize history attribute if it doesn't exist
    if not hasattr(log_artifacts, "history"):
        log_artifacts.history = {key: {"train": [], "test": []} for key in train_metrics_for_one_epoch}

    # Append the new metrics to the history
    for key in train_metrics_for_one_epoch:
        log_artifacts.history[key]["train"].append(train_metrics_for_one_epoch[key])
        log_artifacts.history[key]["test"].append(test_metrics_for_one_epoch[key])
    
    torch.save(checkpoints, os.path.join(artifacts_path, "checkpoints.pt"))
    
    with open(os.path.join(artifacts_path, "history.json"), "w") as outfile:
        json.dump(log_artifacts.history, outfile, indent=4)


if __name__ == "__main__":
    train_metrics_for_one_epoch = {"Recall": 10, "precision": 30}
    test_metrics_for_one_epoch = {"Recall": 5, "precision": 2}
    saved_path = "./artifacts/"
    log_artifacts(train_metrics_for_one_epoch, test_metrics_for_one_epoch, saved_path)
    log_artifacts(train_metrics_for_one_epoch, test_metrics_for_one_epoch, saved_path)
    log_artifacts(train_metrics_for_one_epoch, test_metrics_for_one_epoch, saved_path)
    
    print(log_artifacts.history)



