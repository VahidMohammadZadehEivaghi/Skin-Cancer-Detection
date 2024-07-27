from utils import SkinCancerDataset
from src import SkinCancerClassifier
import torch 
from utils import train

if __name__ == "__main__":
    
    train_path = "F:\\Data-pool\\HAM10000\\train"
    test_path = "F:\\Data-pool\\HAM10000\\test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    lr = 1e-3
    num_epochs = 11

    train_dataset = SkinCancerDataset(train_path, transformation=True)
    test_dataset = SkinCancerDataset(test_path, transformation=True)
    
    model = SkinCancerClassifier(freeze_backbone=False, mode="train")

    train(
        model=model, 
        train_dataset=train_dataset, 
        test_dataset=test_dataset, 
        batch_size=batch_size, 
        lr=lr,
        epochs=num_epochs, 
        device=device, 
        class_weights_computation_method="median_frequency_method",
        artifacts_path="./artifacts",
        per_epoch_averaging=True
    )
   

