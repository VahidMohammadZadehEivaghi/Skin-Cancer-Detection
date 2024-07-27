if __name__ == "__main__":
    from src import SkinCancerClassifier
    from utils import SkinCancerDataset

    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report 

    from torchmetrics.classification import MulticlassConfusionMatrix

    import torch 
    from torch.utils.data import DataLoader

    import json 
    import os
    import tqdm  

    path_to_test = "F:\\Data-pool\\HAM10000\\test"
    path_to_artifacts = "./artifacts"

    model_path = os.path.join(path_to_artifacts, "best.pt")

    with open(os.path.join(path_to_artifacts, "history.json")) as outfile:
        history = json.load(outfile)

    for key, value in history.items():
        train_hist = value.get("train")
        test_hist = value.get("test")
        plt.plot(train_hist, "b")
        plt.plot(test_hist, "r")
        plt.title(key)
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend(["Train", "Test"])
        plt.savefig(f"./{key}.png")
        plt.clf()

    test_data = SkinCancerDataset(path_to_test)
    test_loader = DataLoader(test_data, batch_size=1)
    model = SkinCancerClassifier(freeze_backbone=False, mode="test")
    model.load_state_dict(torch.load(model_path))


    confusion_matrix = MulticlassConfusionMatrix(num_classes=7)

    predict = []
    true = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in tqdm.tqdm(test_loader, total=len(test_loader), desc="Inference"):
            x = x.to(device)
            y_pred = model(x)
            predict.append(torch.argmax(y_pred).item())
            true.append(torch.argmax(y).item())

    confusion_matrix.update(torch.tensor(predict), torch.tensor(true))
    confusion_matrix.plot()
    plt.savefig("./confusion_matrix.png")
    print(classification_report(true, predict))