from torchvision.models import ResNet50_Weights, resnet50
from torch import nn


class SkinCancerClassifier(nn.Module):

    def __init__(self, freeze_backbone: bool = True, mode: str="train"):
        super(SkinCancerClassifier, self).__init__()

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) 
        backbone = list(model.children())[:-1]
        self.feature_maps = nn.Sequential(*backbone)

        if freeze_backbone:
            for k, v in list(self.feature_maps.named_parameters())[:-3]:
                v.requires_grad = False
                print(f"{k} is freezed!")
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 7, bias=True),
            nn.Softmax(dim=1)
        )
        if mode == "train":
            model = nn.Sequential(
                self.feature_maps, self.classifier
            )
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {trainable_params}")

    def forward(self, x):
        phi = self.feature_maps(x)
        return self.classifier(phi)

if __name__ == "__main__":
    import torch 
    x = torch.randn(1, 3, 224, 224)
    model = SkinCancerClassifier()
    y = model(x)
 