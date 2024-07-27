from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd 
import torch 
import cv2
from torchvision import transforms


class SkinCancerDataset(Dataset):

    __label_to_index = {
        'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'vasc':4,  'bcc':5, 'akiec':6
    }
    __index_to_label = {
        value:key for key, value in __label_to_index.items()
        }

    def __init__(self, path_to_data_dir, num_classes=7, transformation=True):

        self.path_to_data_dir = path_to_data_dir
        self.transformation = transformation

        self._img_indexes = os.listdir(
            os.path.join(path_to_data_dir, 'image')
            )
        self.metadata = pd.read_csv(
            os.path.join(path_to_data_dir, "metadata", "HAM10000_metadata.csv")
        )

        self.label = {row[1].get("image_id"): row[1].get("dx") for row in self.metadata.iterrows()}

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
            ]
            )
      
        self.num_classes = num_classes

    def __len__(self):
        return len(self._img_indexes)
    
    def __getitem__(self, index):
        img_index = self._img_indexes[index]
        label = self.__to_one_hot(
            self.label.get(img_index.split(".")[0])
        )
        image = cv2.imread(
            os.path.join(self.path_to_data_dir, 'image',  img_index)
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transformation:
            image = self.transform(image)
        
        return image, label
    
    def __to_one_hot(self, label):
        one_hot = torch.zeros(self.num_classes)
        index = SkinCancerDataset.__label_to_index.get(label)
        one_hot[index] = 1.0
        return one_hot
    
    def to_category(self, one_hot_vector):
        index = torch.argmax(one_hot_vector).item()
        return SkinCancerDataset.__index_to_label.get(index)

if __name__ == "__main__":
    path_to_data_dir = "F:\\Data-pool\\HAM10000\\test"
    data = SkinCancerDataset(path_to_data_dir)
    data_loader = DataLoader(data, batch_size=64)
    for image, label in data_loader:
        print(image.shape, label.shape)
