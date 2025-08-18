import os

from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
import numpy as np
from PIL import Image


class KenyanFood13Dataset(Dataset):
    """
    Dataset class for the Kenyan Food 13 dataset.
    """
    def __init__(self, 
                 labels, 
                 image_directory, 
                 transform=None, 
                 class_to_idx=None):
        self.labels = labels
        self.image_directory = image_directory
        self.transform = transform
        
        if class_to_idx is None:
            classes = sorted(self.labels.iloc[:, 1].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "target": self.process_label(idx),
            "image": self.process_image(idx)            
        }

    def process_label(self, idx):
        label_str = self.labels.iloc[idx, 1]
        label = self.class_to_idx[label_str]
        return label

    def process_image(self, idx):
        image_filepath = self.get_image_name(idx)
        image = self.import_image(image_filepath)
        image = self.apply_transform_if_any(image)
        return image

    def get_image_name(self, idx):
        image_filepath = str(self.labels.iloc[idx, 0])
        
        if not image_filepath.lower().endswith('.jpg'):
            image_filepath += '.jpg'
        
        return image_filepath

    def import_image(self, image_filepath):
        image_filepath = os.path.join(self.image_directory, image_filepath)
        image = Image.open(image_filepath).convert('RGB')
        return np.array(image)
    
    def apply_transform_if_any(self, image):
        if self.transform:
            image = self.transform(image=image)['image']
            image = transforms.ToTensor()(image)
        return image