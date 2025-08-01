"""
Data Loader
This module provides a data loader for the Kenyan Food 13 dataset.
"""

import os

import pandas as pd
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

from PIL import Image

import albumentations

from sklearn.model_selection import train_test_split


class KenyanFood13Dataset(Dataset):
    """
    Dataset class for the Kenyan Food 13 dataset.
    """
    def __init__(self, 
                 csv_file, 
                 data_dir, 
                 transform=None, 
                 class_to_idx=None):
        """
        Constuctor, set up the data folder, clean the classes.
        """
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        
        if class_to_idx is None:
            classes = sorted(self.data.iloc[:, 1].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        Read the image, apply transformations, and return the image and its label.
        """
        
        img_name = str(self.data.iloc[idx, 0])
        
        if not img_name.lower().endswith('.jpg'):
            img_name += '.jpg'
        
        label_str = self.data.iloc[idx, 1]
        label = self.class_to_idx[label_str]
        
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']            
            image = transforms.ToTensor()(image)
            
        return {"image": image, "target": label}
    
    
"""
Data Augmentation Pipeline
"""
training_augmentation_pipeline = albumentations.Compose([
    # First, resize to at least 300x300, then crop to 224x224
    albumentations.OneOf([
        albumentations.Compose([
            albumentations.LongestMaxSize(max_size=300),
            albumentations.PadIfNeeded(min_height=300, min_width=300, border_mode=0),
            albumentations.RandomCrop(height=224, width=224)
        ]),
        albumentations.Compose([
            albumentations.LongestMaxSize(max_size=300),
            albumentations.PadIfNeeded(min_height=300, min_width=300, border_mode=0),
            albumentations.RandomCrop(height=300, width=300),
            albumentations.Resize(height=224, width=224)
        ]),
        albumentations.Resize(height=224, width=224)
    ], p=1.0),
        
    # Second, noise or blur
    albumentations.OneOf([
        albumentations.AdditiveNoise(
            noise_type="gaussian",
            spatial_mode="shared",
            noise_params={"mean_range":[0,0],"std_range":[0.05,0.15]},
            approximation=1), 
        
        albumentations.Blur(
        blur_limit=[3, 7]
        ),
        
        albumentations.Defocus(
            radius=[3, 6],
            alias_blur=[0.1, 0.5]
        ),
        
        albumentations.GaussNoise(
            std_range=[0.1, 0.2],
            mean_range=[0, 0],
            per_channel=True,
            noise_scale_factor=0.2
        ),
        
        albumentations.GlassBlur(
            sigma=0.3,
            max_delta=3,
            iterations=2,
            mode="fast"
        ),
        
        albumentations.MotionBlur(
            blur_limit=[5, 11],
            allow_shifted=False,
            angle_range=[0, 90],
            direction_range=[-1, 1]
        )
    ], p = 1),
    
    # Flips
    albumentations.HorizontalFlip(p=0.5),
    
    # Rotation
    albumentations.RandomRotate90(p=0.8),

    
    # HSL adjustments
    albumentations.OneOf([
        albumentations.CLAHE(
            clip_limit=4,
            tile_grid_size=[8, 8]
        ),
        
        albumentations.RandomBrightnessContrast(
            brightness_limit=[-0.3, 0.3],
            contrast_limit=[-0.3, 0.3],
            brightness_by_max=True,
            ensure_safe_range=False
        ),
        
        albumentations.ColorJitter(
            brightness=[0.8, 1.2],
            contrast=[0.8, 1.2],
            saturation=[0.4, 0.6],
            hue=[-0.05, 0.05]
        )
        
    ], p = 1)
    
    
    ])    
    

resize_pipeline = albumentations.Compose([
    albumentations.Resize(height=224, width=224)
])

    
def get_data_loaders(data_root,
                     batch_size=8,
                     num_workers=2, 
                     seed=10, 
                     data_augmentation=True,
                     test_size=0.2,
                     persistent_workers=False):
    """
    Create data loaders for the Kenyan Food 13 dataset.
    """
    
    csv_file = os.path.join(data_root, 'train.csv')
    img_dir = os.path.join(data_root, 'images/images')
    
    df = pd.read_csv(csv_file)
    
    classes = sorted(df.iloc[:, 1].unique())
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Transformations
    if data_augmentation:
        # Use Albumentations for data augmentation
        pipeline = training_augmentation_pipeline
    else:
        # Use a simple resize transform
        pipeline = resize_pipeline


    # Split
    indices = list(range(len(df)))
    
    train_idx, val_idx = train_test_split(indices, 
                                          test_size=test_size,        #!!!! This should be a parameter
                                          random_state=seed, 
                                          stratify=df.iloc[:, 1])
    
    # Create datasets
    train_dataset = KenyanFood13Dataset(csv_file, 
                                        img_dir, 
                                        transform=pipeline, 
                                        class_to_idx=class_to_idx)
    
    val_dataset = KenyanFood13Dataset(csv_file, 
                                      img_dir, 
                                      transform=resize_pipeline, 
                                      class_to_idx=class_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(Subset(train_dataset, train_idx), 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers,
                              persistent_workers=persistent_workers)

    val_loader = DataLoader(Subset(val_dataset, val_idx), 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            persistent_workers=persistent_workers)
    
    
    return train_loader, val_loader, len(classes)



