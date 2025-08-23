import pandas as pd

from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

from neural_network.configuration import dataConfig, trainingConfig
from neural_network.data.dataset import KenyanFood13Dataset
from neural_network.data.augmentation import get_training_augmentation_pipeline, get_resize_pipeline    


def get_data_loaders(): # TODO: decide what term to use: test or validation.
        
    labels, class_to_indexes, class_counts = _parse_lables_and_classes()
    train_dataset, validation_dataset = _get_torch_datasets(labels, class_to_indexes)
    train_indexes, validaiton_indexes = _get_splitted_indexes(labels)

    if trainingConfig.single_batch_overfitting:
        single_batch = Subset(train_dataset, train_indexes[:dataConfig.batch_size])
        train_loader = DataLoader(single_batch,
                                  batch_size=dataConfig.batch_size,
                                  num_workers=0,
                                  persistent_workers=False,
                                  shuffle=True)
    else:
        train_loader = DataLoader(Subset(train_dataset, train_indexes),                     # Production train loader
                                  batch_size=dataConfig.batch_size, 
                                  num_workers=dataConfig.num_workers,
                                  persistent_workers=dataConfig.persistent_workers,
                                  shuffle = True)

    if trainingConfig.run_test_process:
        validation_loader = DataLoader(Subset(validation_dataset, validaiton_indexes),      # Production test loader
                                    batch_size=dataConfig.batch_size,
                                    num_workers=dataConfig.num_workers,
                                    persistent_workers=dataConfig.persistent_workers,
                                    shuffle=False)  
    else:
        validation_loader = None

    return train_loader, validation_loader, class_counts


def _parse_lables_and_classes():
    labels = pd.read_csv(dataConfig.label_file)  
    classes = sorted(labels.iloc[:, 1].unique())    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    class_counts = {idx: labels[labels.iloc[:, 1] == cls].shape[0] for cls, idx in class_to_idx.items()}
    
    return labels, class_to_idx, class_counts


def _get_torch_datasets(labels, class_to_idx):
    train_dataset = KenyanFood13Dataset(labels=labels, 
                                        image_directory=dataConfig.image_directory, 
                                        transform=_choose_augmentation_pipeline_for_training(), 
                                        class_to_idx=class_to_idx)
    
    validation_dataset = KenyanFood13Dataset(labels=labels, 
                                      image_directory=dataConfig.image_directory, 
                                      transform=get_resize_pipeline(), 
                                      class_to_idx=class_to_idx)
    
    return train_dataset, validation_dataset


def _choose_augmentation_pipeline_for_training():
    if dataConfig.data_augmentation:
        return get_training_augmentation_pipeline()
    else:
        return get_resize_pipeline()


def _get_splitted_indexes(labels):
    indexes = list(range(len(labels)))
    if trainingConfig.run_test_process:
        train_indexes, validaiton_indexes = train_test_split(indexes, 
                                            test_size=dataConfig.test_size,        
                                            random_state=dataConfig.seed, 
                                            stratify=labels.iloc[:, 1] if dataConfig.strategy else None)
    else:
        train_indexes, validaiton_indexes = indexes, []
    
    return train_indexes, validaiton_indexes

def get_indexes_to_class_mapping():
    _, class_to_idx, _ = _parse_lables_and_classes()
    return {idx: cls for cls, idx in class_to_idx.items()}

