import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from data.cxr import LAD_Dataset


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])
    
transforms = {
    "CXR": {"train": T.Compose(
            [
                #T.Resize((512, 512)),
                T.RandomResizedCrop((224, 224)),    # for previous version of GbP, only resize to 224 is used here
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        ),
        "ext_valid": T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    },
    'heart_disease':{'train':T.Compose(
            [
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),
            ]
        ),
        'valid':T.Compose(
            [
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),    
            ]
        )
    }
}

def get_dataset(dataset_tag, dataset_split, transform_split, num_heads=8, proportion=0.8):
    if "pneumothorax" in dataset_tag or "pneumonia" in dataset_tag:
        transform = transforms["CXR"][transform_split]
        if dataset_tag == "Gender_pneumothorax_case1":
            csv_dir = './PBBL/dataset/csv/GbP-case1'
        elif dataset_tag == "Gender_pneumothorax_case2":
            csv_dir = './PBBL/dataset/csv/GbP-case2'
        elif dataset_tag == "Source_pneumonia_bias90":
            csv_dir = './PBBL/dataset/csv/SbP-bias90'
        elif dataset_tag == "Source_pneumonia_bias95":
            csv_dir = './PBBL/dataset/csv/SbP-bias95'
        elif dataset_tag == "Source_pneumonia_bias99":
            csv_dir = './PBBL/dataset/csv/SbP-bias99'
        elif dataset_tag == "Drain_pneumothorax_case1":
            csv_dir = './PBBL/dataset/csv/DbP-case1'
        elif dataset_tag == 'Source_pneumonia_balanced':
            csv_dir = './PBBL/dataset/csv/SbP-balanced'
        elif dataset_tag == 'Source_pneumonia_bias50':
            csv_dir = './PBBL/dataset/csv/SbP-bias50'
        elif dataset_tag == 'Source_pneumonia_bias75':
            csv_dir = './PBBL/dataset/csv/SbP-bias75'
        else:
            raise NotImplementedError

        dataset = LAD_Dataset(
            csv_dir=csv_dir,
            split=dataset_split,
            dataset_tag=dataset_tag,
            transform=transform,
            num_heads=num_heads,
            proportion=proportion)

    elif 'heart_disease' in dataset_tag:
        transform = transforms['heart_disease'][transform_split]
        if dataset_tag == 'Age_heart_disease':
            csv_dir = './PBBL/dataset/csv/OL3I-age'
        elif dataset_tag == 'Gender_heart_disease':
            csv_dir = './PBBL/dataset/csv/OL3I-gender'

        dataset = LAD_Dataset(
            csv_dir=csv_dir,
            split=dataset_split,
            dataset_tag=dataset_tag,
            transform=transform,
            num_heads=num_heads,
            proportion=proportion)

    return dataset


