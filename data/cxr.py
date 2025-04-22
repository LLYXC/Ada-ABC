import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import os
import h5py
import math

class LAD_Dataset(Dataset):
    def __init__(self, split, csv_dir, dataset_tag, num_heads=8, proportion=0.8, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        self.dataset_tag = dataset_tag


        super(LAD_Dataset, self).__init__()
        if split == 'train':
            csv_file = 'train.csv'
            print('Training with: ', csv_file)
        elif split == 'valid':
            csv_file = 'valid.csv'
            print('Validation: ', csv_file)
        else:
            csv_file = 'test.csv'
            print('Testing: ', csv_file)
        df = pd.read_csv(os.path.join(csv_dir, csv_file))

        if 'Gender_pneumothorax' in dataset_tag:
            pneumothorax = df['pneumothorax']
            gender = df['gender']
            attr = np.stack([np.array(pneumothorax), np.array(gender)], axis=1)
            self.filename = df['path']
        elif 'Source_pneumonia' in dataset_tag:
            pneumonia = df['pneumonia']
            data_source = df['MIMIC']
            attr = np.stack([np.array(pneumonia), np.array(data_source)], axis=1)
            self.filename = df['path']
        elif 'Age_heart_disease' in dataset_tag:
            heart_disease = df['heart_disease']
            age = df['age']
            attr = np.stack([np.array(heart_disease), np.array(age)], axis=1)
            self.filename = df['anon_id']
        elif 'Gender_heart_disease' in dataset_tag:
            heart_disease = df['heart_disease']
            gender = df['gender']
            attr = np.stack([np.array(heart_disease), np.array(gender)], axis=1)
            self.filename = df['anon_id']
        elif 'Drain_pneumothorax' in dataset_tag:
            pneumothorax = df['pneumothorax']
            is_drain = df['drain']
            attr = np.stack([np.array(pneumothorax), np.array(is_drain)], axis=1)
            self.filename = df['path']
        else:
            raise NotImplementedError
        dataset_len = len(attr)
        set_size = math.ceil(dataset_len * proportion)
        self.attr = torch.LongTensor(attr)
        self.transform = transform
        self.masks = np.array([[]] * num_heads).tolist()
        for i in range(num_heads):
            self.masks[i].append(np.random.choice(np.arange(dataset_len), set_size, replace=False))

        self.masks_place = torch.zeros(num_heads, dataset_len)
        for i in range(num_heads):
            self.masks_place[i, self.masks[i]] = 1
        
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        path = self.filename[index]


        #  dataset is OL3I
        if 'heart_disease' in self.dataset_tag:
            with h5py.File('/home/luoluyang/minghao_code/PBBL/dataset/image/OL3I/l3_slices.h5') as file:
                path = file[path]
                image_data = np.array(path)
                image_data = image_data.astype(np.float32)
                image = Image.fromarray(image_data).convert('RGB')


        else:

            if "Gender_pneumothorax" in self.dataset_tag:
                path = path.replace('./dataset/GbP', '/home/luoluyang/minghao_code/PBBL/dataset/image/NIH_GbP/useful_images')
            if "Source_pneumonia" in self.dataset_tag:
                if "NIH" in path:
                    path = path.replace('./dataset/NIH_JPG/useful_images', 
                                        '/home/luoluyang/minghao_code/PBBL/dataset/image/NIH_SbP/useful_images')
                if "MIMIC" in path:
                    path = path.replace('./dataset/MIMIC', 
                                        '/home/luoluyang/minghao_code/PBBL/dataset/image/MIMIC')
            if 'Drain_pneumothorax' in self.dataset_tag:
                path = path.replace('./dataset/GbP', 
                                        '/home/luoluyang/minghao_code/NIH-cut')
    
            image = Image.open(path).convert('L')
        
        attr = self.attr[index]

        if self.transform is not None:
            image = self.transform(image)
        mask = self.masks_place[:, index].tolist()

        if 'heart_disease' in self.dataset_tag:
            return image_data, image, attr, mask
        else:
            return path, image, attr, mask
    
    
    def update_prob(self,prob):
        self.prob = torch.cumsum(prob,dim=0)

    def idx_sample(self):
        return torch.clamp(torch.sum(torch.rand(1)>self.prob), 0, len(self.filename)-1 ).detach().cpu().numpy().tolist()

    def prob_sample_on(self):
        self.prob_on = True
    
    def prob_sample_off(self):
        self.prob_on = False

    def __len__(self):
        return len(self.filename)

    
    
