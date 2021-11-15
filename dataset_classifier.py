import os
import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from glob import glob
import csv
import pandas as pd
import torchvision
import PIL
import torchvision.transforms as transforms

torch.manual_seed(17)

# CheXpert
image_path = 'D:\Data\AI604_project\CheXpert-v1.0-small'

class Chexpert_Dataset(Dataset):
    """ CheXpert dataset."""

    def __init__(self, path, transfrom=None, task='train'): # task = ['train', 'valid']

        self.image_list = []
        self.label_list = []
        self.path = path
        self.transfomer = transfrom
        
        ## LOAD METADATA
        csv_file = os.path.join(self.path, 'CheXpert-v1.0-small\\{}.csv'.format(task))
        csv_file = pd.read_csv(csv_file)
        
        ## Extract only Frontal part
        only_frontal = csv_file[csv_file['Frontal/Lateral'] == 'Frontal'].reset_index()
        image_list = only_frontal['Path'].values.tolist()

        ## MAKE THE IMAGE LIST
        for image in image_list :
            image_path = os.path.join(self.path, image)
            self.image_list += [image_path]
        print("Total %s images : " % task, len(self.image_list) )

        ## MAKE THE LABEL LIST
        ## labeled only present (1)
        self.label_list = [[0]*14 for i in range(len(self.image_list))]

        for idx in range(6,20) : 
            a = only_frontal[only_frontal.iloc[:, idx] == 1.0]

            for i in a.index : 
                self.label_list[i][idx-6] = 1

    def __getitem__(self, index):   
   
        image = self.image_list[index]
        image = PIL.Image.open(image)
        image_tf = self.transfomer(image)

        label = self.label_list[index]
        label_tf = torch.tensor(label)

        return image_tf, label_tf

    def __len__(self):  
        return len(self.image_list)


# EXAMPLE
transform_dataset = transforms.Compose([
            transforms.Resize((256, 256)),  # Have to fit same size 
            transforms.ToTensor(),
        ])
dataset_valid_dataset = Chexpert_Dataset(image_path, transfrom=transform_dataset, task='valid')  

img5 = dataset_valid_dataset.__getitem__(5)
print("Image 5 : \n", img5[0])              # image
print("Image 5 shape : ", img5[0].shape)  
print("Label 5 : \n", img5[1])              # label

len_data = dataset_valid_dataset.__len__()
print(len_data) # length of dataset


data_loader = DataLoader(dataset=dataset_valid_dataset, batch_size=4, shuffle=False)
for batch_idx, samples in enumerate(data_loader):
    print(batch_idx)
    print(samples[0].shape) # image size : torch.Size([batch_size, 1, 256, 256])
    print(samples[1].shape) # label size : torch.Size([batch_size, 14])

