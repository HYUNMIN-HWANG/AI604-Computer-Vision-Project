from libauc.losses import AUCMLoss, CrossEntropyLoss, AUCM_MultiLabel
from libauc.optimizers import PESG, Adam
from libauc.models import DenseNet121, DenseNet169, DenseNet201, DenseNet161
# import dataset_14cases #import CheXpert2


import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
from PIL import Image
import os
from glob import glob
import matplotlib.pyplot as plt
import ipdb

import warnings
warnings.filterwarnings('ignore')


class NIH_Dataset(Dataset) :
    def __init__(self, image_df, transform=None):   
        self.image_df = image_df
        self.transforms = transform
        self._images_list =  [path for path in self.image_df['path'].tolist()]
        self._labels_list = [path for path in np.stack(self.image_df['disease_vec']).tolist()]

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        data = {}

        image = Image.open(self._images_list[idx]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)

        #print(image)
        data['image'] = image
        # print("image shape : ", data['image'].shape)  # torch.Size([3, 224, 224])
        
        data['classes'] = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        # print("lable : ", data['classes'])
        # print("lable : ", data['classes'].shape)

        return data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':

    data = pd.read_csv('D:\\Data\\AI604_project\\NIH\\Data_Entry_2017.csv')
    data = data[data['Patient Age']<100] #removing datapoints which having age greater than 100
    data_image_paths = {os.path.basename(x): x for x in 
                       glob(os.path.join('D:\\Data\\AI604_project\\NIH', 'images*', '*', '*.png'))}

    print('Scans found:', len(data_image_paths), ', Total Headers', data.shape[0])  # Scans found: 112120 , Total Headers 112104
    data['path'] = data['Image Index'].map(data_image_paths.get)
    data['Patient Age'] = data['Patient Age'].map(lambda x: int(x))
    # data.sample(3)

    data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    from itertools import chain
    all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x)>0]
    print('All Labels ({}): {}'.format(len(all_labels), all_labels))
    for c_label in all_labels:
        if len(c_label)>1: # leave out empty labels
            data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
    #print("\n",list(data))

    MIN_CASES = 1000
    # all_labels = [c_label for c_label in all_labels if data[c_label].sum()>MIN_CASES]
    #print('Clean Labels ({})'.format(len(all_labels)), 
    #    [(c_label,int(data[c_label].sum())) for c_label in all_labels])

    # since the dataset is very unbiased, we can resample it to be a more reasonable collection
    # weight is 0.04 + number of findings
    #sample_weights = data['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
    #sample_weights /= sample_weights.sum()
    #data = data.sample(40000, weights=sample_weights)

    data['disease_vec'] = data.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(data, 
                                    test_size = 0.20, 
                                    random_state = 2018,
                                    stratify = data['Finding Labels'].map(lambda x: x[:4]))
    print('train', train_df.shape[0], 'test', test_df.shape[0]) # train 89683 test 22421

    train_df, valid_df = train_test_split(train_df, 
                                   test_size = 0.10, 
                                   random_state = 2018,
                                   stratify = train_df['Finding Labels'].map(lambda x: x[:4]))
    print('train', train_df.shape[0], 'valid', valid_df.shape[0])   # train 80714 valid 8969

    # paramaters
    SEED = 123
    BATCH_SIZE = 32
    lr = 1e-4
    weight_decay = 1e-5



    # Dataset and DataLoader
    root = 'D:\\Data\\AI604_project\\NIH\\'

    transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
    trainSet = NIH_Dataset(image_df=train_df, transform=transform_train)
    trainloader =  torch.utils.data.DataLoader(trainSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)

    transform_valid = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
    validSet = NIH_Dataset(image_df=valid_df, transform=transform_valid)
    validloader =  torch.utils.data.DataLoader(validSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

    transform_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
    testSet = NIH_Dataset(image_df=test_df, transform=transform_test)
    testloader =  torch.utils.data.DataLoader(testSet, batch_size=1, num_workers=2, shuffle=False)

    

    name = 'DenseNet121' # ['DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161']
    if name == 'DenseNet121' :
            model_name = DenseNet121
    elif name == 'DenseNet169' :
        model_name = DenseNet169
    elif name == 'DenseNet201' :
        model_name = DenseNet201
    else :
        model_name = DenseNet161

    writer = SummaryWriter('NIH_tensorboard\\DenseNet')
    model = model_name(pretrained=True, last_activation=None, activations='relu', num_classes=14)   
    model = model.to(device)

    # define loss & optimizer
    CELoss = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = 0 
    loss_total = []
    val_auc_mean = []


    for epoch in range(2):
        for idx, data in enumerate(trainloader):
            # ipdb.set_trace()
            # print("train idx " , idx)
            train_data, train_labels = data['image'], data['classes']
            # print(train_data.shape)     # torch.Size([BATCH_SIZE, 3, 224, 224]) 
            # print(train_labels.shape)   # torch.Size([3BATCH_SIZE, 14])

            train_data, train_labels  = train_data.to(device), train_labels.to(device)
            y_pred = model(train_data)
            # print(y_pred) # tensor([[ 0.4322, -0.1867, -0.2636, -0.6886,  0.0105,  0.1465, -0.9642, -0.6246, -0.4796,  0.5838, -0.2036,  0.0762,  0.0081, -0.0258], ...)
            loss = CELoss(y_pred, train_labels)

            loss_total += [loss.item()]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation 
            if idx % 5 == 0:
                # print("========validation========")
                model.eval()
                with torch.no_grad():    
                    test_pred = []
                    test_true = [] 
                    for jdx, data in enumerate(validloader):
                        # print("validation idx", jdx)
                        test_data, test_labels = data['image'], data['classes']
                        test_data = test_data.to(device)
                        y_pred = model(test_data)
                        test_pred.append(y_pred.cpu().detach().numpy())
                        test_true.append(test_labels.numpy())
                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)

                    for i in range (test_true.shape[0]) : 
                        if len(np.unique(test_true[i])) != 2 :
                            pass
                        else :
                            auc_mean =  roc_auc_score(test_true[i], test_pred[i])
                            val_auc_mean += [auc_mean.item()]
                    
                    writer.add_scalar('Pretraining/CELoss', np.mean(loss_total), epoch*len(trainloader) + idx)
                    writer.add_scalar('Pretraining/VAL AUC', np.mean(val_auc_mean), epoch*len(trainloader) + idx)

                    if best_val_auc < np.mean(val_auc_mean):
                        best_val_auc = np.mean(val_auc_mean)
                        torch.save(model.state_dict(), 'checkpoints\\NIH_Dennsenet121.pth')

                    print ('Epoch=%s, BatchID=[%s/%s] | Loss=%f, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, len(trainloader), np.mean(loss_total), np.mean(val_auc_mean), best_val_auc ))
                    
                    model.train()
    print ('Best Val_AUC is %.4f' % best_val_auc)
    # Best Val_AUC is 0.8443