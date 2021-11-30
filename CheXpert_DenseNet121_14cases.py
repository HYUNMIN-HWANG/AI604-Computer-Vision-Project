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
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
from PIL import Image
import pandas as pd

class CheXpert2(Dataset):
    '''
    Reference: 
        Large-scale Robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification
        Zhuoning Yuan, Yan Yan, Milan Sonka, Tianbao Yang
        International Conference on Computer Vision (ICCV 2021)
    Use all 14 categories 
    '''
    def __init__(self, 
                 csv_path, 
                 image_root_path='',
                 image_size=320,
                 class_index=0, 
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 transforms= None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
                            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 
                            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                            'Pleural Other','Fracture','Support Devices'],
                 mode='train'):
        
    
        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']  
            
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print ('Upsampling %s...'%col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # impute missing values 
        for col in train_cols:
            # print(col)
            #if col in ['Edema', 'Atelectasis']:
            #    self.df[col].replace(-1, 1, inplace=True)  
            #    self.df[col].fillna(0, inplace=True) 
            if col in ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
                            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 
                            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                            'Pleural Other','Fracture','Support Devices']:
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)
        
        self._num_images = len(self.df)
        
        # 0 --> -1
        if flip_label and class_index != -1: # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)   
            
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
        
        
        #assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1: # 14 classes
            if verbose:
                print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
                print ('-'*30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:       # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()
        
        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size
        self.transforms = transforms
        
        self._images_list =  [image_root_path+path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()
    
        if True:
            if class_index != -1:
                if flip_label:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[-1]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
                else:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[0]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
            else:
                # import ipdb; ipdb.set_trace()
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    if len(self.value_counts_dict[class_key]) == 1 :
                        # self.value_counts_dict[class_key][0] = 202
                        self.value_counts_dict[class_key][1] = 0
                        imratio = 0
                    else :
                        imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    if verbose:
                        #print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                        print ()
                        #print ('-'*30)
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                
            
    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)
       
    @property  
    def data_size(self):
        return self._num_images 
    
    def image_augmentation(self, image):
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train' :
            if self.transforms is None:
                image = self.image_augmentation(image)
            else:
                image = self.transforms(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]  ]]) 
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        if self.class_index != -1: # multi-class mode
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        return image, label

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    root = 'D:/Data/AI604_project/CheXpert-v1.0-small/CheXpert-v1.0-small/'
    # Index: -1 denotes multi-label mode including 14 diseases
    traindSet = CheXpert2(csv_path=root+'train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1)
    testSet =  CheXpert2(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1)
    trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=32, num_workers=2, shuffle=True)
    testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, shuffle=False)

    # print(len(trainloader))   # 
    # print(len(testloader))    #  

    # paramaters
    SEED = 123
    BATCH_SIZE = 32
    lr = 1e-4
    weight_decay = 1e-5

    name = 'DenseNet121' # ['DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161']

    if name == 'DenseNet121' :
        model_name = DenseNet121
    elif name == 'DenseNet169' :
        model_name = DenseNet169
    elif name == 'DenseNet201' :
        model_name = DenseNet201
    else :
        model_name = DenseNet161

    writer = SummaryWriter('CheXpert_tensorboard\\14_cases')

    # model
    set_all_seeds(SEED)
    model = model_name(pretrained=True, last_activation=None, activations='relu', num_classes=14)   
    model = model.to(device)
    model

    # define loss & optimizer
    CELoss = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = 0 
    loss_total = []
    val_auc_mean = []
    for epoch in range(2):
        for idx, data in enumerate(trainloader):
            train_data, train_labels = data
            # print(train_data.shape)    # torch.Size([32,3,224,224])
            # print(train_labels)        # torch.Size([32, 14]) → exapmle :: [[0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.], [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
            # print(train_labels.shape)
            train_data, train_labels  = train_data.to(device), train_labels.to(device)
            y_pred = model(train_data)
            # print("y_pred shape", y_pred.shape) # torch.Size([32, 14])
            # print(y_pred) # [ 0.1937, -0.0255, -0.9805,  0.2090,  0.5745, -1.2703, -0.0286,  0.4180, 0.2593,  0.3300,  0.0505, -0.1631, -0.7381, -0.0025],
            loss = CELoss(y_pred, train_labels)

            loss_total += [loss.item()]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation  
            # if idx % 400 == 0:
            if idx % 50 == 0:
                model.eval()
                with torch.no_grad():    
                    test_pred = []
                    test_true = [] 
                    for jdx, data in enumerate(testloader):
                        test_data, test_labels = data
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
                    model.train()
                    
                    if best_val_auc < np.mean(val_auc_mean):
                        best_val_auc = np.mean(val_auc_mean)
                        torch.save(model.state_dict(), 'checkpoints\\DenseNet121_14classes_model.pth')
                    
                    print ('Epoch=%s, BatchID=[%s/%s] | Loss=%f, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, len(trainloader), loss.item(), np.mean(val_auc_mean), best_val_auc ))

    print ('Best Val_AUC is %.4f' % best_val_auc)


