import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as tfs
import cv2


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
                 transforms=None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                             'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                             'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                             'Pleural Other', 'Fracture', 'Support Devices'],
                 mode='train'):

        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace(
            'CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # impute missing values
        for col in train_cols:
            # print(col)
            # if col in ['Edema', 'Atelectasis']:
            #    self.df[col].replace(-1, 1, inplace=True)
            #    self.df[col].fillna(0, inplace=True)
            if col in ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                       'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                       'Pleural Other', 'Fracture', 'Support Devices']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        #assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 14 classes
            if verbose:
                print(
                    'Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
                print('-'*30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts(
                ).to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:       # 1 class
            # this var determines the number of classes
            self.select_cols = [train_cols[class_index]]
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts(
            ).to_dict()

        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size
        self.transforms = transforms

        self._images_list = [image_root_path +
                             path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:,
                                                           class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()

        if True:
            if class_index != -1:
                if flip_label:
                    self.imratio = self.value_counts_dict[1]/(
                        self.value_counts_dict[-1]+self.value_counts_dict[1])
                    if verbose:
                        print('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                            self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                        print('%s(C%s): imbalance ratio is %.4f' %
                              (self.select_cols[0], class_index, self.imratio))
                        print('-'*30)
                else:
                    self.imratio = self.value_counts_dict[1]/(
                        self.value_counts_dict[0]+self.value_counts_dict[1])
                    if verbose:
                        print('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                            self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                        print('%s(C%s): imbalance ratio is %.4f' %
                              (self.select_cols[0], class_index, self.imratio))
                        print('-'*30)
            else:
                # import ipdb; ipdb.set_trace()
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    if len(self.value_counts_dict[class_key]) == 1:
                        # self.value_counts_dict[class_key][0] = 202
                        self.value_counts_dict[class_key][1] = 0
                        imratio = 0
                    else:
                        imratio = self.value_counts_dict[class_key][1]/(
                            self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    if verbose:
                        #print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                            self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                        print('%s(C%s): imbalance ratio is %.4f' %
                              (select_col, class_key, imratio))
                        print()
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
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(
            0.05, 0.05), scale=(0.95, 1.05), fill=128)])  # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        data = {}

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train':
            if self.transforms is None:
                image = self.image_augmentation(image)
            else:
                image = self.transforms(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(
            self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        data['image'] = image

        if self.class_index != -1:  # multi-class mode
            label = np.array(
                self._labels_list[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(
                self._labels_list[idx]).reshape(-1).astype(np.float32)
        data['classes'] = label

        imagepath = self._images_list[idx]
        data['imagepath'] = imagepath

        return data

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

        imagepath = self._images_list[idx]
        data['imagepath'] = imagepath

        return data