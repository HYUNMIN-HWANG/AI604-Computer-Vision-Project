import argparse
import cv2
import numpy as np
import torch
from torch._C import dtype
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from libauc.models import DenseNet121
from utils import set_all_seeds, CheXpert2, NIH_Dataset
import os
import csv
import torchvision.transforms as transforms
from glob import glob
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    # parser.add_argument(
    #     '--image-path',
    #     type=str,
    #     default='./CheXpert-v1.0-small/valid/patient64579/study1/view1_frontal.jpg', # ./examples/horses.jpg
    #     help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    DataChoice = 'NIH'

    if DataChoice == 'CheXpert':
        # Dataset and DataLoader for CheXpert2
        root = '/home/anya/pytorch-grad-cam/CheXpert-v1.0-small/'
        out_dir = './gradCAM/CheXpert/'
        table_data = []
        model_name = './DenseNet121_14classes_model.pth'

        # Index: -1 denotes multi-label mode including 14 diseases
        # DataSet = CheXpert2(csv_path=root+'train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1)
        # DataSet =  CheXpert2(csv_path=root+'train.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1)
        DataSet =  CheXpert2(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1)
        loader =  torch.utils.data.DataLoader(DataSet, batch_size=1, num_workers=2, shuffle=False)

    elif DataChoice == 'NIH':
        root = '/home/anya/pytorch-grad-cam/NIH'
        out_dir = './gradCAM/NIH_train/'
        table_data = []
        model_name = './NIH_Dennsenet121.pth'

        data = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
        data = data[data['Patient Age']<100] #removing datapoints which having age greater than 100
        data_image_paths = {os.path.basename(x): x for x in 
                        glob(os.path.join(root, 'images*', '*', '*.png'))}

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
        BATCH_SIZE = 1

        # Dataset and DataLoader for NIH
        transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
        DataSet = NIH_Dataset(image_df=train_df, transform=transform_train)
        loader =  torch.utils.data.DataLoader(DataSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

        # transform_valid = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
        # DataSet = NIH_Dataset(image_df=valid_df, transform=transform_valid)
        # loader =  torch.utils.data.DataLoader(DataSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

        # transform_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
        # DataSet = NIH_Dataset(image_df=test_df, transform=transform_test)
        # loader =  torch.utils.data.DataLoader(DataSet, batch_size=1, num_workers=2, shuffle=False)

    # model = models.resnet50(pretrained=True)
    model = DenseNet121(last_activation=None, activations='relu', num_classes=14)
    model.load_state_dict(torch.load(model_name)) 
    # set the evaluation mode - otherwise you can get very random results
    model.eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    target_layers = [model.features[-2]] # not -1, that's batch normalisation

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for idx, data in enumerate(loader):
        images, labels, imagepaths = data['image'], data['classes'], data['imagepath']
        images, labels  = images.to('cpu'), labels.to('cpu')
        y_pred = model(images) # y_pred.shape = torch.Size([1, 14])

        image_path = imagepaths[0]

        if DataChoice == 'CheXpert':
            # image_path = '/home/anya/pytorch-grad-cam/CheXpert-v1.0-small/valid/patient64589/study1/view1_frontal.jpg'
            patient = image_path.split('/')[6] # patient number is 2nd item
            study = image_path.split('/')[7]
            view = image_path.split('/')[8].replace('.jpg', '')
        elif DataChoice == 'NIH':
            # image_path = '/home/anya/pytorch-grad-cam/NIH/images_001/images/00001298_008.png'
            patient = image_path.split('/')[5] # image subgroup
            study = image_path.split('/')[6]
            view = image_path.split('/')[7].replace('.png', '')

        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # args.image_path
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        target_category = None
        # print somewhere the highest class and the corresponding box
        # comparison of boxes and gradCAM - IOU

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32

            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        gb = gb_model(input_tensor, target_category=target_category)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        os.makedirs(out_dir, exist_ok=True)
        gradcam_name = os.path.join(out_dir, f'{args.method}_cam_{patient}_{study}_{view}.jpg')
        cv2.imwrite(gradcam_name, cam_image) # we need only the gradCAM maps
    
        # writing the category and name of the image file in CSV file
        header = ['Path', 'GT Class', 'Class with highest predicted score']
                    
        gt = [i for i, e in enumerate(labels[0]) if e == 1] # ground truth labels
        category = [i for i, e in enumerate(y_pred[0]) if e == torch.max(y_pred[0])] # predictede label with max score
        table_data_i = [f'{image_path}', f'{gt}', f'{category}']
        table_data.append(table_data_i)

        with open(f'{out_dir}/label.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write the data
            writer.writerows(table_data)
        print(f'Image {gradcam_name} finished.')