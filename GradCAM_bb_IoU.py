import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

cam1 = np.load("D:\\Data\\AI604_project\\gradCAM2\\NIH_test\\00000808_002.npy")
cam2 = np.load("D:\\Data\\AI604_project\\gradCAM2\\NIH_test\\00029259_027.npy")

img1 = cv2.imread('D:\\Data\\AI604_project\\NIH\\images_001\\images\\00000808_002.png')
img2 = cv2.imread('D:\\Data\\AI604_project\\NIH\\images_012\\images\\00029259_027.png')

# gt_box1 = (558.101695,	384.307352,	227.796610,	167.050847)
gt_box1 = (558,	384, 227, 167)
# gt_box2=  (787.775661,	386.844444,	112.694180,	291.487831)
gt_box2=  (787,	386, 112, 291)

cam1 = cam1 * 255.       ## only integer
cam1 = cam1.astype("uint8")

cam2 = cam2 * 255.       ## only integer
cam2 = cam2.astype("uint8")

_, src_bin1 = cv2.threshold(cam1, 60, 255, cv2.THRESH_BINARY)
_, src_bin2 = cv2.threshold(cam2, 60, 255, cv2.THRESH_BINARY) 

cnt1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(src_bin1, stats=1)
cnt2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(src_bin2, stats=1)

stats1 = sorted(stats1, key=lambda stats2: stats2[4], reverse=True) 
stats2 = sorted(stats2, key=lambda stats2: stats2[4], reverse=True)

(x, y, w, h, area) = stats1[1]   # The biggest GradCAM region
print(x, y, w, h)
gradcam_box1 = (x, y, w, h)
img1 = cv2.rectangle(img1, (x, y, w, h), (0, 255, 255),5)   # yellow
img1 = cv2.rectangle(img1, gt_box1, (0, 0, 255),5)          # red

(x, y, w, h, area) = stats2[1]   # The biggest GradCAM region
print(x, y, w, h)
gradcam_box2 = (x, y, w, h)
img2 = cv2.rectangle(img2, (x, y, w, h), (0, 255, 255),5)   # yellow : GradCAM
img2 = cv2.rectangle(img2, gt_box2, (0, 0, 255),5)          # red : Ground Truth

cv2.imshow('00000808_002', img1)
cv2.imshow('00029259_027', img2)

cv2.imwrite('./00000808_002.png', img1)
cv2.imwrite('./00029259_027.png', img2)

cv2.waitKey()
cv2.destroyAllWindows()

######################################################################################
# Calculated IoU
SMOOTH = 1e-6

def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):

    # midpoints : input (x,y,w,h)
    # corners : input (x1,y1,x2,y2)

    # boxes_preds shape : (N,4) N is number of bboxes
    # boxes_labels shape : (N,4)

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1] # (N,1)
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = np.max([box1_x1.astype("float").item(), box2_x1.astype("float").item()])
    y1 = np.max([box1_y1.astype("float").item(), box2_y1.astype("float").item()])
    x2 = np.max([box1_x2.astype("float").item(), box2_x2.astype("float").item()])
    y2 = np.max([box1_y2.astype("float").item(), box2_y2.astype("float").item()])

    # .clip(0) : if interconnect is not performed
    # Set the minimum value to zero
    intersection = (x2 - x1).clip(0) * (y2 - y1).clip(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

iou = intersection_over_union(np.array(gt_box1), np.array(gradcam_box1))
print(iou.item())   # 0.35250527702299433
iou = intersection_over_union(np.array(gt_box2), np.array(gradcam_box2))
print(iou.item())   # 0.35250527702299433