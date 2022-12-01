
# from scipy.misc import imsave
import os.path as osp
import numpy as np
import os
import cv2
from skimage import morphology
import scipy
from PIL import Image
from matplotlib.pyplot import imsave
# from keras.preprocessing import image
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from skimage import measure, draw

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# from scipy.misc import imsave
from utils.metrics import *
import cv2
"""Functions for ramping hyperparameters up or down
Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""
import torch.nn.functional as F
import numpy as np
import math
import albumentations as al


def get_augmentation():
    return al.Compose([
        # al.RandomResizedCrop(512, 512),
        al.Compose([
            # NOTE: RandomBrightnessContrast replaces ColorJitter
            al.RandomBrightnessContrast(p=1),
            al.HueSaturationValue(p=1),
        ], p=0.8),
        al.ToGray(p=0.2),
        al.GaussianBlur(5, p=0.5),
    ])
def augment(images, labels, aug):
    """Augments both image and label. Assumes input is a PyTorch tensor with
       a batch dimension and values normalized to N(0,1)."""

    # Transform label shape: B, C, W, H ==> B, W, H, C
    labels_are_3d = (len(labels.shape) == 4)
    if labels_are_3d:
        # print("label shape is:",labels.shape) ##torch.Size([8, 2, 512, 512]
        labels = labels.permute(0, 2, 3, 1)  ##torch.Size([8, 512, 512, 2]

    # Transform each image independently. This is slow, but whatever.
    aug_images, aug_labels = [], []
    for image, label in zip(images, labels):
        # print("image shape is:",image.shape)  ##torch.Size([3, 512, 512]
        # Step 1: Undo normalization transformation, convert to numpy
        image = cv2.cvtColor((image.numpy().transpose(
            1, 2, 0)+1)*127.5, cv2.COLOR_BGR2RGB).astype(np.uint8)   ##torch.Size([512, 512,3]
        label = label.numpy()  # convert to np

        # Step 2: Perform transformations on numpy images
        data = aug(image=image, mask=label)
        image, label = data['image'], data['mask']

        # Step 3: Convert back to PyTorch tensors
        image = torch.from_numpy((cv2.cvtColor((image.astype(
            np.float32))/127.5-1, cv2.COLOR_RGB2BGR)).transpose(2, 0, 1))
        label = torch.from_numpy(label)
        if not labels_are_3d:
            label = label.long()

        # Add to list
        aug_images.append(image)
        aug_labels.append(label)

    # Stack
    images = torch.stack(aug_images, dim=0)
    labels = torch.stack(aug_labels, dim=0)

    # Transform label shape back: B, C, W, H ==> B, W, H, C
    if labels_are_3d:
        labels = labels.permute(0, 3, 1, 2)
    return images, labels
def get_prototype_weight(feat, class_num, prototype):
    weight = torch.cosine_similarity(prototype, feat, dim=1).unsqueeze(1)
    # print(weight.shape)
    # N, C, H, W = feat.shape
    # feat_proto_distance = -torch.ones((N, class_num, H, W)).to(feat.device)
    # for i in range(class_num):
    #     # feat_proto_distance[:, i, :, :] = torch.norm(torch.Tensor(self.objective_vectors[i]).reshape(-1,1,1).expand(-1, H, W).to(feat.device) - feat, 2, dim=1,)
    #     print(prototype.shape,feat.shape)
    #     feat_proto_distance[:, i, :, :] = torch.norm(prototype.reshape(-1, 1, 1).expand(-1, H, W) - feat, 2, dim=1, )
    # # weight = torch.cosine_similarity(prototype.reshape(-1,1),feat.reshape(C,-1),dim=0).reshape(N, class_num, H, W)
    # # feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)
    # #
    # # feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
    # # weight = F.softmax(-feat_proto_distance * 1.0, dim=1)
    # # feat_proto_distance = -feat_proto_distance
    # # weight = (feat_proto_distance-feat_proto_distance.min())/(feat_proto_distance.max()-feat_proto_distance.min())
    # weight = 1 - (feat_proto_distance / feat_proto_distance.max())
    return weight
def adaptation_factor(m):
    den = 1.0 + math.exp(-0.8 * (m+1))
    lamb = 1.0 / den - 0.3
    return lamb
def gen_prototype(pred_oS, xs_feature):
    source_cup_obj = pred_oS[:, 0:1]
    source_disc_obj = pred_oS[:, 1:]
    source_cup_bck = 1.0 - source_cup_obj
    source_disc_bck = 1.0 - source_disc_obj

    sourcefeature_cup_obj = xs_feature * source_cup_obj
    sourcefeature_disc_obj = xs_feature * source_disc_obj
    sourcefeature_cup_bck = xs_feature * source_cup_bck
    sourcefeature_disc_bck = xs_feature * source_disc_bck

    sourcecentroid_0_obj = torch.sum(sourcefeature_cup_obj, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_1_obj = torch.sum(sourcefeature_disc_obj, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_0_bck = torch.sum(sourcefeature_cup_bck, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_1_bck = torch.sum(sourcefeature_disc_bck, dim=[0, 2, 3], keepdim=True)
    source_0_obj_cnt = torch.sum(source_cup_obj, dim=[0, 2, 3], keepdim=True)
    source_1_obj_cnt = torch.sum(source_disc_obj, dim=[0, 2, 3], keepdim=True)
    source_0_bck_cnt = torch.sum(source_cup_bck, dim=[0, 2, 3], keepdim=True)
    source_1_bck_cnt = torch.sum(source_disc_bck, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_0_obj /= source_0_obj_cnt
    sourcecentroid_1_obj /= source_1_obj_cnt
    sourcecentroid_0_bck /= source_0_bck_cnt
    sourcecentroid_1_bck /= source_1_bck_cnt
    return sourcecentroid_0_obj, sourcecentroid_1_obj, sourcecentroid_0_bck, sourcecentroid_1_bck
def gen_prototype_src_trg(pred_oS, xs_feature, pred_oT, xt_feature):
    pred_oS = torch.cat((pred_oS,pred_oT), 0)
    xs_feature = torch.cat((xs_feature,xt_feature), 0)

    source_cup_obj = pred_oS[:, 0:1]
    source_disc_obj = pred_oS[:, 1:]
    source_cup_bck = 1.0 - source_cup_obj
    source_disc_bck = 1.0 - source_disc_obj

    sourcefeature_cup_obj = xs_feature * source_cup_obj
    sourcefeature_disc_obj = xs_feature * source_disc_obj
    sourcefeature_cup_bck = xs_feature * source_cup_bck
    sourcefeature_disc_bck = xs_feature * source_disc_bck

    sourcecentroid_0_obj = torch.sum(sourcefeature_cup_obj, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_1_obj = torch.sum(sourcefeature_disc_obj, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_0_bck = torch.sum(sourcefeature_cup_bck, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_1_bck = torch.sum(sourcefeature_disc_bck, dim=[0, 2, 3], keepdim=True)
    source_0_obj_cnt = torch.sum(source_cup_obj, dim=[0, 2, 3], keepdim=True)
    source_1_obj_cnt = torch.sum(source_disc_obj, dim=[0, 2, 3], keepdim=True)
    source_0_bck_cnt = torch.sum(source_cup_bck, dim=[0, 2, 3], keepdim=True)
    source_1_bck_cnt = torch.sum(source_disc_bck, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_0_obj /= source_0_obj_cnt
    sourcecentroid_1_obj /= source_1_obj_cnt
    sourcecentroid_0_bck /= source_0_bck_cnt
    sourcecentroid_1_bck /= source_1_bck_cnt
    return sourcecentroid_0_obj, sourcecentroid_1_obj, sourcecentroid_0_bck, sourcecentroid_1_bck
def gen_prototype_retrify(oT_before, xt_feature,preds,features,T, stride):

    preds = preds.reshape(T, stride, 2, preds.shape[2], preds.shape[3])
    features = features.reshape(T, stride, 305, 128, 128)

    preds1 = torch.sigmoid(preds)
    preds = torch.sigmoid(preds / 2.0)
    std_map = torch.std(preds, dim=0)

    prediction = torch.mean(preds1, dim=0)
    feature = torch.mean(features, dim=0)
    prediction_small = F.interpolate(prediction, size=feature.size()[2:], mode='bilinear', align_corners=True)
    std_map_small = F.interpolate(std_map, size=feature.size()[2:], mode='bilinear', align_corners=True)

    pred_oT = torch.sigmoid(oT_before)

    pseudo_label = pred_oT.clone()
    pseudo_label[pseudo_label > 0.75] = 1.0;
    pseudo_label[pseudo_label <= 0.75] = 0.0
    target_cup_obj = pseudo_label[:, 0:1]
    target_disc_obj = pseudo_label[:, 1:]
    target_cup_bck = 1.0 - target_cup_obj
    target_disc_bck = 1.0 - target_disc_obj

    # target_cup_obj_weights = get_prototype_weight(xt_feature, 1, src_0_obj)
    # target_cup_bck_weights = get_prototype_weight(xt_feature, 1, src_0_bck)
    # target_disc_obj_weights = get_prototype_weight(xt_feature, 1, src_1_obj)
    # target_disc_bck_weights = get_prototype_weight(xt_feature, 1, src_1_bck)
    # threshold = adaptation_factor(epoch)
    mask_0_obj = torch.zeros(
        [xt_feature.shape[0], 1, xt_feature.shape[2], xt_feature.shape[3]]).cuda()
    mask_0_bck = torch.zeros(
        [xt_feature.shape[0], 1, xt_feature.shape[2], xt_feature.shape[3]]).cuda()
    mask_1_obj = torch.zeros(
        [xt_feature.shape[0], 1, xt_feature.shape[2], xt_feature.shape[3]]).cuda()
    mask_1_bck = torch.zeros(
        [xt_feature.shape[0], 1, xt_feature.shape[2], xt_feature.shape[3]]).cuda()
    # print(std_map_small.max(),std_map_small.min(),std_map_small.mean())
    mask_0_obj[std_map_small[:, 0:1, ...] < 0.04] = 1.0
    mask_0_bck[std_map_small[:, 0:1, ...] < 0.04] = 1.0
    mask_1_obj[std_map_small[:, 1:, ...] < 0.04] = 1.0
    mask_1_bck[std_map_small[:, 1:, ...] < 0.04] = 1.0
    # mask_0_obj[target_cup_obj_weights > threshold] = 1.0
    # mask_0_bck[target_cup_bck_weights > threshold] = 1.0
    # mask_1_obj[target_disc_obj_weights > threshold] = 1.0
    # mask_1_bck[target_disc_bck_weights > threshold] = 1.0
    mask_0 = mask_0_obj + mask_0_bck
    mask_1 = mask_1_obj + mask_1_bck
    targetfeature_cup_obj = xt_feature * target_cup_obj * mask_0_obj;
    targetfeature_disc_obj = xt_feature * target_disc_obj * mask_1_obj
    targetfeature_cup_bck = xt_feature * target_cup_bck * mask_0_bck;
    targetfeature_disc_bck = xt_feature * target_disc_bck * mask_1_bck

    targetcentroid_0_obj = torch.sum(targetfeature_cup_obj*prediction_small[:, 0:1], dim=[0, 2, 3], keepdim=True)
    targetcentroid_1_obj = torch.sum(targetfeature_disc_obj*prediction_small[:, 1:], dim=[0, 2, 3], keepdim=True)
    targetcentroid_0_bck = torch.sum(targetfeature_cup_bck*(1-prediction_small[:, 0:1]), dim=[0, 2, 3], keepdim=True)
    targetcentroid_1_bck = torch.sum(targetfeature_disc_bck*(1-prediction_small[:, 1:]), dim=[0, 2, 3], keepdim=True)
    target_0_obj_cnt = torch.sum(mask_0_obj * target_cup_obj*prediction_small[:, 0:1], dim=[0, 2, 3], keepdim=True)
    target_1_obj_cnt = torch.sum(mask_1_obj * target_disc_obj*prediction_small[:, 1:], dim=[0, 2, 3], keepdim=True)
    target_0_bck_cnt = torch.sum(mask_0_bck * target_cup_bck*(1-prediction_small[:, 0:1]), dim=[0, 2, 3], keepdim=True)
    target_1_bck_cnt = torch.sum(mask_1_bck * target_disc_bck*(1-prediction_small[:, 1:]), dim=[0, 2, 3], keepdim=True)
    targetcentroid_0_obj /= target_0_obj_cnt
    targetcentroid_1_obj /= target_1_obj_cnt
    targetcentroid_0_bck /= target_0_bck_cnt
    targetcentroid_1_bck /= target_1_bck_cnt
    return targetcentroid_0_obj,targetcentroid_1_obj,targetcentroid_0_bck,targetcentroid_1_bck,\
           std_map,mask_0,mask_1

def gen_prototype_src_trg_retrify(pred_oS, xs_feature,oT_before, xt_feature,preds,features,T, stride):

    preds = preds.reshape(T, stride, 2, preds.shape[2], preds.shape[3])
    features = features.reshape(T, stride, 305, 128, 128)

    preds1 = torch.sigmoid(preds)
    preds = torch.sigmoid(preds / 2.0)
    std_map = torch.std(preds, dim=0)

    prediction = torch.mean(preds1, dim=0)
    feature = torch.mean(features, dim=0)
    prediction_small = F.interpolate(prediction, size=feature.size()[2:], mode='bilinear', align_corners=True)
    std_map_small = F.interpolate(std_map, size=feature.size()[2:], mode='bilinear', align_corners=True)

    pred_oT = torch.sigmoid(oT_before)
    pseudo_label = pred_oT.clone()
    pseudo_label[pseudo_label > 0.75] = 1.0;
    pseudo_label[pseudo_label <= 0.75] = 0.0
    target_cup_obj = pseudo_label[:, 0:1]
    target_disc_obj = pseudo_label[:, 1:]
    target_cup_bck = 1.0 - target_cup_obj
    target_disc_bck = 1.0 - target_disc_obj

    # target_cup_obj_weights = get_prototype_weight(xt_feature, 1, src_0_obj)
    # target_cup_bck_weights = get_prototype_weight(xt_feature, 1, src_0_bck)
    # target_disc_obj_weights = get_prototype_weight(xt_feature, 1, src_1_obj)
    # target_disc_bck_weights = get_prototype_weight(xt_feature, 1, src_1_bck)
    # threshold = adaptation_factor(epoch)
    mask_0_obj = torch.zeros(
        [xt_feature.shape[0], 1, xt_feature.shape[2], xt_feature.shape[3]]).cuda()
    mask_0_bck = torch.zeros(
        [xt_feature.shape[0], 1, xt_feature.shape[2], xt_feature.shape[3]]).cuda()
    mask_1_obj = torch.zeros(
        [xt_feature.shape[0], 1, xt_feature.shape[2], xt_feature.shape[3]]).cuda()
    mask_1_bck = torch.zeros(
        [xt_feature.shape[0], 1, xt_feature.shape[2], xt_feature.shape[3]]).cuda()
    mask_0_obj[std_map_small[:, 0:1, ...] < 0.04] = 1.0
    mask_0_bck[std_map_small[:, 0:1, ...] < 0.04] = 1.0
    mask_1_obj[std_map_small[:, 1:, ...] < 0.04] = 1.0
    mask_1_bck[std_map_small[:, 1:, ...] < 0.04] = 1.0
    # mask_0_obj[target_cup_obj_weights > threshold] = 1.0
    # mask_0_bck[target_cup_bck_weights > threshold] = 1.0
    # mask_1_obj[target_disc_obj_weights > threshold] = 1.0
    # mask_1_bck[target_disc_bck_weights > threshold] = 1.0
    targetfeature_cup_obj = xt_feature * target_cup_obj * mask_0_obj;
    targetfeature_disc_obj = xt_feature * target_disc_obj * mask_1_obj
    targetfeature_cup_bck = xt_feature * target_cup_bck * mask_0_bck;
    targetfeature_disc_bck = xt_feature * target_disc_bck * mask_1_bck

    targetcentroid_0_obj = torch.sum(targetfeature_cup_obj*prediction_small[:, 0:1], dim=[0, 2, 3], keepdim=True)
    targetcentroid_1_obj = torch.sum(targetfeature_disc_obj*prediction_small[:, 1:], dim=[0, 2, 3], keepdim=True)
    targetcentroid_0_bck = torch.sum(targetfeature_cup_bck*(1-prediction_small[:, 0:1]), dim=[0, 2, 3], keepdim=True)
    targetcentroid_1_bck = torch.sum(targetfeature_disc_bck*(1-prediction_small[:, 1:]), dim=[0, 2, 3], keepdim=True)
    target_0_obj_cnt = torch.sum(mask_0_obj * target_cup_obj*prediction_small[:, 0:1], dim=[0, 2, 3], keepdim=True)
    target_1_obj_cnt = torch.sum(mask_1_obj * target_disc_obj*prediction_small[:, 1:], dim=[0, 2, 3], keepdim=True)
    target_0_bck_cnt = torch.sum(mask_0_bck * target_cup_bck*(1-prediction_small[:, 0:1]), dim=[0, 2, 3], keepdim=True)
    target_1_bck_cnt = torch.sum(mask_1_bck * target_disc_bck*(1-prediction_small[:, 1:]), dim=[0, 2, 3], keepdim=True)

    source_cup_obj = pred_oS[:, 0:1]
    source_disc_obj = pred_oS[:, 1:]
    source_cup_bck = 1.0 - source_cup_obj
    source_disc_bck = 1.0 - source_disc_obj

    sourcefeature_cup_obj = xs_feature * source_cup_obj
    sourcefeature_disc_obj = xs_feature * source_disc_obj
    sourcefeature_cup_bck = xs_feature * source_cup_bck
    sourcefeature_disc_bck = xs_feature * source_disc_bck

    sourcecentroid_0_obj = torch.sum(sourcefeature_cup_obj, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_1_obj = torch.sum(sourcefeature_disc_obj, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_0_bck = torch.sum(sourcefeature_cup_bck, dim=[0, 2, 3], keepdim=True)
    sourcecentroid_1_bck = torch.sum(sourcefeature_disc_bck, dim=[0, 2, 3], keepdim=True)
    source_0_obj_cnt = torch.sum(source_cup_obj, dim=[0, 2, 3], keepdim=True)
    source_1_obj_cnt = torch.sum(source_disc_obj, dim=[0, 2, 3], keepdim=True)
    source_0_bck_cnt = torch.sum(source_cup_bck, dim=[0, 2, 3], keepdim=True)
    source_1_bck_cnt = torch.sum(source_disc_bck, dim=[0, 2, 3], keepdim=True)
    src_trgcentroid_0_obj = sourcecentroid_0_obj+targetcentroid_0_obj
    src_trgcentroid_1_obj = sourcecentroid_1_obj+targetcentroid_1_obj
    src_trgcentroid_0_bck = sourcecentroid_0_bck+targetcentroid_0_bck
    src_trgcentroid_1_bck = sourcecentroid_1_bck+targetcentroid_1_bck
    src_trgcentroid_0_obj /= (source_0_obj_cnt+target_0_obj_cnt)
    src_trgcentroid_1_obj /= (source_1_obj_cnt+target_1_obj_cnt)
    src_trgcentroid_0_bck /= (source_0_bck_cnt+target_0_bck_cnt)
    src_trgcentroid_1_bck /= (source_1_bck_cnt+target_1_bck_cnt)
    return src_trgcentroid_0_obj,src_trgcentroid_1_obj,src_trgcentroid_0_bck,src_trgcentroid_1_bck
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
class WeightEMA(object):
    def __init__(self, params, src_params, alpha):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
def construct_color_img(prob_per_slice):
    shape = prob_per_slice.shape
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = prob_per_slice * 255
    img[:, :, 1] = prob_per_slice * 255
    img[:, :, 2] = prob_per_slice * 255

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return im_color


def normalize_ent(ent):
    '''
    Normalizate ent to 0 - 1
    :param ent:
    :return:
    '''
    min = np.amin(ent)
    return (ent - min) / 0.4


def draw_ent(prediction, save_root, name):
    '''
    Draw the entropy information for each img and save them to the save path
    :param prediction: [2, h, w] numpy
    :param save_path: string including img name
    :return: None
    '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    smooth = 1e-8
    cup = prediction[0]
    disc = prediction[1]
    cup_ent = - cup * np.log(cup + smooth)
    disc_ent = - disc * np.log(disc + smooth)
    cup_ent = normalize_ent(cup_ent)
    disc_ent = normalize_ent(disc_ent)
    disc = construct_color_img(disc_ent)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup_ent)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)


def draw_mask(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    cup = prediction[0]
    disc = prediction[1]

    disc = construct_color_img(disc)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)

def draw_boundary(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'boundary')):
        os.makedirs(os.path.join(save_root, 'boundary'))
    boundary = prediction[0]
    boundary = construct_color_img(boundary)
    cv2.imwrite(os.path.join(save_root, 'boundary', name.split('.')[0]) + '.png', boundary)


def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def postprocessing(prediction, threshold=0.75, dataset='G'):
    if dataset[0] == 'D':
        prediction = prediction.numpy()
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        disc_mask = (disc_mask > 0.5)  # return binary mask
        cup_mask = (cup_mask > 0.1)  # return binary mask
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        for i in range(5):
            disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
            cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy
    else:
        prediction = prediction.numpy()
        prediction = (prediction > threshold)  # return binary mask
        prediction = prediction.astype(np.uint8)
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        for i in range(5):
            disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
            cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy


def joint_val_image(image, prediction, mask):
    ratio = 0.5
    _pred_cup = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _pred_disc = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _mask = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    image = np.transpose(image, (1, 2, 0))

    _pred_cup[:, :, 0] = prediction[0]
    _pred_cup[:, :, 1] = prediction[0]
    _pred_cup[:, :, 2] = prediction[0]
    _pred_disc[:, :, 0] = prediction[1]
    _pred_disc[:, :, 1] = prediction[1]
    _pred_disc[:, :, 2] = prediction[1]
    _mask[:,:,0] = mask[0]
    _mask[:,:,1] = mask[1]

    pred_cup = np.add(ratio * image, (1 - ratio) * _pred_cup)
    pred_disc = np.add(ratio * image, (1 - ratio) * _pred_disc)
    mask_img = np.add(ratio * image, (1 - ratio) * _mask)

    joint_img = np.concatenate([image, mask_img, pred_cup, pred_disc], axis=1)
    return joint_img


def save_val_img(path, epoch, img):
    name = osp.join(path, "visualization", "epoch_%d.png" % epoch)
    out = osp.join(path, "visualization")
    if not osp.exists(out):
        os.makedirs(out)
    img_shape = img[0].shape
    stack_image = np.zeros([len(img) * img_shape[0], img_shape[1], img_shape[2]])
    for i in range(len(img)):
        stack_image[i * img_shape[0] : (i + 1) * img_shape[0], :, : ] = img[i]
    imsave(name, stack_image)




def save_per_img(patch_image, data_save_path, img_name, prob_map, mask_path=None, ext="bmp"):
    path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')
    path0 = os.path.join(data_save_path, 'original_image', img_name.split('.')[0]+'.png')
    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    disc_map = prob_map[0]
    cup_map = prob_map[1]
    size = disc_map.shape
    disc_map[:, 0] = np.zeros(size[0])
    disc_map[:, size[1] - 1] = np.zeros(size[0])
    disc_map[0, :] = np.zeros(size[1])
    disc_map[size[0] - 1, :] = np.zeros(size[1])
    size = cup_map.shape
    cup_map[:, 0] = np.zeros(size[0])
    cup_map[:, size[1] - 1] = np.zeros(size[0])
    cup_map[0, :] = np.zeros(size[1])
    cup_map[size[0] - 1, :] = np.zeros(size[1])

    disc_mask = (disc_map > 0.75) # return binary mask
    cup_mask = (cup_map > 0.75)
    disc_mask = disc_mask.astype(np.uint8)
    cup_mask = cup_mask.astype(np.uint8)

    for i in range(5):
        disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
    disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8) # return 0,1
    cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8) # return 0,1
    disc_mask = get_largest_fillhole(disc_mask)
    cup_mask = get_largest_fillhole(cup_mask)

    disc_mask = morphology.binary_dilation(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
    cup_mask = morphology.binary_dilation(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1

    disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8) # return 0,1
    cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)


    contours_disc = measure.find_contours(disc_mask, 0.5)
    contours_cup = measure.find_contours(cup_mask, 0.5)

    patch_image2 = patch_image.astype(np.uint8)
    patch_image2 = Image.fromarray(patch_image2)

    patch_image2.save(path0)

    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]

    patch_image = patch_image.astype(np.uint8)
    patch_image = Image.fromarray(patch_image)

    patch_image.save(path1)

def untransform(img, lt):
    img = (img + 1) * 127.5
    lt = lt * 128
    return img, lt