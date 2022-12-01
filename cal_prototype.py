# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import argparse
import os
import os.path as osp
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
# from scipy.misc import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
from networks.deeplabv3 import *
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--model-file', type=str,
                default='/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/beal/logs/Drishti-GS/beal/checkpoint_400.pth.tar',
                    help='Model path')
parser.add_argument('-g', '--gpu', type=int, default=3)

parser.add_argument(
    '--data-dir',
    default='/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/beal/Fundus/',
    help='data root path'
)
parser.add_argument(
    '--datasetT', type=str, default='Drishti-GS', help='refuge / Drishti-GS/ RIM-ONE_r3'
)
parser.add_argument(
    '--out-stride',
    type=int,
    default=16,
    help='out-stride of deeplabv3+',
)
parser.add_argument(
    '--method', type=str, default='beal', help='using method'
)
parser.add_argument(
    '--sync-bn',
    type=bool,
    default=False,
    help='sync-bn in deeplabv3+',
)
parser.add_argument(
    '--freeze-bn',
    type=bool,
    default=False,
    help='freeze batch normalization of deeplabv3+',
)
args = parser.parse_args()
if not os.path.exists('./prototype'):
    os.makedirs('./prototype')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
model_file = args.model_file


def calculate_mean_vector(feat_cls, hard_label):

    scale_factor = F.adaptive_avg_pool2d(hard_label, 1) ##每一类每一个样本有多少个b*num
    vectors = []
    for n in range(feat_cls.size()[0]):
        if scale_factor[n].item()==0:
            continue
        if (hard_label[n] > 0).sum() < 10:
            continue
        s = feat_cls[n] * hard_label[n]
        # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
        #     continue
        s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n]
        # self.update_cls_feature(vector=s, id=t)
        vectors.append(s)
    return vectors
# 1. dataset
composed_transforms_test = transforms.Compose([
    tr.Normalize_tf(),
    tr.ToTensor()
])
db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='train',
                                transform=composed_transforms_test)

test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=6)

# 2. model
sync_bn = True

model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                        sync_bn=sync_bn, freeze_bn=args.freeze_bn,method=args.method).cuda()

if torch.cuda.is_available():
    model = model.cuda()
print('==> Loading %s model file: %s' %
      (model.__class__.__name__, model_file))
checkpoint = torch.load(model_file)
try:
    model.load_state_dict(model_data)
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model_gen.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model_gen.load_state_dict(model_dict)

except Exception:
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# begin training
prototype_x_disc_num = 0
prototype_x_cup_num = 0
prototype_x_bu_num = 0

for batch_idx, sampleT in tqdm.tqdm(
        enumerate(test_loader), total=len(test_loader)):

    imageT = sampleT['image'].cuda()
    target_map = sampleT['map'].cuda()
    target_boundary = sampleT['boundary'].cuda()

    with torch.no_grad():
        oT, boundary_T, feature_T, x_bu_feature_T, x_feature_T, o_before_T \
            , boundary_before_T = model(imageT)
        #vectors, ids = class_features.calculate_mean_vector_by_output(feat_cls, output, model)
        pred_oT = torch.sigmoid(o_before_T).clone()
        proj_query_x_disc = (pred_oT[:,1] > 0.5)  # return binary mask
        proj_query_x_cup = (pred_oT[:,0] > 0.1)  # return binary mask
        # pred_oT[pred_oT > 0.75] = 1
        # pred_oT[pred_oT <= 0.75] = 0
        bu_t = torch.sigmoid(boundary_before_T).clone()
        bu_t[bu_t > 0.5] = 1
        bu_t[bu_t <= 0.5] = 0
        #################################################################
        # pred_oS = F.interpolate(target_map.clone(),
        #                         size=oS_before.size()[2:], mode='bilinear', align_corners=True)
        # bu_s = F.interpolate(target_boundary.clone(),
        #                      size=boundaryS_before.size()[2:], mode='bilinear', align_corners=True)
        b, C, h, w = x_bu_feature_T.size()
        proj_key_x_bu = x_bu_feature_T.view(b, C, -1).permute(0, 2, 1)  ##torch.Size([1, 128*128, 304])
        proj_query_x_bu = bu_t.view(b, 1, -1)  ##torch.Size([1, 1, 128*128])
        prototype_x_bu = torch.bmm(proj_query_x_bu, proj_key_x_bu)  ##torch.Size([1, 1, 304])
        prototype_x_bu = prototype_x_bu / (torch.sum(proj_query_x_bu, dim=2, keepdim=True) + 1)
        prototype_x_bu = torch.mean(prototype_x_bu, dim=0)
        b, C, h, w = x_feature_T.size()
        proj_key_x_cup = x_feature_T.view(b, C, -1).permute(0, 2, 1)  ##torch.Size([1, 128*128, 305])

        proj_query_x_cup = proj_query_x_cup.view(b, 1, -1)  ##torch.Size([1, 1, 128*128])
        prototype_x_cup = torch.bmm(proj_query_x_cup, proj_key_x_cup)  ##torch.Size([1, 1, 305])
        prototype_x_cup = prototype_x_cup / (torch.sum(proj_query_x_cup, dim=2, keepdim=True) + 1)
        prototype_x_cup = torch.mean(prototype_x_cup, dim=0)

        proj_key_x_disc = x_feature_T.view(b, C, -1).permute(0, 2, 1)  ##torch.Size([1, 128*128, 305])

        proj_query_x_disc = proj_query_x_disc.view(b, 1, -1)  ##torch.Size([1, 1, 128*128])
        prototype_x_disc = torch.bmm(proj_query_x_disc, proj_key_x_disc)  ##torch.Size([1, 1, 305])
        prototype_x_disc = prototype_x_disc / (torch.sum(proj_query_x_disc, dim=2, keepdim=True) + 1)
        prototype_x_disc = torch.mean(prototype_x_disc, dim=0)

        prototype_x_bu = prototype_x_bu * prototype_x_bu_num + prototype_x_bu.squeeze()
        prototype_x_bu_num += 1
        prototype_x_bu = prototype_x_bu / prototype_x_bu_num
        prototype_x_bu_num = min(prototype_x_bu_num, 3000)

        prototype_x_cup = prototype_x_cup * prototype_x_cup_num + prototype_x_cup.squeeze()
        prototype_x_cup_num += 1
        prototype_x_cup = prototype_x_cup / prototype_x_cup_num
        prototype_x_cup_num = min(prototype_x_cup_num, 3000)

        prototype_x_disc = prototype_x_disc * prototype_x_disc_num + prototype_x_disc.squeeze()
        prototype_x_disc_num += 1
        prototype_x_disc = prototype_x_disc / prototype_x_disc_num
        prototype_x_disc_num = min(prototype_x_disc_num, 3000)

objective_vectors = {'bu': prototype_x_bu, 'cup':prototype_x_cup, 'disc':prototype_x_disc}

save_path = os.path.join('./prototype', "prototypes_on_{}_from_{}".format(args.datasetT, args.method))
torch.save(objective_vectors, save_path)

