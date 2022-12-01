from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import time

import numpy as np
import pytz
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import math
import tqdm
import socket
from utils.metrics import *
from utils.Utils import *

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer(object):

    def __init__(self, cuda, model_gen, model_dis, model_uncertainty_dis, optimizer_gen, optimizer_dis, optimizer_uncertainty_dis,
                 val_loader, domain_loaderS, domain_loaderT, out, max_epoch, stop_epoch=None,
                 lr_gen=1e-3, lr_dis=1e-3, lr_decrease_rate=0.1, interval_validate=None, batch_size=8, warmup_epoch=25,
                 target_name='Drishti-GS'):
        self.target_name = target_name
        self.cuda = cuda
        self.warmup_epoch = warmup_epoch
        self.model_gen = model_gen
        self.model_dis2 = model_uncertainty_dis
        self.model_dis = model_dis
        self.optim_gen = optimizer_gen
        self.optim_dis = optimizer_dis
        self.optim_dis2 = optimizer_uncertainty_dis
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.domain_loaderS = domain_loaderS
        self.domain_loaderT = domain_loaderT
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)
        self.objective_vectors =torch.load(
            os.path.join('/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/beal/prototype',
                         "prototypes_on_{}_from_{}".format(self.target_name, 'beal')))
        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/cup_dice',
            'train/disc_dice',
            'train/loss_adv',
            'train/loss_D_same',
            'train/loss_D_diff',
            'valid/loss_CE',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.best_epoch = -1

    def feat_prototype_distance(self, feat,prototype,class_numbers):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, class_numbers, H, W)).to(feat.device)
        for i in range(class_numbers):
            #feat_proto_distance[:, i, :, :] = torch.norm(torch.Tensor(self.objective_vectors[i]).reshape(-1,1,1).expand(-1, H, W).to(feat.device) - feat, 2, dim=1,)
            feat_proto_distance[:, i, :, :] = torch.norm(prototype.reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1,)
        return feat_proto_distance

    def get_prototype_weight(self, feat, class_num, level):
        prototype = self.objective_vectors[level]
        feat_proto_distance = self.feat_prototype_distance(feat,prototype, class_num)
        # weight = torch.cosine_similarity(prototype.reshape(-1,1),feat.reshape(C,-1),dim=0).reshape(N, class_num, H, W)
        # feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)
        #
        # feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
        # weight = F.softmax(-feat_proto_distance * 1.0, dim=1)
        # feat_proto_distance = -feat_proto_distance
        weight = (feat_proto_distance-feat_proto_distance.min())/(feat_proto_distance.max()-feat_proto_distance.min())
        return weight
    def update_objective_SingleVector(self, id, vector, name='moving_average'):
        if vector.sum().item() == 0:
            return
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (1 - 0.001) + 0.001 * vector.squeeze()
            # self.objective_vectors_num[id] += 1
            # self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)

    def validate(self):
        training = self.model_gen.training
        self.model_gen.eval()

        val_loss = 0
        val_cup_dice = 0
        val_disc_dice = 0
        val_cup_pa = 0
        val_disc_pa = 0
        val_cup_iou = 0
        val_disc_iou = 0
        metrics = []
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['image']
                target_map = sample['map']
                target_boundary = sample['boundary']
                if self.cuda:
                    data, target_map, target_boundary = data.cuda(), target_map.cuda(), target_boundary.cuda()
                with torch.no_grad():
                    predictions, boundary, feature,_,_,_,_ = self.model_gen(data)

                loss = F.binary_cross_entropy_with_logits(predictions, target_map)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
                PA_cup, PA_disc, IOU_cup, IOU_disc = pixel_acc(predictions, target_map)

                val_cup_dice += dice_cup
                val_disc_dice += dice_disc

                val_cup_pa += PA_cup
                val_disc_pa += PA_disc

                val_cup_iou += IOU_cup
                val_disc_iou += IOU_disc

            val_loss /= len(self.val_loader)
            val_cup_dice /= len(self.val_loader)
            val_disc_dice /= len(self.val_loader)
            val_disc_pa /= len(self.val_loader)
            val_cup_pa /= len(self.val_loader)
            val_cup_iou /= len(self.val_loader)
            val_disc_iou /= len(self.val_loader)
            metrics.append((val_loss, val_cup_dice, val_disc_dice))
            self.writer.add_scalar('val_data/val_CUP_PA', val_cup_pa, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_DISC_PA', val_disc_pa, self.epoch * (len(self.domain_loaderS)))

            self.writer.add_scalar('val_data/val_CUP_IOU', val_cup_iou, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_DISC_IOU', val_disc_iou, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/loss_CE', val_loss, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.domain_loaderS)))

            mean_dice = val_cup_dice + val_disc_dice
            is_best = mean_dice > self.best_mean_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model_gen.__class__.__name__,
                    'optim_state_dict': self.optim_gen.state_dict(),
                    'optim_dis_state_dict': self.optim_dis.state_dict(),
                    'optim_dis2_state_dict': self.optim_dis2.state_dict(),
                    'model_state_dict': self.model_gen.state_dict(),
                    'model_dis_state_dict': self.model_dis.state_dict(),
                    'model_dis2_state_dict': self.model_dis2.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_gen),
                    'learning_rate_dis': get_lr(self.optim_dis),
                    'learning_rate_dis2': get_lr(self.optim_dis2),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
            else:
                if (self.epoch + 1) % 50 == 0:
                    torch.save({
                        'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model_gen.__class__.__name__,
                    'optim_state_dict': self.optim_gen.state_dict(),
                    'optim_dis_state_dict': self.optim_dis.state_dict(),
                    'optim_dis2_state_dict': self.optim_dis2.state_dict(),
                    'model_state_dict': self.model_gen.state_dict(),
                    'model_dis_state_dict': self.model_dis.state_dict(),
                    'model_dis2_state_dict': self.model_dis2.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_gen),
                    'learning_rate_dis': get_lr(self.optim_dis),
                    'learning_rate_dis2': get_lr(self.optim_dis2),
                    'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))


            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] * 5 + \
                       list(metrics) + [elapsed_time] + ['best model epoch: %d' % self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.domain_loaderS)))
            if training:
                self.model_gen.train()
                self.model_dis.train()
                self.model_dis2.train()

    def adaptation_factor(self,m):
        den = 1.0 + math.exp(-0.8 * (m+1))
        lamb = 1.0 / den - 0.3
        return lamb
    def train_epoch(self):
        source_domain_label = 1
        target_domain_label = 0
        smooth = 1e-7
        self.model_gen.train()
        self.model_dis.train()
        self.model_dis2.train()
        self.running_seg_loss = 0.0
        self.running_adv_loss = 0.0
        self.running_dis_diff_loss = 0.0
        self.running_dis_same_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0
        self.running_dis_bu = 0.0
        self.running_dis_cup = 0.0
        self.running_dis_disc = 0.0
        loss_adv_diff_data = 0
        loss_D_same_data = 0
        loss_D_diff_data = 0

        domain_t_loader = enumerate(self.domain_loaderT)
        start_time = timeit.default_timer()
        for batch_idx, sampleS in tqdm.tqdm(
                enumerate(self.domain_loaderS), total=len(self.domain_loaderS),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            metrics = []

            iteration = batch_idx + self.epoch * len(self.domain_loaderS)
            self.iteration = iteration

            assert self.model_gen.training
            assert self.model_dis.training
            assert self.model_dis2.training

            self.optim_gen.zero_grad()
            self.optim_dis.zero_grad()
            self.optim_dis2.zero_grad()

            # 1. train generator with random images
            for param in self.model_dis.parameters():
                param.requires_grad = False
            for param in self.model_dis2.parameters():
                param.requires_grad = False
            for param in self.model_gen.parameters():
                param.requires_grad = True

            imageS = sampleS['image'].cuda()
            target_map = sampleS['map'].cuda()
            target_boundary = sampleS['boundary'].cuda()

            # if self.epoch > self.warmup_epoch:
            # # 2. train generator with images from different domain
            try:
                id_, sampleT = next(domain_t_loader)
            except:
                domain_t_loader = enumerate(self.domain_loaderT)
                id_, sampleT = next(domain_t_loader)

            imageT = sampleT['image'].cuda()

            images_all = torch.cat((imageS, imageT), 0)
            o_all, boundary_all, feature_all, x_bu_feature_all, x_feature_all, o_before_all\
                , boundary_before_all  = self.model_gen(images_all)

            oS, oT = o_all[:self.batch_size], o_all[self.batch_size:]
            boundaryS, boundaryT = boundary_all[:self.batch_size], boundary_all[self.batch_size:]
            featureS, featureT = feature_all[:self.batch_size], feature_all[self.batch_size:]

            xs_bu_feature, xt_bu_feature = x_bu_feature_all[:self.batch_size], x_bu_feature_all[self.batch_size:]
            xs_feature, xt_feature = x_feature_all[:self.batch_size], x_feature_all[self.batch_size:]
            oS_before, oT_before = o_before_all[:self.batch_size], o_before_all[self.batch_size:]
            boundaryS_before, boundaryT_before = boundary_before_all[:self.batch_size], boundary_before_all[self.batch_size:]


            # oT, boundaryT, featureT, xt_bu_feature, xt_feature, oT_before, boundaryT_before = self.model_gen(imageT)
            # oS, boundaryS, featureS, xs_bu_feature, xs_feature, oS_before, boundaryS_before = self.model_gen(imageS)

            loss_seg1 = bceloss(torch.sigmoid(oS), target_map)
            loss_seg2 = mseloss(torch.sigmoid(boundaryS), target_boundary)
            loss_seg = loss_seg1 + loss_seg2

            self.running_seg_loss += loss_seg.item()
            loss_seg_data = loss_seg.data.item()
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')

            # cup_dice, disc_dice = dice_coeff_2label(oS, target_map)

            # loss_seg.backward()
            # self.optim_gen.step()

            # write image log
            if iteration % 30 == 0:
                grid_image = make_grid(
                    imageS[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/image', grid_image, iteration)
                grid_image = make_grid(
                    target_map[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/target_cup', grid_image, iteration)
                grid_image = make_grid(
                    target_map[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/target_disc', grid_image, iteration)
                grid_image = make_grid(
                    target_boundary[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/target_boundary', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(oS)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/prediction_cup', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(oS)[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/prediction_disc', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(boundaryS)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/prediction_boundary', grid_image, iteration)

#################################################################
            if self.epoch > self.warmup_epoch:
                pred_oS = F.interpolate(target_map.clone(),
                                        size=oS_before.size()[2:], mode='bilinear', align_corners=True)
                bu_s = F.interpolate(target_boundary.clone(),
                                     size=boundaryS_before.size()[2:], mode='bilinear', align_corners=True)
                b, C, h, w = xs_bu_feature.size()
                proj_key_x_bu = xs_bu_feature.view(b, C, -1).permute(0, 2, 1)  ##torch.Size([1, 128*128, 304])
                proj_query_x_bu = bu_s.view(b, 1, -1)  ##torch.Size([1, 1, 128*128])
                prototype_x_bu = torch.bmm(proj_query_x_bu, proj_key_x_bu)  ##torch.Size([1, 1, 304])
                prototype_x_bu = prototype_x_bu/ (torch.sum(proj_query_x_bu,dim=2,keepdim=True)+1)
                prototype_x_bu = torch.mean(prototype_x_bu,dim=0)
                b, C, h, w = xs_feature.size()
                proj_key_x_cup = xs_feature.view(b, C, -1).permute(0, 2, 1)  ##torch.Size([1, 128*128, 305])
                proj_query_x_cup = pred_oS[:,0]
                proj_query_x_cup =proj_query_x_cup.view(b, 1, -1)  ##torch.Size([1, 1, 128*128])
                prototype_x_cup = torch.bmm(proj_query_x_cup, proj_key_x_cup)  ##torch.Size([1, 1, 305])
                prototype_x_cup = prototype_x_cup/ (torch.sum(proj_query_x_cup,dim=2,keepdim=True)+1)
                prototype_x_cup = torch.mean(prototype_x_cup,dim=0)

                proj_key_x_disc = xs_feature.view(b, C, -1).permute(0, 2, 1)  ##torch.Size([1, 128*128, 305])
                proj_query_x_disc = pred_oS[:,1]
                proj_query_x_disc =proj_query_x_disc.view(b, 1, -1)  ##torch.Size([1, 1, 128*128])
                prototype_x_disc = torch.bmm(proj_query_x_disc, proj_key_x_disc)  ##torch.Size([1, 1, 305])
                prototype_x_disc = prototype_x_disc/ (torch.sum(proj_query_x_disc,dim=2,keepdim=True)+1)
                prototype_x_disc = torch.mean(prototype_x_disc,dim=0)

                pred_oT = torch.sigmoid(oT_before)
                # pred_oT[pred_oT > 0.75] = 1
                # pred_oT[pred_oT <= 0.75] = 0

                # bu_t[bu_t > 0.75] = 1
                # bu_t[bu_t <= 0.75] = 0
                ##############proto_rectify
                # threshold = self.adaptation_factor(self.epoch)
                # print(threshold)
                bu_t = torch.sigmoid(boundaryT_before).clone()
                bu_weights = self.get_prototype_weight(xt_bu_feature,class_num=1,level='bu')
                # bu_rectified = bu_weights * bu_t
                bu_rectified = torch.sigmoid(boundaryT_before).clone()
                bu_threshold = self.adaptation_factor(self.epoch)
                    # torch.max(bu_rectified)-0.2
                bu_rectified[bu_rectified > bu_threshold] = 1
                bu_rectified[bu_rectified <= bu_threshold] = 0

                ###############proto_rectify
                b, C, h, w = xt_bu_feature.size()
                proj_key_y_bu = xt_bu_feature.view(b, C, -1).permute(0, 2, 1)  ##torch.Size([1, 128*128, 304])
                proj_query_y_bu = bu_rectified.view(b, 1, -1)  ##torch.Size([1, 1, 128*128])
                prototype_y_bu = torch.bmm(proj_query_y_bu, proj_key_y_bu)  ##torch.Size([1, 1, 304])
                prototype_y_bu = prototype_y_bu/ (torch.sum(proj_query_y_bu,dim=2,keepdim=True)+1)
                prototype_y_bu = torch.mean(prototype_y_bu,dim=0)
                ###############proto_rectify
                cup_weights = self.get_prototype_weight(xt_feature,class_num=1,level='cup')
                proj_query_y_cup = pred_oT[:, 0:1]
                # cup_rectified = cup_weights * proj_query_y_cup
                cup_rectified = pred_oT[:, 0:1].clone()
                if self.target_name=='Drishti-GS':
                    cup_threshold = self.adaptation_factor(self.epoch)
                        # torch.max(cup_rectified) - 0.2
                    cup_rectified[cup_rectified > cup_threshold] = 1
                    cup_rectified[cup_rectified <= cup_threshold] = 0
                elif self.target_name=='RIM-ONE_r3':
                    cup_threshold = self.adaptation_factor(self.epoch)
                        # torch.max(cup_rectified) - 0.2
                    cup_rectified[cup_rectified > cup_threshold] = 1
                    cup_rectified[cup_rectified <= cup_threshold] = 0
                ###############proto_rectify

                b, C, h, w = xt_feature.size()
                proj_key_y_cup = xt_feature.view(b, C, -1).permute(0, 2, 1)  ##torch.Size([1, 128*128, 305])
                proj_query_y_cup = cup_rectified
                proj_query_y_cup = proj_query_y_cup.view(b, 1, -1)  ##torch.Size([1, 1, 128*128])
                prototype_y_cup = torch.bmm(proj_query_y_cup, proj_key_y_cup)  ##torch.Size([1, 1, 305])
                prototype_y_cup = prototype_y_cup/ (torch.sum(proj_query_y_cup,dim=2,keepdim=True)+1)
                prototype_y_cup = torch.mean(prototype_y_cup,dim=0)
                ###############proto_rectify
                disc_weights = self.get_prototype_weight(xt_feature, class_num=1, level='disc')
                proj_query_y_disc = pred_oT[:, 1:]
                # disc_rectified = disc_weights * proj_query_y_disc
                disc_rectified = pred_oT[:, 1:].clone()

                disc_threshold = self.adaptation_factor(self.epoch)
                    # torch.max(disc_rectified) - 0.2
                disc_rectified[disc_rectified > disc_threshold] = 1
                disc_rectified[disc_rectified <= disc_threshold] = 0

                ###############proto_rectify
                proj_key_y_disc = xt_feature.view(b, C, -1).permute(0, 2, 1)  ##torch.Size([1, 128*128, 305])
                proj_query_y_disc = disc_rectified
                proj_query_y_disc = proj_query_y_disc.view(b, 1, -1)  ##torch.Size([1, 1, 128*128])
                prototype_y_disc = torch.bmm(proj_query_y_disc, proj_key_y_disc) ##torch.Size([1, 1, 305])
                prototype_y_disc = prototype_y_disc/ (torch.sum(proj_query_y_disc,dim=2,keepdim=True)+1)
                prototype_y_disc = torch.mean(prototype_y_disc,dim=0)

                dis_bu = torch.mean(torch.pow((prototype_x_bu-prototype_y_bu),2))
                dis_cup = torch.mean(torch.pow((prototype_x_cup - prototype_y_cup), 2))
                dis_disc = torch.mean(torch.pow((prototype_x_disc - prototype_y_disc), 2))
                self.running_dis_bu += dis_bu.item()
                dis_bu_data = dis_bu.data.item()
                self.running_dis_cup += dis_cup.item()
                dis_cup_data = dis_cup.data.item()
                self.running_dis_disc += dis_disc.item()
                dis_disc_data = dis_disc.data.item()

###################################################
            uncertainty_mapT = -1.0 * torch.sigmoid(oT) * torch.log(torch.sigmoid(oT) + smooth)
            D_out2 = self.model_dis(torch.sigmoid(boundaryT))
            D_out1 = self.model_dis2(uncertainty_mapT)

            loss_adv_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_domain_label).cuda())
            loss_adv_diff2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_domain_label).cuda())
            loss_adv_diff = 0.01 * (loss_adv_diff1 + loss_adv_diff2)
            self.running_adv_diff_loss += loss_adv_diff.item()
            loss_adv_diff_data = loss_adv_diff.data.item()
            if np.isnan(loss_adv_diff_data):
                raise ValueError('loss_adv_diff_data is nan while training')
            if self.epoch > self.warmup_epoch:
                loss_all = loss_seg+loss_adv_diff+0.05*(dis_disc+dis_cup+dis_bu)
            else:
                loss_all = loss_seg + loss_adv_diff
            loss_all.backward()
            self.optim_gen.step()

            # 3. train discriminator with images from same domain
            for param in self.model_dis.parameters():
                param.requires_grad = True
            for param in self.model_dis2.parameters():
                param.requires_grad = True
            for param in self.model_gen.parameters():
                param.requires_grad = False

            boundaryS = boundaryS.detach()
            oS = oS.detach()
            uncertainty_mapS = -1.0 * torch.sigmoid(oS) * torch.log(torch.sigmoid(oS) + smooth)
            D_out2 = self.model_dis(torch.sigmoid(boundaryS))
            D_out1 = self.model_dis2(uncertainty_mapS)

            loss_D_same1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                source_domain_label).cuda())
            loss_D_same2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                source_domain_label).cuda())
            loss_D_same = loss_D_same1+loss_D_same2

            self.running_dis_same_loss += loss_D_same.item()
            loss_D_same_data = loss_D_same.data.item()
            if np.isnan(loss_D_same_data):
                raise ValueError('loss is nan while training')
            loss_D_same.backward()

            # 4. train discriminator with images from different domain
            boundaryT = boundaryT.detach()
            oT = oT.detach()
            uncertainty_mapT = -1.0 * torch.sigmoid(oT) * torch.log(torch.sigmoid(oT) + smooth)
            D_out2 = self.model_dis(torch.sigmoid(boundaryT))
            D_out1 = self.model_dis2(uncertainty_mapT)

            loss_D_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                target_domain_label).cuda())
            loss_D_diff2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                target_domain_label).cuda())
            loss_D_diff = loss_D_diff1 + loss_D_diff2
            self.running_dis_diff_loss += loss_D_diff.item()
            loss_D_diff_data = loss_D_diff.data.item()
            if np.isnan(loss_D_diff_data):
                raise ValueError('loss is nan while training')
            loss_D_diff.backward()

            # 5. update parameters
            self.optim_dis.step()
            self.optim_dis2.step()
            if self.epoch > self.warmup_epoch:
                # update prototype
                self.update_objective_SingleVector('bu', prototype_y_bu.detach(), name='moving_average')
                self.update_objective_SingleVector('cup', prototype_y_cup.detach(), name='moving_average')
                self.update_objective_SingleVector('disc', prototype_y_disc.detach(), name='moving_average')
            if iteration % 30 == 0:
                grid_image = make_grid(
                    imageT[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainT/image', grid_image, iteration)
                grid_image = make_grid(
                    sampleT['map'][0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainT/target_cup', grid_image, iteration)
                grid_image = make_grid(
                    sampleT['map'][0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainT/target_disc', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(oT)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainT/prediction_cup', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(oT)[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainT/prediction_disc', grid_image, iteration)
                grid_image = make_grid(boundaryS[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/boundaryS', grid_image, iteration)
                grid_image = make_grid(boundaryT[0, 0, ...].clone().cpu().data, 1,
                                       normalize=True)
                self.writer.add_image('DomainT/boundaryT', grid_image, iteration)
                if self.epoch > self.warmup_epoch:
                    grid_image = make_grid(torch.sigmoid(boundaryT_before)[0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainT/bu_origin', grid_image, iteration)
                    grid_image = make_grid(pred_oT[:, 0:1][0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainT/cup_origin', grid_image, iteration)
                    grid_image = make_grid(pred_oT[:, 1:][0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainT/disc_origin', grid_image, iteration)

                    grid_image = make_grid(bu_rectified[0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainT/bu_rectified', grid_image, iteration)
                    grid_image = make_grid(cup_rectified[0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainT/cup_rectified', grid_image, iteration)
                    grid_image = make_grid(disc_rectified[0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainT/disc_rectified', grid_image, iteration)

                    grid_image = make_grid(bu_weights[0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainT/bu_weights', grid_image, iteration)
                    grid_image = make_grid(cup_weights[0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainT/cup_weights', grid_image, iteration)
                    grid_image = make_grid(disc_weights[0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainT/disc_weights', grid_image, iteration)


                    grid_image = make_grid(bu_s[0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/bu_S', grid_image, iteration)
                    grid_image = make_grid(pred_oS[0,0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/cup_S', grid_image, iteration)
                    grid_image = make_grid(pred_oS[0,1, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/disc_S', grid_image, iteration)

            self.writer.add_scalar('train_adv/loss_adv_diff', loss_adv_diff_data, iteration)
            self.writer.add_scalar('train_dis/loss_D_same', loss_D_same_data, iteration)
            self.writer.add_scalar('train_dis/loss_D_diff', loss_D_diff_data, iteration)
            if self.epoch > self.warmup_epoch:
                self.writer.add_scalar('train_pro/loss_bu', dis_bu_data, iteration)
                self.writer.add_scalar('train_pro/loss_cup', dis_cup_data, iteration)
                self.writer.add_scalar('train_pro/loss_disc', dis_disc_data, iteration)
                self.writer.add_scalar('train_pro/disc_threshold', disc_threshold, iteration)
                self.writer.add_scalar('train_pro/cup_threshold', cup_threshold, iteration)
                self.writer.add_scalar('train_pro/bu_threshold', bu_threshold, iteration)

            self.writer.add_scalar('train_gen/loss_seg', loss_seg_data, iteration)
            if self.epoch > self.warmup_epoch:
                metrics.append((loss_seg_data, loss_adv_diff_data, loss_D_same_data, loss_D_diff_data
                                ,dis_bu_data, dis_cup_data, dis_disc_data))
            else:
                metrics.append((loss_seg_data, loss_adv_diff_data, loss_D_same_data, loss_D_diff_data
                               ))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration]  + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

        self.running_seg_loss /= len(self.domain_loaderS)
        self.running_adv_diff_loss /= len(self.domain_loaderS)
        self.running_dis_same_loss /= len(self.domain_loaderS)
        self.running_dis_diff_loss /= len(self.domain_loaderS)
        if self.epoch > self.warmup_epoch:
            self.running_dis_bu /= len(self.domain_loaderS)
            self.running_dis_cup /= len(self.domain_loaderS)
            self.running_dis_disc /= len(self.domain_loaderS)
        stop_time = timeit.default_timer()
        if self.epoch > self.warmup_epoch:
            print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, '
                  ' Average advLoss: %f, Average dis_same_Loss: %f, '
                  'Average dis_diff_Lyoss: %f,'
                 ' Average bu_loss: %f, Average cup_Loss: %f, '
                  'Average disc_Loss: %f,'
                  'Execution time: %.5f' %
                  (self.epoch, get_lr(self.optim_gen), self.running_seg_loss,
                   self.running_adv_diff_loss,
                   self.running_dis_same_loss, self.running_dis_diff_loss,
                   self.running_dis_bu, self.running_dis_cup, self.running_dis_disc
                   , stop_time - start_time))
        else:
            print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, '
                  ' Average advLoss: %f, Average dis_same_Loss: %f, '
                  'Average dis_diff_Lyoss: %f,'
                  'Execution time: %.5f' %
                  (self.epoch, get_lr(self.optim_gen), self.running_seg_loss,
                   self.running_adv_diff_loss,
                   self.running_dis_same_loss, self.running_dis_diff_loss,
                   stop_time - start_time))


    def train(self):

        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch+1) % 100 == 0:
                _lr_gen = self.lr_gen * 0.2
                for param_group in self.optim_gen.param_groups:
                    param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr_gen', get_lr(self.optim_gen), self.epoch * (len(self.domain_loaderS)))
            if (self.epoch+1) % self.interval_validate == 0:
                self.validate()
        self.writer.close()



