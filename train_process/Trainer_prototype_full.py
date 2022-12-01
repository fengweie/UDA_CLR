from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid

import pytz

import torch.nn.functional as F

from tensorboardX import SummaryWriter
import math
import tqdm
import socket
from utils.metrics import *
from utils.Utils import *
import torch.nn as nn
bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer(object):

    def __init__(self, cuda, model_gen, model_dis, model_uncertainty_dis, optimizer_gen, optimizer_dis, optimizer_uncertainty_dis,
                 val_loader, domain_loaderS, domain_loaderT, out, max_epoch,use_global,use_pid, retrify_pesudo,
                 global_pro_weight,pro_weight, stop_epoch=None,
                 lr_gen=1e-3, lr_dis=1e-3, lr_decrease_rate=0.1, interval_validate=None, batch_size=8, warmup_epoch=25,
                 target_name='Drishti-GS'):
        self.First_src = True
        self.First = True
        self.target_name = target_name
        self.use_global = use_global
        self.use_pid = use_pid
        self.retrify_pesudo = retrify_pesudo
        self.global_pro_weight =global_pro_weight
        self.pro_weight = pro_weight
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
    def update_pro(self, centroid_0_obj,global_centroid_0_obj, name='moving_average'):
        # if centroid_0_obj.sum().item() == 0:
        #     return
        if name == 'moving_average':
            global_pro_weight = self.global_pro_weight
            # global_pro_weight=torch.cosine_similarity(centroid_0_obj.reshape(-1,1),global_centroid_0_obj.reshape(-1,1),dim=0)
            # print(global_pro_weight,global_pro_weight.shape)
            global_centroid_0_obj = global_centroid_0_obj * (1 - global_pro_weight) + global_pro_weight * centroid_0_obj
            # print(global_centroid_0_obj.shape)
        return global_centroid_0_obj
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
        if self.use_pid:
            self.running_intra = 0.0
            self.running_inter = 0.0
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

            oT, boundaryT, _, _, xt_feature, oT_before, boundaryT_before = self.model_gen(imageT)
            oS, boundaryS, _, _, xs_feature, oS_before, boundaryS_before = self.model_gen(imageS)
            # if self.use_fix_initial:
            #     _, _, _, _, _, oT_initial_pesudolabel, boundaryT_initial_pesudolabel = \
            #         self.model_geninitial_pesudolabel(imageT)
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
            if self.use_pid:
                if self.epoch > self.warmup_epoch:
                    pred_oS = F.interpolate(target_map.clone(),
                                            size=oS_before.size()[2:], mode='nearest')
                    current_sourcecentroid_0_obj, current_sourcecentroid_1_obj, current_sourcecentroid_0_bck, \
                    current_sourcecentroid_1_bck = \
                        gen_prototype(pred_oS, xs_feature)
                    if self.use_global:
                        if self.First_src:
                            sourcecentroid_0_obj = current_sourcecentroid_0_obj
                            sourcecentroid_1_obj = current_sourcecentroid_1_obj
                            sourcecentroid_0_bck = current_sourcecentroid_0_bck
                            sourcecentroid_1_bck = current_sourcecentroid_1_bck
                            self.sourcecentroid_0_obj = sourcecentroid_0_obj.detach()
                            self.sourcecentroid_1_obj = sourcecentroid_1_obj.detach()
                            self.sourcecentroid_0_bck = sourcecentroid_0_bck.detach()
                            self.sourcecentroid_1_bck = sourcecentroid_1_bck.detach()
                            self.First_src = False
                        else:
                            decay = self.global_pro_weight
                            sourcecentroid_0_obj = (1-decay) * self.sourcecentroid_0_obj + decay * current_sourcecentroid_0_obj
                            sourcecentroid_1_obj = (1-decay) * self.sourcecentroid_1_obj + decay * current_sourcecentroid_1_obj
                            sourcecentroid_0_bck = (1-decay) * self.sourcecentroid_0_bck + decay * current_sourcecentroid_0_bck
                            sourcecentroid_1_bck = (1-decay) * self.sourcecentroid_1_bck + decay * current_sourcecentroid_1_bck
                            self.sourcecentroid_0_obj = sourcecentroid_0_obj.detach()
                            self.sourcecentroid_1_obj = sourcecentroid_1_obj.detach()
                            self.sourcecentroid_0_bck = sourcecentroid_0_bck.detach()
                            self.sourcecentroid_1_bck = sourcecentroid_1_bck.detach()

                    ## for target domain
                    pred_oT = torch.sigmoid(oT_before)
                    T = 8
                    volume_batch_r = imageT.repeat(2, 1, 1, 1)
                    stride = volume_batch_r.shape[0] // 2
                    preds_trg = torch.zeros([stride * T, 2, imageT.shape[2], imageT.shape[3]]).cuda()
                    features_trg = torch.zeros([stride * T, 305, 128, 128]).cuda()
                    for i in range(T // 2):
                        with torch.no_grad():
                            preds_trg[2 * stride * i:2 * stride * (i + 1)], _, _, _, features_trg[2 * stride * i:2 * stride * (
                                        i + 1)], _, _ = self.model_gen(
                                volume_batch_r)
                    if self.retrify_pesudo:
                        current_targetcentroid_0_obj,current_targetcentroid_1_obj,current_targetcentroid_0_bck,\
                        current_targetcentroid_1_bck,\
                        target_std_map,mask_0,mask_1= \
                            gen_prototype_retrify(oT_before, xt_feature,preds_trg,features_trg,T, stride)
                    else:
                        current_targetcentroid_0_obj, current_targetcentroid_1_obj, current_targetcentroid_0_bck, \
                        current_targetcentroid_1_bck = \
                            gen_prototype(pred_oT, xt_feature)
                    if self.use_global:
                        if self.First:
                            targetcentroid_0_obj = current_targetcentroid_0_obj
                            targetcentroid_1_obj = current_targetcentroid_1_obj
                            targetcentroid_0_bck = current_targetcentroid_0_bck
                            targetcentroid_1_bck = current_targetcentroid_1_bck
                            self.targetcentroid_0_obj = targetcentroid_0_obj.detach()
                            self.targetcentroid_1_obj = targetcentroid_1_obj.detach()
                            self.targetcentroid_0_bck = targetcentroid_0_bck.detach()
                            self.targetcentroid_1_bck = targetcentroid_1_bck.detach()
                            self.First = False
                        else:
                            decay = self.global_pro_weight
                            targetcentroid_0_obj = (1-decay) * self.targetcentroid_0_obj + decay * current_targetcentroid_0_obj
                            targetcentroid_1_obj = (1-decay) * self.targetcentroid_1_obj + decay * current_targetcentroid_1_obj
                            targetcentroid_0_bck = (1-decay) * self.targetcentroid_0_bck + decay * current_targetcentroid_0_bck
                            targetcentroid_1_bck = (1-decay) * self.targetcentroid_1_bck + decay * current_targetcentroid_1_bck
                            self.targetcentroid_0_obj = targetcentroid_0_obj.detach()
                            self.targetcentroid_1_obj = targetcentroid_1_obj.detach()
                            self.targetcentroid_0_bck = targetcentroid_0_bck.detach()
                            self.targetcentroid_1_bck = targetcentroid_1_bck.detach()
                    # ## for source+target domain
                    # if self.retrify_pesudo:
                    #     current_src_trgcentroid_0_obj,current_src_trgcentroid_1_obj,\
                    #     current_src_trgcentroid_0_bck,current_src_trgcentroid_1_bck = \
                    #         gen_prototype_src_trg_retrify(pred_oS, xs_feature, oT_before, xt_feature ,preds_trg,features_trg,T, stride)
                    # else:
                    #     current_src_trgcentroid_0_obj, current_src_trgcentroid_1_obj, \
                    #     current_src_trgcentroid_0_bck, current_src_trgcentroid_1_bck = \
                    #         gen_prototype_src_trg(pred_oS, xs_feature, pred_oT, xt_feature)
                    # if self.use_global:
                    #     if self.epoch==0:
                    #         src_trgcentroid_0_obj = current_src_trgcentroid_0_obj
                    #         src_trgcentroid_1_obj = current_src_trgcentroid_1_obj
                    #         src_trgcentroid_0_bck = current_src_trgcentroid_0_bck
                    #         src_trgcentroid_1_bck = current_src_trgcentroid_1_bck
                    #         self.src_trgcentroid_0_obj = src_trgcentroid_0_obj.detach()
                    #         self.src_trgcentroid_1_obj = src_trgcentroid_1_obj.detach()
                    #         self.src_trgcentroid_0_bck = src_trgcentroid_0_bck.detach()
                    #         self.src_trgcentroid_1_bck = src_trgcentroid_1_bck.detach()
                    #     else:
                    #         decay = self.global_pro_weight
                    #         src_trgcentroid_0_obj = (1-decay) * self.src_trgcentroid_0_obj + decay * current_src_trgcentroid_0_obj
                    #         src_trgcentroid_1_obj = (1-decay) * self.src_trgcentroid_1_obj + decay * current_src_trgcentroid_1_obj
                    #         src_trgcentroid_0_bck = (1-decay) * self.src_trgcentroid_0_bck + decay * current_src_trgcentroid_0_bck
                    #         src_trgcentroid_1_bck = (1-decay) * self.src_trgcentroid_1_bck + decay * current_src_trgcentroid_1_bck
                    #         self.src_trgcentroid_0_obj = src_trgcentroid_0_obj.detach()
                    #         self.src_trgcentroid_1_obj = src_trgcentroid_1_obj.detach()
                    #         self.src_trgcentroid_0_bck = src_trgcentroid_0_bck.detach()
                    #         self.src_trgcentroid_1_bck = src_trgcentroid_1_bck.detach()
                    intra_obj_0_loss = mseloss(sourcecentroid_0_obj,targetcentroid_0_obj)
                                     # +mseloss(sourcecentroid_0_obj ,src_trgcentroid_0_obj)\
                                     # +mseloss(targetcentroid_0_obj ,src_trgcentroid_0_obj)
                    intra_obj_1_loss = mseloss(sourcecentroid_1_obj , targetcentroid_1_obj)
                                     # +mseloss(sourcecentroid_1_obj , src_trgcentroid_1_obj)\
                                     # +mseloss(targetcentroid_1_obj , src_trgcentroid_1_obj)

                    intra_bck_0_loss = mseloss(sourcecentroid_0_bck ,targetcentroid_0_bck)
                                     # +mseloss(sourcecentroid_0_bck , src_trgcentroid_0_bck)\
                                     # +mseloss(targetcentroid_0_bck , src_trgcentroid_0_bck)
                    intra_bck_1_loss = mseloss(sourcecentroid_1_bck , targetcentroid_1_bck)
                                     # +mseloss(sourcecentroid_1_bck , src_trgcentroid_1_bck)\
                                     # +mseloss(targetcentroid_1_bck, src_trgcentroid_1_bck)
                    intra_loss = intra_obj_0_loss+intra_obj_1_loss+intra_bck_0_loss+intra_bck_1_loss

                    inter_loss = mseloss(sourcecentroid_1_obj , sourcecentroid_1_bck)\
                                +mseloss(sourcecentroid_0_obj ,sourcecentroid_0_bck)

                    self.running_intra += intra_loss.item()
                    dis_intra_data = intra_loss.data.item()
                    self.running_inter += inter_loss.item()
                    dis_inter_data = inter_loss.data.item()

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
            if self.use_pid:
                if self.epoch > self.warmup_epoch:
                    loss_all = loss_seg+loss_adv_diff+self.pro_weight*(intra_loss)
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
                grid_image = make_grid(boundaryT[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainT/boundaryT', grid_image, iteration)
                if self.use_pid:
                    # if self.epoch > self.warmup_epoch:
                    if self.retrify_pesudo:
                        # grid_image = make_grid(cup_rectified[0,0, ...].clone().cpu().data, 1, normalize=True)
                        # self.writer.add_image('DomainT/cup_rectified', grid_image, iteration)
                        # grid_image = make_grid(disc_rectified[0,0, ...].clone().cpu().data, 1, normalize=True)
                        # self.writer.add_image('DomainT/disc_rectified', grid_image, iteration)
                        # if self.retrify_pesudo:
                        # grid_image = make_grid(bu_weights[0,0, ...].clone().cpu().data, 1, normalize=True)
                        # self.writer.add_image('DomainT/bu_weights', grid_image, iteration)
                        # grid_image = make_grid(target_cup_obj_weights[0,0, ...].clone().cpu().data, 1, normalize=True)
                        # self.writer.add_image('DomainT/target_cup_obj_weights', grid_image, iteration)
                        # grid_image = make_grid(target_cup_bck_weights[0,0, ...].clone().cpu().data, 1, normalize=True)
                        # self.writer.add_image('DomainT/target_cup_bck_weights', grid_image, iteration)
                        #
                        # grid_image = make_grid(target_disc_obj_weights[0,0, ...].clone().cpu().data, 1, normalize=True)
                        # self.writer.add_image('DomainT/target_disc_obj_weights', grid_image, iteration)
                        # grid_image = make_grid(target_disc_bck_weights[0,0, ...].clone().cpu().data, 1, normalize=True)
                        # self.writer.add_image('DomainT/target_disc_bck_weights', grid_image, iteration)
                        grid_image = make_grid(target_std_map[0,0, ...].clone().cpu().data, 1, normalize=True)
                        self.writer.add_image('DomainT/target_cup_std_map', grid_image, iteration)
                        grid_image = make_grid(target_std_map[0,1, ...].clone().cpu().data, 1, normalize=True)
                        self.writer.add_image('DomainT/target_disc_std_map', grid_image, iteration)
                        grid_image = make_grid(mask_0[0,0, ...].clone().cpu().data, 1, normalize=True)
                        self.writer.add_image('DomainT/mask_0', grid_image, iteration)
                        grid_image = make_grid(mask_1[0,0, ...].clone().cpu().data, 1, normalize=True)
                        self.writer.add_image('DomainT/mask_1', grid_image, iteration)

            self.writer.add_scalar('train_adv/loss_adv_diff', loss_adv_diff_data, iteration)
            self.writer.add_scalar('train_dis/loss_D_same', loss_D_same_data, iteration)
            self.writer.add_scalar('train_dis/loss_D_diff', loss_D_diff_data, iteration)
            if self.use_pid:
                if self.epoch > self.warmup_epoch:
                    # self.writer.add_scalar('train_pro/loss_bu', dis_bu_data, iteration)
                    self.writer.add_scalar('train_pro/loss_intra', dis_intra_data, iteration)
                    self.writer.add_scalar('train_pro/loss_inter', dis_inter_data, iteration)
                    # self.writer.add_scalar('train_pro/disc_threshold', disc_threshold, iteration)
                    # self.writer.add_scalar('train_pro/cup_threshold', cup_threshold, iteration)


            self.writer.add_scalar('train_gen/loss_seg', loss_seg_data, iteration)
            if self.use_pid:
                if self.epoch > self.warmup_epoch:
                    metrics.append((loss_seg_data, loss_adv_diff_data, loss_D_same_data, loss_D_diff_data
                                    ,dis_intra_data, dis_inter_data))
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
        if self.use_pid:
            if self.epoch > self.warmup_epoch:
                self.running_intra /= len(self.domain_loaderS)
                self.running_inter /= len(self.domain_loaderS)
        stop_time = timeit.default_timer()
        if self.use_pid:
            if self.epoch > self.warmup_epoch:
                print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, '
                      ' Average advLoss: %f, Average dis_same_Loss: %f, '
                      'Average dis_diff_Lyoss: %f,'
                     ' Average intra_Loss: %f, '
                      'Average inter_Loss: %f,'
                      'Execution time: %.5f' %
                      (self.epoch, get_lr(self.optim_gen), self.running_seg_loss,
                       self.running_adv_diff_loss,
                       self.running_dis_same_loss, self.running_dis_diff_loss,
                       self.running_intra, self.running_inter
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



