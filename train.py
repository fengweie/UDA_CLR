from datetime import datetime
import os
import os.path as osp

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer, Trainer_baseline, Trainer_beal, Trainer_posal, Trainer_baseline_wob \
    , Trainer_prototype, Trainer_prototype_full

# Custom includes
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *

from networks.GAN import BoundaryDiscriminator, UncertaintyDiscriminator

here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=3, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')

    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--method', type=str, default='prototype', help='using method'
    )
    parser.add_argument(
        '--datasetS', type=str, default='refuge', help='test folder id contain images ROIs to test'
    )
    parser.add_argument(
        '--datasetT', type=str, default='Drishti-GS', help='refuge / Drishti-GS/ RIM-ONE_r3'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8, help='batch size for training the model'
    )
    parser.add_argument(
        '--group-num', type=int, default=1, help='group number for group normalization'
    )
    parser.add_argument(
        '--max-epoch', type=int, default=500, help='max epoch'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=500, help='stop epoch'
    )
    parser.add_argument(
        '--warmup-epoch', type=int, default=25, help='warmup epoch begin train GAN'
    )

    parser.add_argument(
        '--interval-validate', type=int, default=10, help='interval epoch number to valide the model'
    )
    parser.add_argument(
        '--lr-gen', type=float, default=1e-3, help='learning rate',
    )
    parser.add_argument(
        '--lr-dis', type=float, default=2.5e-5, help='learning rate',
    )
    parser.add_argument(
        '--lr-decrease-rate', type=float, default=0.1, help='ratio multiplied to initial lr',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--data-dir',
        default='/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/beal/Fundus/',
        help='data root path'
    )
    parser.add_argument(
        '--pretrained-model',
        default='../../../models/pytorch/fcn16s_from_caffe.pth',
        help='pretrained model of FCN16s',
    )
    parser.add_argument(
        '--out-stride',
        type=int,
        default=16,
        help='out-stride of deeplabv3+',
    )
    parser.add_argument(
        '--freeze-bn',
        type=bool,
        default=False,
        help='freeze batch normalization of deeplabv3+',
    )
    parser.add_argument(
        '--use_TN'
        , dest='use_TN', action='store_true',
        help='use_TN in deeplabv3+',
    )
    parser.add_argument(
        '--use_fix_initial'
        , dest='use_fix_initial', action='store_true',
        help='whether use_fix_initial',
    )
    parser.add_argument(
        '--use_pid'
        , dest='use_pid', action='store_true',
        help='whether use_pid',
    )
    parser.add_argument(
        '--retrify_pesudo'
        , dest='retrify_pesudo', action='store_true',
        help='whether use_retrify_pesudo',
    )
    parser.add_argument('--initial_resume',
                        default='/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/beal/logs/RIM-ONE_r3/beal/checkpoint_450.pth.tar'
                        , help='initial_resume checkpoint path')
    parser.add_argument(
        '--pro_weight', type=float, default=0, help='momentum',
    )
    parser.add_argument(
        '--global_pro_weight', type=float, default=0, help='momentum',
    )

    args = parser.parse_args()

    args.model = 'FCN8s'

    now = datetime.now()
    args.out = osp.join(here, 'logs', 'small_dataset','{}_to_{}'.format(args.datasetS,args.datasetT), args.method)
    # 'EA_use_PID_{}_use_TN_{}_use_fix_initial_{}_use_retrify_{}_gpw_{}_pw{}'.format(args.use_pid,
    #                                         args.use_TN,args.use_fix_initial,args.retrify_pesudo,
    #                                         args.global_pro_weight,args.pro_weight))
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(4)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(512),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomCrop(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_S = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS, split='train',
                                     transform=composed_transforms_tr)
    domain_loaderS = DataLoader(domain_S, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    domain_T = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='train',
                                     transform=composed_transforms_tr)
    domain_loaderT = DataLoader(domain_T, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='test',
                                       transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=6,
                                   pin_memory=True)

    # 2. model
    model_gen = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                        sync_bn=not args.use_TN, freeze_bn=args.freeze_bn, method=args.method).cuda()
    ######################################################################
    if args.method == 'prototype_full':
        model_geninitial_pesudolabel = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                            sync_bn=True, freeze_bn=args.freeze_bn, method=args.method).cuda()
        print('=====> loading geninitial_pesudolabel resume model')
        print('==> Loading %s model file: %s' %
              (model_geninitial_pesudolabel.__class__.__name__, args.initial_resume))
        model_geninitial_pesudolabel_checkpoint = torch.load(args.initial_resume)
        model_geninitial_pesudolabel_pretrained_dict = model_geninitial_pesudolabel_checkpoint['model_state_dict']
        model_geninitial_pesudolabel_dict = model_geninitial_pesudolabel.state_dict()
        # 1. filter out unnecessary keys
        model_geninitial_pesudolabel_pretrained_dict = \
            {k: v for k, v in model_geninitial_pesudolabel_pretrained_dict.items() if k in model_geninitial_pesudolabel_dict}
        # 2. overwrite entries in the existing state dict
        model_geninitial_pesudolabel_dict.update(model_geninitial_pesudolabel_pretrained_dict)
        # 3. load the new state dict
        model_geninitial_pesudolabel.load_state_dict(model_geninitial_pesudolabel_dict)
        model_geninitial_pesudolabel.eval()
        ######################################################################

    model_dis = BoundaryDiscriminator().cuda()
    model_dis2 = UncertaintyDiscriminator().cuda()

    start_epoch = 0
    start_iteration = 0

    # 3. optimizer

    optim_gen = torch.optim.Adam(
        model_gen.parameters(),
        lr=args.lr_gen,
        betas=(0.9, 0.99)
    )
    optim_dis = torch.optim.SGD(
        model_dis.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optim_dis2 = torch.optim.SGD(
        model_dis2.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.resume:
        print('=====> loading resume model')
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model_gen.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model_gen.load_state_dict(model_dict)

        pretrained_dict = checkpoint['model_dis_state_dict']
        model_dict = model_dis.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_dis.load_state_dict(model_dict)

        pretrained_dict = checkpoint['model_dis2_state_dict']
        model_dict = model_dis2.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_dis2.load_state_dict(model_dict)

        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration'] + 1
        optim_gen.load_state_dict(checkpoint['optim_state_dict'])
        optim_dis.load_state_dict(checkpoint['optim_dis_state_dict'])
        optim_dis2.load_state_dict(checkpoint['optim_dis2_state_dict'])
    print("=====>method is:", args.method)
    if args.method == 'baseline':
        trainer = Trainer_baseline.Trainer(
            cuda=cuda,
            model_gen=model_gen,
            optimizer_gen=optim_gen,
            lr_gen=args.lr_gen,
            lr_decrease_rate=args.lr_decrease_rate,
            val_loader=domain_loader_val,
            domain_loaderS=domain_loaderS,
            domain_loaderT=domain_loaderT,
            out=args.out,
            max_epoch=args.max_epoch,
            stop_epoch=args.stop_epoch,
            interval_validate=args.interval_validate,
            batch_size=args.batch_size,
            warmup_epoch=args.warmup_epoch,
        )
    elif args.method == 'prototype_full':
        trainer = Trainer_prototype_full.Trainer(
            cuda=cuda,
            model_gen=model_gen,
            model_geninitial_pesudolabel=model_geninitial_pesudolabel,
            model_dis=model_dis,
            model_uncertainty_dis=model_dis2,
            optimizer_gen=optim_gen,
            optimizer_dis=optim_dis,
            optimizer_uncertainty_dis=optim_dis2,
            lr_gen=args.lr_gen,
            lr_dis=args.lr_dis,
            lr_decrease_rate=args.lr_decrease_rate,
            val_loader=domain_loader_val,
            domain_loaderS=domain_loaderS,
            domain_loaderT=domain_loaderT,
            out=args.out,
            max_epoch=args.max_epoch,
            stop_epoch=args.stop_epoch,
            interval_validate=args.interval_validate,
            batch_size=args.batch_size,
            warmup_epoch=args.warmup_epoch,
            target_name=args.datasetT,
            use_fix_initial=args.use_fix_initial,
            use_pid=args.use_pid,
            use_TN=args.use_TN,
            retrify_pesudo=args.retrify_pesudo,
            global_pro_weight=args.global_pro_weight,
            pro_weight=args.pro_weight,
        )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
