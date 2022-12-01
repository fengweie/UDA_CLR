import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import BatchNorm2d
from networks.aspp import build_aspp
from networks.decoder import build_decoder
from networks.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, method='prototype'):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            print("=====================================>使用batchnorm")
            # BatchNorm = SynchronizedBatchNorm2d
            BatchNorm = nn.BatchNorm2d
        else:
            print("=====================================>使用transnorm")
            BatchNorm = BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, method, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input) ## x:torch.Size([8, 320, 32, 32]),torch.Size([8, 24, 128, 128])
        x = self.aspp(x) ##torch.Size([8, 1280, 32, 32])
        feature = x

        x1_before, x2_before, x_bu_feature, x_feature = self.decoder(x, low_level_feat)

        x2 = F.interpolate(x2_before, size=input.size()[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(x1_before, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x1, x2, feature, x_bu_feature, x_feature,x1_before, x2_before

    def freeze_bn(self):
        for m in self.modules():
            # if isinstance(m, SynchronizedBatchNorm2d):
            #     m.eval()
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            elif isinstance(m, BatchNorm2d):
                m.eval()
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


