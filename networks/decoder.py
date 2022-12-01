import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import BatchNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, method, BatchNorm):
        super(Decoder, self).__init__()
        self.method = method
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            # nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            #                            BatchNorm(256),
            #                            nn.ReLU(),
            #                            nn.Dropout(0.5),
            #                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(305),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(305, num_classes, kernel_size=1, stride=1))
        self.last_conv_boundary = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, 1, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)  ##torch.Size([8, 24, 128, 128])>>torch.Size([8, 48, 128, 128])
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)##torch.Size([8, 256, 32, 32])==>torch.Size([8, 256, 128, 128])
        x_bu_feature = torch.cat((x, low_level_feat), dim=1)##torch.Size([8, 304, 128, 128])
        boundary = self.last_conv_boundary(x_bu_feature)  ##torch.Size([8, 1, 128, 128])
        x_feature = torch.cat([x_bu_feature, boundary], 1)  #torch.Size([8, 305, 128, 128])
        x1 = self.last_conv(x_feature)  #torch.Size([8, 2, 128, 128])
        # if self.method=='prototype':
        return x1, boundary, x_bu_feature, x_feature
        # else:
        #     return x1, boundary

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def build_decoder(num_classes, backbone, method, BatchNorm):
    return Decoder(num_classes, backbone, method, BatchNorm)
