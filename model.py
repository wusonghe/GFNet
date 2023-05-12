import torch
from torch import nn
import math
from model_utils import *
from collections import OrderedDict


class Gate(nn.Module):
    def __init__(self, in_plane):
        super(Gate, self).__init__()
        self.gate = nn.Conv3d(in_plane, in_plane, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))

    def forward(self, rgb_fea, flow_fea):
        gate = torch.sigmoid(self.gate(torch.cat((rgb_fea, flow_fea), 2)))
        gate_fea = flow_fea * gate
        fuse = torch.cat((gate_fea, rgb_fea), 2)

        return fuse


class VideoSaliencyModel(nn.Module):
    def __init__(self, use_upsample=True, num_hier=3, num_clips=32):
        super(VideoSaliencyModel, self).__init__()

        self.backbone = BackBoneS3D()
        self.num_hier = num_hier
        self.decoder = DecoderConvUp()

    def forward(self, x):
        [y0, y1, y2, y3] = self.backbone(x)
        if self.num_hier == 0:
            return self.decoder(y0)
        if self.num_hier == 1:
            return self.decoder(y0, y1)
        if self.num_hier == 2:
            return self.decoder(y0, y1, y2)
        if self.num_hier == 3:
            return self.decoder(y0, y1, y2, y3)


class DecoderConvUp(nn.Module):
    def __init__(self):
        super(DecoderConvUp, self).__init__()
        # 上采样
        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        # 解码器
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(1024, 832, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(832, 480, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(480, 192, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,
        )
        self.convtsp5 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,
        )

        # 将T=4
        self.conv14 = nn.Conv3d(192, 192, kernel_size=(4, 1, 1), stride=(4, 1, 1), bias=True)
        self.conv24 = nn.Conv3d(480, 480, kernel_size=(4, 1, 1), stride=(4, 1, 1), bias=True)
        self.conv34 = nn.Conv3d(832, 832, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=True)

        # 解码
        self.convouta = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

        # 使用ASPP
        self.aspp_conv1 = nn.Conv3d(1024, 256, kernel_size=(1, 3, 3), padding=(0, 2, 2), dilation=(1, 2, 2))
        self.aspp_bn1 = nn.BatchNorm3d(256)
        self.aspp_conv2 = nn.Conv3d(1024, 256, kernel_size=(1, 3, 3), padding=(0, 4, 4), dilation=(1, 4, 4))
        self.aspp_bn2 = nn.BatchNorm3d(256)
        self.aspp_conv3 = nn.Conv3d(1024, 256, kernel_size=(1, 3, 3), padding=(0, 8, 8), dilation=(1, 8, 8))
        self.aspp_bn3 = nn.BatchNorm3d(256)
        self.aspp_conv4 = nn.Conv3d(1024, 256, kernel_size=(1, 3, 3), padding=(0, 16, 16), dilation=(1, 16, 16))
        self.aspp_bn4 = nn.BatchNorm3d(256)

        self.aspp_conv = nn.Conv3d(2048, 1024, kernel_size=1, bias=False)
        self.aspp_bn = nn.BatchNorm3d(1024)
        self.aspp_relu = nn.ReLU()

        # 融合
        self.GA = Gate(1024)
        self.GB = Gate(832)
        self.GC = Gate(480)
        self.GD = Gate(192)

    def forward(self, y0, y1, y2, y3):
        y3 = self.conv14(y3)
        y2 = self.conv24(y2)
        y1 = self.conv34(y1)

        # ASPP
        Fea2 = self.aspp_bn1(self.aspp_conv1(y0))
        Fea4 = self.aspp_bn2(self.aspp_conv2(y0))
        Fea8 = self.aspp_bn3(self.aspp_conv3(y0))
        Fea16 = self.aspp_bn4(self.aspp_conv4(y0))
        temp = torch.cat((Fea2, Fea4, Fea8, Fea16, y0), 1)
        ya0 = self.aspp_relu(self.aspp_bn(self.aspp_conv(temp)))

        z = self.GA(ya0, y0)

        zA = self.convtsp1(z)

        z = self.GB(zA, y1)

        zB = self.convtsp2(z)

        z = self.GC(zB, y2)

        zC = self.convtsp3(z)

        z = self.GD(zC, y3)

        zD = self.convtsp4(z)

        zE = self.convtsp5(zD)

        za = self.convouta(zE)

        za = za.view(za.size(0), za.size(3), za.size(4))

        return za


class BackBoneS3D(nn.Module):
    def __init__(self):
        super(BackBoneS3D, self).__init__()

        self.base1 = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
        )
        self.maxp2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.base2 = nn.Sequential(
            Mixed_3b(),
            Mixed_3c(),
        )
        self.maxp3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.base3 = nn.Sequential(
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
        )
        self.maxt4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.maxp4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        self.base4 = nn.Sequential(
            Mixed_5b(),
            Mixed_5c(),
        )

    def forward(self, x):
        y3 = self.base1(x)

        y = self.maxp2(y3)

        y2 = self.base2(y)

        y = self.maxp3(y2)

        y1 = self.base3(y)

        y = self.maxt4(y1)
        y = self.maxp4(y)

        y0 = self.base4(y)

        return [y0, y1, y2, y3]
