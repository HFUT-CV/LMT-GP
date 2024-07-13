import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
import math

from timm.models.layers import trunc_normal_, DropPath



class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res



class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()

        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4, x6):

        x = torch.cat([x1, x2, x4, x6], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            Group(conv=default_conv, dim=out_plane//4, kernel_size=3, blocks=1),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            Group(conv=default_conv, dim=out_plane // 2, kernel_size=3, blocks=1),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class LatenD(nn.Module):
    def __init__(self, out_plane):
        super(LatenD, self).__init__()

        # main_out = out_plane // 2
        # self.main0 = nn.Sequential(
        #     BasicConv(main_out, out_plane//2, kernel_size=3, stride=1, relu=True),
        #     BasicConv(out_plane//2, out_plane // 4, kernel_size=1, stride=1, relu=True),
        #     BasicConv(out_plane // 4, out_plane // 8, kernel_size=3, stride=1, relu=True),
        #     BasicConv(out_plane // 8, out_plane //16 , kernel_size=1, stride=1, relu=True)
        # )
        # self.main1 = nn.Sequential(
        #     BasicConv(main_out, out_plane//2, kernel_size=3, stride=1, relu=True),
        #     BasicConv(out_plane//2, out_plane // 4, kernel_size=1, stride=1, relu=True),
        #     BasicConv(out_plane // 4, out_plane // 8, kernel_size=3, stride=1, relu=True),
        #     BasicConv(out_plane // 8, out_plane //16 , kernel_size=1, stride=1, relu=True)
        # )
        self.main = nn.Sequential(
            BasicConv(out_plane, out_plane//2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane//2, out_plane // 4, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 8, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 8, out_plane //8, kernel_size=3, stride=1, relu=True)
        )
        # self.conv = BasicConv(out_plane // 8, out_plane // 8, kernel_size=1, stride=1, relu=True)

    def forward(self, x):
        # sp = x.shape[1] // 2
        #
        # split0 = self.main0(x[:,:sp,:,:])
        # split1 = self.main1(x[:, sp:, :, :])
        # x = torch.cat([split0, split1], dim=1)

        return self.main(x)


class LatenU(nn.Module):
    def __init__(self, out_plane):
        super(LatenU, self).__init__()

        self.main = nn.Sequential(
            BasicConv(out_plane, out_plane * 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane * 2, out_plane * 4, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane * 4, out_plane * 8, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane * 8, out_plane *8, kernel_size=3, stride=1, relu=False)
        )

        # self.conv = BasicConv(out_plane*8, out_plane*8, kernel_size=1, stride=1, relu=False)

    def forward(self, x):

        # x = torch.cat([x, self.main(x)], dim=1)
        return self.main(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=True)

        self.coefficient = nn.Parameter(torch.Tensor(np.ones((2, int(int(channel))))), requires_grad=True)
        self.fusion = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
    def forward(self, x1, x2):

        x = self.merge(x1 * x2)
        y = self.coefficient[0,:][None, :, None, None] * x1 + self.coefficient[1,:][None, :, None, None] * x2
        out = self.fusion(x + y)
        return out
# class FAM(nn.Module):
#     def __init__(self, channel):
#         super(FAM, self).__init__()
#         self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
#
#     def forward(self, x1, x2):
#         x = x1 * x2
#         out = x1 + self.merge(x)
#         return out


class SemiLL(nn.Module):
    def __init__(self, num_res=8):
        super(SemiLL, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            # EBlock(base_channel, num_res),
            # EBlock(base_channel*2, num_res),
            # EBlock(base_channel*4, num_res),
            Group(conv=default_conv, dim=base_channel * 1, kernel_size=3, blocks=4),
            Group(conv=default_conv, dim=base_channel * 2, kernel_size=3, blocks=4),
            Group(conv=default_conv, dim=base_channel * 4, kernel_size=3, blocks=4),
            Group(conv=default_conv, dim=base_channel * 8, kernel_size=3, blocks=4),

        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=2),

            BasicConv(base_channel * 8, base_channel * 4, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            Group(conv=default_conv, dim=base_channel * 8, kernel_size=3, blocks=4),
            Group(conv=default_conv, dim=base_channel * 4, kernel_size=3, blocks=4),
            Group(conv=default_conv, dim=base_channel * 2, kernel_size=3, blocks=4),
            Group(conv=default_conv, dim=base_channel * 1, kernel_size=3, blocks=4),
            # TransformerBlock(base_channel * 4),
            # TransformerBlock(base_channel * 2),
            # TransformerBlock(base_channel * 1)

        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 1, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 8, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 15, base_channel*1),
            AFF(base_channel * 15, base_channel*2),
            AFF(base_channel * 15, base_channel * 4)
        ])

        self.latentD = LatenD(base_channel * 8)
        self.latentU = LatenU(base_channel * 1)

        self.SCMs = nn.ModuleList([
            SCM(base_channel * 2),
            SCM(base_channel * 4),
            SCM(base_channel * 8),
        ])
        self.FAMs = nn.ModuleList([
            FAM(base_channel * 2),
            FAM(base_channel * 4),
            FAM(base_channel * 8),
            FAM(base_channel * 4),
            FAM(base_channel * 2),
            FAM(base_channel * 1),
        ])


    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        x_6 = F.interpolate(x, scale_factor=0.125, mode='bilinear')


        z2 = self.SCMs[0](x_2)
        z4 = self.SCMs[1](x_4)
        z6 = self.SCMs[2](x_6)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAMs[0](z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAMs[1](z, z4)
        res3 = self.Encoder[2](z)

        z = self.feat_extract[3](res3)
        z = self.FAMs[2](z, z6)
        z = self.Encoder[3](z)


        z12 = F.interpolate(res1, scale_factor=0.5, mode='bilinear')
        z14 = F.interpolate(res1, scale_factor=0.25, mode='bilinear')
        # z16 = F.interpolate(res1, scale_factor=0.125, mode='bilinear')

        z21 = F.interpolate(res2, scale_factor=2, mode='bilinear')
        z24 = F.interpolate(res2, scale_factor=0.5, mode='bilinear')
        # z26 = F.interpolate(res2, scale_factor=0.25, mode='bilinear')

        z41 = F.interpolate(res3, scale_factor=4, mode='bilinear')
        z42 = F.interpolate(res3, scale_factor=2, mode='bilinear')
        # z46 = F.interpolate(res3, scale_factor=0.5, mode='bilinear')

        z61 = F.interpolate(z, scale_factor=8, mode='bilinear')
        z62 = F.interpolate(z, scale_factor=4, mode='bilinear')
        z64 = F.interpolate(z, scale_factor=2, mode='bilinear')

        res3 = self.AFFs[2](z14, z24, res3, z64)
        res2 = self.AFFs[1](z12, res2, z42, z62)
        res1 = self.AFFs[0](res1, z21, z41, z61)


        z = self.latentD(z)
        latent = z
        z = self.latentU(latent)


        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[4](z)
        outputs.append(z_)

        z = self.FAMs[3](z, res3)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[5](z)
        outputs.append(z_)

        z = self.FAMs[4](z, res2)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z_ = self.ConvsOut[2](z)
        z = self.feat_extract[6](z)
        outputs.append(z_)

        z = self.FAMs[5](z, res1)
        z = self.Convs[2](z)
        z = self.Decoder[3](z)
        z = self.feat_extract[7](z)
        outputs.append(z)

        return outputs[::-1], latent

#
# net = SemiLL().cuda()
# x = torch.randn(1, 3, 256, 256).cuda()
#
# z= net(x)
# print('total parameters:', sum(param.numel() for param in net.parameters()))
