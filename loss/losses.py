import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import loss.pytorch_ssim as pytorch_ssim
import pyiqa
class PerpetualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, res, gt):
        res = (res + 1.0) * 127.5
        gt = (gt + 1.0) * 127.5
        r_mean = (res[:, 0, :, :] + gt[:, 0, :, :]) / 2.0
        r = res[:, 0, :, :] - gt[:, 0, :, :]
        g = res[:, 1, :, :] - gt[:, 1, :, :]
        b = res[:, 2, :, :] - gt[:, 2, :, :]
        p_loss_temp = (((512 + r_mean) * r * r) / 256) + 4 * g * g + (((767 - r_mean) * b * b) / 256)
        p_loss = torch.mean(torch.sqrt(p_loss_temp + 1e-8)) / 255.0
        return p_loss

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        self.perpetual = PerpetualLoss(vgg_model)
        self.color = ColorLoss()
        self.ssim_loss = pytorch_ssim.SSIM()
    def forward(self, xs, ys, type):
        label_orgin = ys
        label_d = F.interpolate(label_orgin, scale_factor=0.5, mode='bilinear')
        label_d2 = F.interpolate(label_orgin, scale_factor=0.25, mode='bilinear')
        label_d4 = F.interpolate(label_orgin, scale_factor=0.125, mode='bilinear')

        labels = []
        labels.append(label_orgin)
        labels.append(label_d)
        labels.append(label_d2)
        labels.append(label_d4)

        pre_orign, pre_d, pre_d2, pre_d4 = [i for i in xs]

        pres = []
        pres.append(pre_orign)
        pres.append(pre_d)
        pres.append(pre_d2)
        pres.append(pre_d4)

        if type == 'recon':
            L_total = 0.0
            i = 1
            for pre, label in zip(pres, labels):

                L2_temp = 0.2 * self.L2(pre, label)
                L1_temp = 0.8 * self.L1(pre, label)
                L_total = L_total + i*(L1_temp + L2_temp)
                i -= 0.2
            # L2_temp = 0.2 * self.L2(pres[0], labels[0])
            # L1_temp = 0.8 * self.L1(pres[0], labels[0])
            # L_total = L1_temp + L2_temp
            return L_total
        elif type == 'perpetual':
            L_total = 0.0
            i = 1
            for pre, label in zip(pres, labels):
                L_pertemp = self.perpetual(pre, label)
                L_total = L_total + i*L_pertemp
                i -= 0.2
            return L_total
            #
            # L_total = self.perpetual(pres[0], labels[0])
            # return L_total
        elif type == 'color':
            L_total = 0.0
            i = 1
            for pre, label in zip(pres, labels):
                L_pertemp = self.color(pre, label)
                L_total = L_total + i*L_pertemp
                i -= 0.2
            return L_total
        elif type == 'ssim':
            L_total = 0.0
            i = 1
            for pre, label in zip(pres, labels):
                L_pertemp = self.ssim_loss(pre, label)
                L_total = L_total + i*L_pertemp
                i -= 0.2
            return 4-L_total
        elif type == 'rfft':
            L_total = 0.0

            for pre, label in zip(pres, labels):
                pre_ = torch.fft.fft2(pre, dim=(-2, -1))
                pre_n = torch.stack((pre_.real, pre_.imag), -1)
                label_ = torch.fft.fft2(label, dim=(-2, -1))
                label_n = torch.stack((label_.real, label_.imag), -1)
                L_pertemp = self.L1(pre_n, label_n)
                L_total = L_total + L_pertemp
            return L_total

class MyLoss_single_recon(nn.Module):
    def __init__(self):
        super(MyLoss_single_recon, self).__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()

    def forward(self, xs, ys):

        L_total = 0.0
        i = 1
        for pre, label in zip(xs, ys):
            L2_temp = 0.2 * self.L2(pre, label)
            L1_temp = 0.8 * self.L1(pre, label)
            L_total = L_total + i * (L1_temp + L2_temp)
            i -= 0.2
        # L2_temp = 0.2 * self.L2(pres[0], labels[0])
        # L1_temp = 0.8 * self.L1(pres[0], labels[0])
        # L_total = L1_temp + L2_temp
        return L_total

class MyLoss_single_perpe(nn.Module):
    def __init__(self):
        super(MyLoss_single_perpe, self).__init__()

        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        self.perpetual = PerpetualLoss(vgg_model)
        self.type = type

    def forward(self, xs, ys):
        L_total = 0.0

        L_total = self.perpetual(xs, ys)

        return L_total



# Charbonnier loss
class CharLoss(nn.Module):
    def __init__(self):
        super(CharLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        diff = torch.add(pred, -target)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

