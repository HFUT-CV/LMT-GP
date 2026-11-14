import pyiqa
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from itertools import cycle
import torchvision
import torch.distributed as dist
from torch.optim import lr_scheduler
import PIL.Image as Image
from utils import *
from torch.autograd import Variable
from adamp import AdamP
from torchvision.models import vgg16
from loss.losses import *
from loss.GP import GPStruct
import loss.pytorch_ssim as pytorch_ssim
import os
class Trainer:
    def __init__(self, model, tmodel, args, supervised_loader, unsupervised_loader, val_loader, iter_per_epoch, writer, dataset_len, undataset_len):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.dataset_len = dataset_len
        self.undataset_len = undataset_len
        self.val_loader = val_loader
        self.args = args
        self.iter_per_epoch = iter_per_epoch
        self.writer = writer
        self.model = model
        self.tmodel = tmodel
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.num_epochs
        self.save_period = 10
        self.loss_unsup = nn.L1Loss()
        self.loss_str = MyLoss().cuda()
        self.loss_single_per = MyLoss_single_perpe().cuda()
        self.loss_single_recon = MyLoss_single_recon().cuda()
        self.consistency = 0.2
        self.consistency_rampup = 100.0


        self.curiter = 0
        self.model.cuda()
        self.tmodel.cuda()
        self.device, available_gpus = self._get_available_devices(self.args.gpus)
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
        # set optimizer and learning rate
        self.optimizer_s = AdamP(self.model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        # self.lr_scheduler_s = lr_scheduler.StepLR(self.optimizer_s, step_size=100, gamma=0.1)
        self.lr_scheduler_s = lr_scheduler.MultiStepLR(self.optimizer_s, milestones=[100, 150], gamma=0.1)

    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.996):
        # exponential moving average(EMA)
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def predict_with_out_grad(self, image):
        with torch.no_grad():
            predict_target_ul, latent = self.tmodel(image)

        return predict_target_ul, latent

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False


    def train(self):
        if not os.path.isdir('./logs'):
            os.makedirs('./logs')
        if not os.path.isdir('./logs/val_image'):
            os.makedirs('./logs/val_image')
        self.freeze_teachers_parameters()
        if self.start_epoch == 1:
            initialize_weights(self.model)
        else:
            checkpoint = torch.load(self.args.resume_path)
            self.model.load_state_dict(checkpoint['state_dict'])
        gp_struct = GPStruct(self.dataset_len, self.undataset_len, self.args.train_batchsize, device='cuda:0')
        best_val_psnr = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            loss_ave, psnr_train, ssim_train = self._train_epoch(epoch, gp_struct)
            loss_val = loss_ave.item() / self.args.crop_size * self.args.train_batchsize
            train_psnr = sum(psnr_train) / len(psnr_train)
            train_ssim = sum(ssim_train) / len(ssim_train)
            psnr_val, ssim_val = self._valid_epoch(max(0, epoch))

            val_psnr = sum(psnr_val) / len(psnr_val)
            val_ssim = sum(ssim_val) / len(ssim_val)

            print('[%d] main_loss: %.6f, train psnr: %.6f, train ssim: %.6f, val psnr: %.6f, val ssim: %.6f, lr: %.8f' % (
                epoch, loss_val, train_psnr, train_ssim, val_psnr, val_ssim, self.lr_scheduler_s.get_last_lr()[0]))
            # print('[%d] main_loss: %.6f, train psnr: %.6f, val psnr: %.6f, lr: %.8f' % (
            #     epoch, loss_val, train_psnr, val_psnr, self.lr_scheduler_s.get_last_lr()[0]))

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f"{name}", param, 0)

            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict()}
                ckpt_name = str(self.args.save_path) + 'best.pth'
                print("Saving the best checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)
            # Save checkpoint
            if epoch % self.save_period == 0 and self.args.local_rank <= 0:
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict()}
                ckpt_name = str(self.args.save_path) + 'model_e{}.pth'.format(str(epoch))
                print("Saving a checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)

    def _train_epoch(self, epoch, gp_struct):
        sup_loss = AverageMeter()
        unsup_loss = AverageMeter()
        loss_total_ave = 0.0
        psnr_train = []
        ssim_train = []
        self.model.train()
        self.freeze_teachers_parameters()
        train_loader = iter(zip(self.supervised_loader, cycle(self.unsupervised_loader)))
        unsupervised_iter = iter(self.unsupervised_loader)
        tbar = range(len(self.supervised_loader))
        tbar = tqdm(tbar, ncols=100, leave=True)
        gp_struct.gen_featmaps(self.supervised_loader, self.model, self.tmodel)

        for i in tbar:

            # try:
            #     (img_data, label, img_id), (unpaired_data_s, unimg_id) = next(train_loader)
            #     if unpaired_data_s.shape[0] < img_data.shape[0]:
            #         raise StopIteration()
            # except StopIteration:
            #     train_loader = iter(zip(self.supervised_loader, cycle(unsupervised_iter)))
            #     (img_data, label, img_id), (unpaired_data_s, unimg_id) = next(train_loader)
            # print(img_id)
            # print(unimg_id)
            (img_data, label, img_id), (unpaired_data_s, unimg_id) = next(train_loader)
            img_data = Variable(img_data).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)

            unpaired_data_s = Variable(unpaired_data_s).cuda(non_blocking=True)

            # teacher output
            predict_target_u, predict_latent_u = self.predict_with_out_grad(unpaired_data_s)
            origin_target_u = [predict_target_u_.detach().clone() for predict_target_u_ in predict_target_u]
            # origin_target_u = predict_target_u.detach().clone()
            orign_latent_u = predict_latent_u.detach().clone()

            predict_target_su, predict_latent_su = self.predict_with_out_grad(img_data)
            origin_target_su = [predict_target_su_.detach().clone() for predict_target_su_ in predict_target_su]
            # origin_target_su = predict_target_su.detach().clone()
            orign_latent_su = predict_latent_su.detach().clone()

            # student output
            outputs_l, latent_l = self.model(img_data)

            outputs_ul, latent_ul = self.model(unpaired_data_s)

            structure_loss = self.loss_str(outputs_l, label, type='recon')
            perpetual_loss = self.loss_str(outputs_l, label, type='perpetual')
            # rfft_loss = self.loss_str(outputs_l, label, type='rfft')
            # structure_loss = self.loss_single_recon(outputs_l, label)
            # perpetual_loss = self.loss_single_per(outputs_l, label)
            ssim_loss = self.loss_str(outputs_l, label, type='ssim')
            # color_loss = self.loss_str(outputs_l, label, type='color')
            # structure_loss = self.loss_single(outputs_l, label)
            # perpetual_loss = self.loss_single(outputs_l, label)

            sup_gp_loss = gp_struct.compute_gploss(latent_l, img_id, orign_latent_su, label_flg=0)

            loss_sup = structure_loss + 0.1 * sup_gp_loss + 0.4 * perpetual_loss + 0.6*ssim_loss#+ 0.05 * rfft_loss # + color_loss#+ ##
            sup_loss.update(loss_sup.mean().item())

            unsu_gp_loss = gp_struct.compute_gploss(latent_ul, unimg_id, orign_latent_u, label_flg=1)



            unsu_structure_loss = self.loss_single_recon(origin_target_u, outputs_ul)

            loss_unsu = 0.5*unsu_gp_loss + unsu_structure_loss
            unsup_loss.update(loss_unsu.mean().item())
            consistency_weight = self.get_current_consistency_weight(epoch)
            total_loss = consistency_weight * loss_unsu + loss_sup
            total_loss = total_loss.mean()
            temp_psnr, temp_ssim, _ = compute_psnr_ssim(outputs_l[0], label)

            psnr_train.append(temp_psnr)
            ssim_train.append(temp_ssim)
            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()

            tbar.set_description('Train-Student Epoch {} | Ls {:.4f} Lu {:.4f}|'
                                 .format(epoch, sup_loss.avg, unsup_loss.avg))

            del img_data, label, img_id, unpaired_data_s, unimg_id,
            with torch.no_grad():
                self.update_teachers(teacher=self.tmodel, itera=self.curiter)
                self.curiter = self.curiter + 1

        loss_total_ave = loss_total_ave + total_loss

        self.writer.add_scalar('Train_loss', total_loss, global_step=epoch)
        self.writer.add_scalar('sup_loss', sup_loss.avg, global_step=epoch)
        self.writer.add_scalar('unsup_loss', unsup_loss.avg, global_step=epoch)
        self.lr_scheduler_s.step(epoch=epoch - 1)
        return loss_total_ave, psnr_train, ssim_train

    def _valid_epoch(self, epoch):
        iqa_psnr = pyiqa.create_metric('psnr',  as_loss=False)
        iqa_ssim = pyiqa.create_metric('ssim',  as_loss=False)
        psnr_val = []
        ssim_val = []
        self.model.eval()
        self.tmodel.eval()
        val_psnr = AverageMeter()
        val_ssim = AverageMeter()
        total_loss_val = AverageMeter()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for i, (val_data, val_label, hs, ws, image_name, label_path) in enumerate(tbar):
                val_data = Variable(val_data).cuda()
                val_label = Variable(val_label).cuda()
                # forward
                val_outputs, _ = self.model(val_data)
                val_output = val_outputs[0]
                val_output = torch.clamp(val_output, 0, 1)
                val_output = val_output[:, :, :hs[0], :ws[0]]
                temp_res = torchvision.transforms.ToPILImage()(val_output[0, :].cpu())
                temp_res.save(f'./logs/val_image/{image_name[0]}')
                temp_psnr = iqa_psnr(f'./logs/val_image/{image_name[0]}', label_path[0]).item()
                temp_ssim = iqa_ssim(f'./logs/val_image/{image_name[0]}', label_path[0]).item()
                val_psnr.update(temp_psnr, 1)
                val_ssim.update(temp_ssim, 1)
                psnr_val.append(temp_psnr)
                ssim_val.append(temp_ssim)
                tbar.set_description('{} Epoch {} | PSNR: {:.4f}, SSIM: {:.4f}|'.format(
                    "Eval-Student", epoch, val_psnr.avg, val_ssim.avg))

            self.writer.add_scalar('Val_psnr', val_psnr.avg, global_step=epoch)
            self.writer.add_scalar('Val_ssim', val_ssim.avg, global_step=epoch)
            del val_output, val_label, val_data
            return psnr_val, ssim_val

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

