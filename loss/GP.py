import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pdb
import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import inv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pyiqa
from loss.losses import *

def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
    return AB

def kernel_se(x,y,var):
    sigma_1 = 1.0
    pw = 0.6
    l_1 = torch.max(var)#.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = kernel_distance(x,y)
    Ker = sigma_1**2 *torch.exp(-0.5*d/l_1**2)
    return Ker

# numpy version of Linear kernel function
def kernel_linear(X_u, X_l):
    xu_shape = X_u.shape
    xl_shape = X_l.shape
    # pdb.set_trace()
    if len(xu_shape) == 2:
        X_u_norm = X_u / X_u.norm(dim=-1).view(xu_shape[0], 1)
        X_l_norm = X_l / X_l.norm(dim=-1).view(xl_shape[0], 1)
        ker_t = torch.mm(X_u_norm, X_l_norm.transpose(0, 1))
    elif len(xu_shape) == 3:
        X_u_norm = X_u / X_u.norm(dim=-1).view(xu_shape[0], xu_shape[1], 1)
        X_l_norm = X_l / X_l.norm(dim=-1).view(xl_shape[0], xl_shape[1], 1)
        ker_t = torch.bmm(X_u_norm, X_l_norm.transpose(1, 2))
    elif len(xu_shape) == 4:
        X_u = X_u.reshape(xu_shape[0] * xu_shape[1], xu_shape[2], xu_shape[3])
        X_l = X_l.reshape(xl_shape[0] * xl_shape[1], xl_shape[2], xl_shape[3])
        X_u_norm = X_u / X_u.norm(dim=-1).view(xu_shape[0] * xu_shape[1], xu_shape[2], 1)
        X_l_norm = X_l / X_l.norm(dim=-1).view(xl_shape[0] * xl_shape[1], xl_shape[2], 1)
        X_l_norm = X_l_norm.transpose(1, 2)
        ker_t = torch.bmm(X_u_norm, X_l_norm)
        ker_t = ker_t.view(xu_shape[0], xu_shape[1], xu_shape[2], xl_shape[2])
    return ker_t


pdist_ker = nn.PairwiseDistance(p=2)


def kernel_distance(X_u, X_l):
    xu_shape = X_u.shape
    xl_shape = X_l.shape
    # pdb.set_trace()
    if len(xu_shape) == 2:
        x_l_t = X_l.repeat(xu_shape[0], 1)
        x_u_t = X_u.repeat(1, xl_shape[0]).view(xl_shape[0] * xu_shape[0], xu_shape[1])
        ker_t = pdist_ker(x_u_t, x_l_t)
        ker_t = ker_t.view(xu_shape[0], xl_shape[0])
    elif len(xu_shape) == 3:
        x_l_t = X_l.repeat(1, xu_shape[1], 1)
        x_u_t = X_u.repeat(1, 1, xl_shape[1]).view(xu_shape[0], xl_shape[1] * xu_shape[1], xu_shape[2])
        ker_t = pdist_ker(x_u_t, x_l_t)
        ker_t = ker_t.view(xu_shape[0], xu_shape[1], xl_shape[1])
    elif len(xu_shape) == 4:
        X_u = X_u.reshape(xu_shape[0] * xu_shape[1], xu_shape[2], xu_shape[3])
        X_l = X_l.reshape(xl_shape[0] * xl_shape[1], xl_shape[2], xl_shape[3])
        x_l_t = X_l.repeat(1, xu_shape[2], 1)
        x_u_t = X_u.repeat(1, 1, xl_shape[2]).view(xu_shape[0] * xu_shape[1], xl_shape[2] * xu_shape[2], xu_shape[3])
        ker_t = pdist_ker(x_u_t, x_l_t)
        ker_t = ker_t.view(xu_shape[0], xu_shape[1], xu_shape[2], xl_shape[2])
    return ker_t


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    distt = torch.clamp(dist, 0.0, np.inf)
    dist[dist != dist] = 0
    return dist





class GPStruct(object):
    def __init__(self, num_lbl, num_unlbl, train_batch_size, device):
        self.num_lbl = num_lbl  # number of labeled images
        self.num_unlbl = num_unlbl  # number of unlabeled images
        self.z_height = 32  # height of the feature map z i.e dim 2
        self.z_width = 32  # width of the feature map z i.e dim 3
        self.z_numchnls = 32  # number of feature maps in z i.e dim 1
        self.num_nearest = 16  # number of nearest neighbors for unlabeled vector
        self.device = device
        self.Fz_lbl = torch.zeros((self.num_lbl, self.z_numchnls, self.z_height, self.z_width),
                                  dtype=torch.float32).cuda()  # Feature matrix Fzl for latent space labeled vector matrix
        self.Fz_unlbl = torch.zeros((self.num_unlbl, self.z_numchnls, self.z_height, self.z_width),
                                    dtype=torch.float32).cuda()  # Feature matrix Fzl for latent space unlabeled vector matrix

        self.Kmeans_vec = torch.zeros((self.num_nearest, self.z_numchnls, self.z_height, self.z_width),
                                    dtype=torch.float32).cuda()
        self.sigma_noise = nn.Parameter(torch.tensor(0.1), requires_grad=False)

        self.dict_lbl = {}  # dictionary helpful in saving the feature vectors
        self.dict_unlbl = {}  # dictionary helpful in saving the feature vectors
        self.lambda_var = 0.33  # factor multiplied with minimizing variance
        self.train_batch_size = train_batch_size

        self.KL_div = torch.nn.KLDivLoss()
        # declaring kernel function

        self.kernel_comp = kernel_se



        self.kernel_comp_np = cosine_similarity



    def gen_featmaps(self, dataloader, s_net, t_net):
        iqa_metric = pyiqa.create_metric('PSNR', device=self.device, as_loss=False)
        count = 0
        print("Labelled: started storing feature vectors and kernel matrix")
        for batch_id, train_data in enumerate(dataloader):

            input_im, gt, imgid = train_data
            input_im = input_im.to(self.device)
            gt = gt.to(self.device)
            gt_d = F.interpolate(gt, scale_factor=0.5, mode='bilinear')
            gt_d2 = F.interpolate(gt, scale_factor=0.25, mode='bilinear')
            gt_d4 = F.interpolate(gt, scale_factor=0.125, mode='bilinear')
            labels = []
            labels.append(gt)
            labels.append(gt_d)
            labels.append(gt_d2)
            labels.append(gt_d4)

            s_pred_image, s_zy_in = s_net(input_im)
            t_pred_image, t_zy_in = t_net(input_im)
            shape = t_zy_in.data.shape[0]  # batch
            s_list = [[s_pred_image[j][i].unsqueeze(0) for i in range(shape)] for j in range(len(s_pred_image))]
            t_list = [[t_pred_image[j][i].unsqueeze(0) for i in range(shape)] for j in range(len(t_pred_image))]
            gt_list = [[labels[j][i].unsqueeze(0) for i in range(shape)] for j in range(len(labels))]
            # tensor_mat = s_zy_in.data  # torch.squeeze(zy_in.data)
            # tensor_mat = t_zy_in.data
            # saving latent space feature vectors
            for i in range(shape):
                if imgid[i] not in self.dict_lbl.keys():
                    self.dict_lbl[imgid[i]] = count
                    count += 1
                tmp_i = self.dict_lbl[imgid[i]]
                t_scores = 0
                s_scores = 0
                weight = [1,0,0,0]
                for j in range(1):
                    t_score = iqa_metric(t_list[j][i], gt_list[j][i])
                    s_score = iqa_metric(s_list[j][i], gt_list[j][i])
                    t_scores = t_scores + weight[j] * t_score
                    s_scores = s_scores + weight[j] * s_score

                self.Fz_lbl[tmp_i, :, :, :] = t_zy_in.data[i, :, :, :].data if s_scores < t_scores else s_zy_in.data[i,
                                                                                                        :, :, :].data
                # tensor = torch.squeeze(tensor_mat[i,:,:,:])

        self.var_Fz_lbl = torch.std(self.Fz_lbl, axis=0)
        temp_Fz_lbl = self.Fz_lbl.view(self.num_lbl, -1)

        pca = PCA(n_components=self.num_nearest)
        pca.fit(temp_Fz_lbl.cpu().numpy())
        components = pca.components_
        basis = components[:self.num_nearest].reshape(self.num_nearest, self.z_numchnls, self.z_height, self.z_width)

        self.PCA_vec = torch.from_numpy(basis).cuda()
        del temp_Fz_lbl, pca, components, basis
        print("Labelled: stored feature vectors and kernel matrix")
        return

    def loss(self, pred, target):
        # pred = pred.view(-1,self.z_height*self.z_width)
        # target = target.view(-1,self.z_height*self.z_width)
        diff = pred - target
        loss = diff ** 2  # torch.matmul(self.metric_m,diff))
        return loss.mean(dim=-1).mean(dim=-1)

    def compute_gploss(self, zy_in, imgid, zy_in_unsu, label_flg=0):
        tensor_mat = zy_in
        tensor_mat_unsu = zy_in_unsu
        gp_loss = 0

        B, N, H, W = tensor_mat.shape

        # tensor_vec = tensor_mat.view(-1, self.z_numchnls, 1, self.z_height * self.z_width)
        multiplier = torch.ones((B, 1)).cuda()

        tensor_vec = tensor_mat.view(-1, 1, self.z_numchnls * self.z_height * self.z_width)
        ker_UU = self.kernel_comp(tensor_vec, tensor_vec, self.Fz_lbl)  # k(z,z), i.e kernel value for z,z


        pre_base_vec_lbl = self.PCA_vec.expand(B, *self.Kmeans_vec.size())
        base_vec_lbl = pre_base_vec_lbl.view(B, self.num_nearest, self.z_numchnls*self.z_height * self.z_width)
        # base_vec_lbl = pre_base_vec_lbl.transpose(1, 2)


        ker_LL = self.kernel_comp(base_vec_lbl, base_vec_lbl, self.Fz_lbl)

        Eye = torch.eye(self.num_nearest)
        Eye = Eye.view(1, self.num_nearest, self.num_nearest).cuda()
        Eye = Eye.repeat(B, 1, 1)

        ker_LL = ker_LL + (self.sigma_noise ** 2) * Eye
        inv_ker_LL = torch.inverse(ker_LL)
        tensor_vec = tensor_vec.view(B, 1, self.z_numchnls * self.z_height * self.z_width)

        ker_UL = self.kernel_comp(tensor_vec, base_vec_lbl, self.Fz_lbl)
        mean_pred = torch.bmm(ker_UL, torch.bmm(inv_ker_LL, base_vec_lbl))

        Eye = torch.eye(1)
        Eye = Eye.view(1, 1, 1).cuda()
        Eye = Eye.repeat(B , 1, 1)

        sigma_est = ker_UU - torch.bmm(ker_UL, torch.bmm(inv_ker_LL, ker_UL.transpose(1, 2))) + (
                    self.sigma_noise ** 2) * Eye



        tensor_vec_unsu = tensor_mat_unsu.view(B, 1, self.z_numchnls * self.z_height * self.z_width)
        if label_flg:
            for i in range(tensor_mat.shape[0]):
                loss_unsup = torch.mean((self.loss(tensor_vec_unsu[i, :, :], mean_pred[i, :, :])))
                # loss_unsup = torch.mean(self.loss(tensor_vec_unsu[i, :, :], mean_pred[i, :, :])) - 1.0 * self.lambda_var * torch.mean(torch.abs(sigma_est[i, :, :]))
                # loss_unsup = torch.mean((self.loss(tensor_vec_unsu[i, :, :], mean_pred[i, :, :])) / sigma_est[i, :,
                #                                                                                :]) + 1.0 * self.lambda_var * torch.log(
                #     torch.det(sigma_est[i, :, :]))
                if loss_unsup == loss_unsup:
                    gp_loss += torch.mean(multiplier[i] * ((1.0 * loss_unsup / self.train_batch_size)))
        else:
            for i in range(tensor_mat.shape[0]):
                loss_unsup = torch.mean((self.loss(tensor_vec_unsu[i, :, :], mean_pred[i, :, :])) )
                # loss_unsup = torch.mean(self.loss(tensor_vec[i, :, :], mean_pred[i, :, :])) - 1.0 * self.lambda_var * torch.mean(
                #     torch.abs(sigma_est[i, :, :]))
                # loss_unsup = torch.mean((self.loss(tensor_vec_unsu[i, :, :], mean_pred[i, :, :])) / sigma_est[i, :,
                #                                                                                :]) + 1.0 * self.lambda_var * torch.log(
                #     torch.det(sigma_est[i, :, :]))
                if loss_unsup == loss_unsup:
                    gp_loss += torch.mean(multiplier[i] * ((1.0 * loss_unsup / self.train_batch_size)))

        return gp_loss