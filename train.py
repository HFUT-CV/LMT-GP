import os
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my import
from dataset_all import TrainLabeled, TrainUnlabeled, ValLabeled
from model_mimo import SemiLL
from utils import *
from trainer import Trainer
import sys
import time
class Logger(object):
    def __init__(self, filename="Default.log"):

        t = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
        filename = 'train'+ t + '.log'
        self.terminal = sys.stdout

        self.log = open(filename, "w", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(gpu, args):
    args.local_rank = gpu
    # random seed
    setup_seed(2023)
    # load data
    train_folder = args.data_dir
    paired_dataset = TrainLabeled(dataroot=train_folder, phase='labeled', finesize=args.crop_size)
    unpaired_dataset = TrainUnlabeled(dataroot=train_folder, phase='unlabeled', finesize=args.crop_size, k= args.unlabel_nums)
    val_dataset = ValLabeled(dataroot=train_folder, phase='val', finesize=args.crop_size)
    paired_sampler = None
    unpaired_sampler = None
    val_sampler = None
    paired_loader = DataLoader(paired_dataset, batch_size=args.train_batchsize, sampler=paired_sampler)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=args.train_batchsize, sampler=unpaired_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batchsize, sampler=val_sampler)
    print('there are total %s batches for train' % (len(paired_loader)))
    print('there are total %s batches for val' % (len(val_loader)))
    # create model
    net = SemiLL()
    ema_net = SemiLL()
    ema_net = create_emamodel(ema_net)
    print('student model params: %d' % count_parameters(net))
    # tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    trainer = Trainer(model=net, tmodel=ema_net, args=args, supervised_loader=paired_loader,
                      unsupervised_loader=unpaired_loader,
                      val_loader=val_loader, iter_per_epoch=len(unpaired_loader), writer=writer, dataset_len=len(paired_dataset), undataset_len=len(unpaired_dataset))

    trainer.train()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N')
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--train_batchsize', default=6, type=int, help='train batchsize')
    parser.add_argument('--val_batchsize', default=1, type=int, help='val batchsize')
    parser.add_argument('--crop_size', default=256, type=int, help='crop size')
    parser.add_argument('--resume', default='False', type=str, help='if resume')
    parser.add_argument('--resume_path', type=str, help='if resume')
    parser.add_argument('--use_pretain', default='False', type=str, help='use pretained model')
    parser.add_argument('--pretrained_path', default='./pretained/net.pth', type=str, help='if pretrained')
    # parser.add_argument('--data_dir', default='/home/a1005/Chinn/Kmeans_otherlabel/', type=str, help='data root path')
    parser.add_argument('--data_dir', default='/home/a1005/Chinn/Kmeans_otherlabel', type=str, help='data root path')
    parser.add_argument('--save_path', default='./model/ckpt/', type=str)
    parser.add_argument('--log_dir', default='./model/log', type=str)
    parser.add_argument('--unlabel_nums', default=120, type=int)
    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = Logger()
    sys.stdout = log
    main(-1, args)
