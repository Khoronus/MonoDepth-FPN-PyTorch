import numpy as np
import os, sys
from constants import *
from model_fpn import I2D
import argparse, time
#from utils.net_utils import adjust_learning_rate
import torch
from torch.autograd import Variable
# from dataset.dataloader import DepthDataset
#from dataset.nyuv2_dataset import NYUv2Dataset
from dataset.custom_dataset import CustomDataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from collections import Counter
import matplotlib, cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor, RandomHorizontalFlip
from PIL import Image

class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            #fake = F.upsample(fake, size=(H,W), mode='bilinear')
            #print('RMSE_logforward: {} {}'.format(H, W))
            fake = F.interpolate(fake, size=(H,W), mode='bilinear', align_corners=True)
        loss = torch.sqrt( torch.mean( torch.abs(torch.log(real)-torch.log(fake)) ** 2 ) )
        return loss

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            #fake = F.upsample(fake, size=(H,W), mode='bilinear')
            #print('L1forward: {} {}'.format(H, W))
            fake = F.interpolate(fake, size=(H,W), mode='bilinear', align_corners=True)
        loss = torch.mean( torch.abs(10.*real-10.*fake) )
        return loss

class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            #fake = F.upsample(fake, size=(H,W), mode='bilinear')
            #print('L1_logforward: {} {}'.format(H, W))
            fake = F.interpolate(fake, size=(H,W), mode='bilinear', align_corners=True)
        loss = torch.mean( torch.abs(torch.log(real)-torch.log(fake)) )
        return loss
    
class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold
    
    def forward(real, fake):
        mask = real>0
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            #fake = F.upsample(fake, size=(H,W), mode='bilinear')
            #print('BerHuforward: {} {}'.format(H, W))
            fake = F.interpolate(fake, size=(H,W), mode='bilinear', align_corners=True)
        fake = fake * mask
        diff = torch.abs(real-fake)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss
    
class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            #fake = F.upsample(fake, size=(H,W), mode='bilinear')
            #print('RMSE_logforward: {} {}'.format(H, W))
            fake = F.interpolate(fake, size=(H,W), mode='bilinear', align_corners=True)
        loss = torch.sqrt( torch.mean( torch.abs(10.*real-10.*fake) ** 2 ) )
        return loss

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    
    # L1 norm
    def forward(self, grad_fake, grad_real):

        if not grad_fake.shape == grad_real.shape:
            _,H,W = grad_real.shape
            #fake = F.upsample(fake, size=(H,W), mode='bilinear')
            #print('GradLoss_logforward: {} {}'.format(H, W))
            grad_fake = torch.unsqueeze(grad_fake, 0)
            grad_fake = F.interpolate(grad_fake, size=(H,W), mode='bilinear', align_corners=True)
            grad_fake = torch.squeeze(grad_fake, 0)
        
        return torch.sum( torch.mean( torch.abs(grad_real-grad_fake) ) )

    
class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
    
    def forward(self, grad_fake, grad_real):

        if not grad_fake.shape == grad_real.shape:
            _,H,W = grad_real.shape
            #fake = F.upsample(fake, size=(H,W), mode='bilinear')
            #print('NormalLoss_logforward: {} {}'.format(H, W))
            grad_fake = torch.unsqueeze(grad_fake, 0)
            grad_fake = F.interpolate(grad_fake, size=(H,W), mode='bilinear', align_corners=True)
            grad_fake = torch.squeeze(grad_fake, 0)

        prod = ( grad_fake[:,:,None,:] @ grad_real[:,:,:,None] ).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt( torch.sum( grad_fake**2, dim=-1 ) )
        real_norm = torch.sqrt( torch.sum( grad_real**2, dim=-1 ) )
        
        return 1 - torch.mean( prod/(fake_norm*real_norm) )
            
# def get_acc(output, target):
#     # takes in two tensors to compute accuracy
#     pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#     correct = pred.eq(target.data.view_as(pred)).cpu().sum()
#     print("Target: ", Counter(target.data.cpu().numpy()))
#     print("Pred: ", Counter(pred.cpu().numpy().flatten().tolist()))
#     return float(correct)*100 / target.size(0) 


# https://discuss.pytorch.org/t/adaptive-learning-rate/320/3
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# https://github.com/xanderchf/MonoDepth-FPN-PyTorch/issues/14
# Correct one
def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


NUM_EPOCHS = 10
DOUBLE_BIAS = 0.001
WEIGHT_DECAY = 0.001

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Single image depth estimation')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='custom', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=NUM_EPOCHS, type=int)
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
    parser.add_argument('--bs', dest='bs',
                      help='batch_size',
                      default=16, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=1, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='display interval',
                      default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                      help='output directory',
                      default='saved_models', type=str)

# config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-3, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
    parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)
    parser.add_argument('--eval_epoch', dest='eval_epoch',
                      help='number of epoch to evaluate',
                      default=2, type=int)

# resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
    parser.add_argument('--start_at', dest='start_epoch',
                      help='epoch to start with',
                      default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)

# training parameters
    parser.add_argument('--gamma_sup', dest='gamma_sup',
                      help='factor of supervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_unsup', dest='gamma_unsup',
                      help='factor of unsupervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_reg', dest='gamma_reg',
                      help='factor of regularization loss',
                      default=10., type=float)

    args = parser.parse_args()
    return args

def get_coords(b, h, w):
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(b,1,h,w))  # [B, 1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(b,1,h,w))  # [B, 1, H, W]
    coords = torch.cat((j_range, i_range), dim=1)
    norm = Variable(torch.Tensor([w,h]).view(1,2,1,1))
    coords = coords * 2. / norm - 1.
    coords = coords.permute(0, 2, 3, 1)
   
    return coords
        
def resize_tensor(img, coords):
    return nn.functional.grid_sample(img, coords, mode='bilinear', padding_mode='zeros')

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

#     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    
    return grad_y, grad_x

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx)/255.)
    
    
class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

def collate_fn(data):
    imgs, depths = zip(*data)
    B = len(imgs)
    #im_batch = torch.ones((B,3,376,1242))
    #d_batch = torch.ones((B,1,376,1242))
    im_batch = torch.ones((B,3,720,1280))
    d_batch = torch.ones((B,1,720,1280))
    for ind in range(B):
        im, depth = imgs[ind], depths[ind]
        #print('ind: {} | {} {}'.format(ind, im.shape, depth.shape))
        #im_batch[ind, :, im.shape[1]:, :im.shape[2]] = im
        #d_batch[ind, :, depth.shape[1]:, :depth.shape[2]] = depth
        im_batch[ind, :, :, :] = im
        d_batch[ind, :, :, :] = depth
    return im_batch, d_batch

def collate2_fn(data):
    imgs, depths = zip(*data)
    B = len(imgs)
    H = 180
    W = 320
    #im_batch = torch.ones((B,3,376,1242))
    #d_batch = torch.ones((B,1,376,1242))
    im_batch = torch.ones((B,3,H,W))
    d_batch = torch.ones((B,1,H,W))
    for ind in range(B):
        im, depth = imgs[ind], depths[ind]
        print('ind: {} | {} {}'.format(ind, im.shape, depth.shape))
        #im_batch[ind, :, im.shape[1]:, :im.shape[2]] = im
        #d_batch[ind, :, depth.shape[1]:, :depth.shape[2]] = depth

        im = torch.unsqueeze(im, 0)
        depth = torch.unsqueeze(depth, 0)
        depth = torch.unsqueeze(depth, 0)
        im = F.interpolate(im, size=(H,W), mode='bilinear', align_corners=True)
        depth = F.interpolate(depth, size=(H,W), mode='bilinear', align_corners=True)
        im = torch.squeeze(im, 0)
        depth = torch.squeeze(depth, 0)
        depth = torch.squeeze(depth, 0)

        im_batch[ind, :, :, :] = im
        d_batch[ind, :, :, :] = depth
    return im_batch, d_batch


if __name__ == '__main__':

    # models
    # C:\Users\Moro/.cache\torch\checkpoints
    # Downloading: "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth" to C:\Users\Moro/.cache\torch\checkpoints\resnet101-5d3b4d8f.pth
    args = parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You might want to run with --cuda")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # network initialization
    print('Initializing model...')
    i2d = I2D(fixed_feature_weights=False).to(device)
    #if args.cuda:
    #    i2d = i2d.cuda()
        
    print('Done!')

    # hyperparams
    lr = args.lr
    bs = args.bs
    lr_decay_step = args.lr_decay_step
    lr_decay_gamma = args.lr_decay_gamma

    # params
    params = []
    for key, value in dict(i2d.named_parameters()).items():
      if value.requires_grad:
        if 'bias' in key:
          params += [{'params':[value],'lr':lr*(DOUBLE_BIAS + 1), \
                  'weight_decay': 4e-5 and WEIGHT_DECAY or 0}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': 4e-5}]

    # optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    
    # resume
    if args.resume:
        load_name = os.path.join(args.output_dir,
          'i2d_1_{}.pth'.format(args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        state = i2d.state_dict()
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        i2d.load_state_dict(state)
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint
        torch.cuda.empty_cache()
    
    grad_factor = 10.
    normal_factor = 1.

    # setting to eval mode
    i2d.eval()

    #img = Variable(torch.FloatTensor(1), volatile=True)
    #if args.cuda:
    #    img = img.cuda()

    # https://discuss.pytorch.org/t/out-of-memory-error-during-evaluation-but-training-works-fine/12274/3
    with torch.no_grad():
        with open('D:/DataSets/RGB2Depth/20200602_112100/train_images.txt') as f:    
            for line in f:
                line = line.rstrip('\n')
                print('line: {}'.format(line))
                img_in = ToTensor()( Image.open(line) ).to(device)

                print('evaluating...')
                #img = torch.from_numpy(img_in.transpose(2, 0, 1)).float().to(device)
                img = img_in
                img = torch.unsqueeze(img, 0)
                print('img {}'.format(img.shape))

                z_fake = i2d(img)
                z_fake = F.interpolate(z_fake, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=False)  # resize new line to reduce the computation time
                z_fake = torch.squeeze(z_fake, 0)
                z_fake = torch.squeeze(z_fake, 0)
                img = torch.squeeze(img, 0)
                print(z_fake)

                img_color = img.cpu().numpy().transpose(1, 2, 0)
                cv2.imshow('img_color', img_color)
                img_depth = z_fake.detach().cpu().numpy()
                cv2.imshow('depth', img_depth)
                cv2.imwrite('depth.png', img_depth * 255)
                print(img_depth)
                #waits for user to press any key  
                #(this is necessary to avoid Python kernel form crashing) 
                cv2.waitKey(0)  
                torch.cuda.empty_cache()
  
    #closing all open windows  
    cv2.destroyAllWindows() 