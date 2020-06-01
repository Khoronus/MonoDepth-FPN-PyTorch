import torch.utils.data as data
import numpy as np
from PIL import Image
from path import Path
from constants import *
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor, RandomHorizontalFlip
import torch, time
import torch.nn.functional as F

def load_depth(filename):
    depth_png = np.asarray(Image.open(filename))

    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / (256. * 256.)
    
    depth[depth == 0] = 1.
    
    return depth

class CustomDataset(data.Dataset):
    def __init__(self, root='D:/DB/Test4/20200601_092858/', seed=None, train=True):
        
        np.random.seed(seed)
        self.root = Path(root)
        img_dir = self.root/'train_images.txt' if train else self.root/'kitti_val_images.txt'
        depth_dir = self.root/'train_depths.txt' if train else self.root/'kitti_val_depth_maps.txt'

        print('img_dir: {}'.format(img_dir))
        print('depth_dir: {}'.format(depth_dir))

        # intr_dir = self.root/'kitti_train_intrinsics.txt' if train else self.root/'kitti_val_intrinsics.txt'
        self.img_l_paths = [line.rstrip('\n') for line in open(img_dir)]
        self.depth_l_paths = [line.rstrip('\n') for line in open(depth_dir)]

        self.length = len(self.img_l_paths)

        print('self.img_l_paths: {}'.format(self.img_l_paths))
        print('self.depth_l_paths: {}'.format(self.depth_l_paths))
        print('self.length: {}'.format(self.length))
            
    def __getitem__(self, index):
        depth = torch.FloatTensor( load_depth(self.depth_l_paths[index])[None,:,:] )
        img = ToTensor()( Image.open(self.img_l_paths[index]) )
        
        #tpad = 376 - img.size(1) 
        #rpad = 1242 - img.size(2)
        # (padLeft, padRight, padTop, padBottom)
        #img = F.pad(img.unsqueeze(0), pad=(0, rpad, tpad, 0), mode='reflect')
        #depth = F.pad(depth.unsqueeze(0), pad=(0, rpad, tpad, 0), mode='constant', value=1.)
        
        return img.data.squeeze(0), depth.data.squeeze(0)

    def __len__(self):
#         return 16 # for debug purpose
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = CustomDataset()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
        print(item)
