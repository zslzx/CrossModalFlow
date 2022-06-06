from albumentations import augmentations
from albumentations.core.composition import OneOf
import torch
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
import numpy as np
import cv2
import os
import random
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
import tf_args
from torchvision import transforms

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def imread(path):
    img = cv2.imread(path)
    if img is None:
        print(path)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_axis(img):
    x = np.linspace(-1,1,img.shape[0]) 
    y = np.linspace(-1,1,img.shape[1])
    Y, X = np.meshgrid(y, x)
    tense_points=np.stack((X, Y),axis=-1)

    return tense_points

def gradual_change_lighter(img, **params):
    dtype = img.dtype
    #print(dtype)
    axis = get_axis(img)
    #print(axis.shape)
    #print(axis[1,0])
    alpha = np.random.rand(1, 1, 6)
    alpha = alpha * 2.0 - 1.0
    x = []
    lst = [0,1,-1]
    for a1 in range(3):
        b1 = lst[a1]
        if b1 == -1:
            b1 = 1
        else:
            b1 = axis[:,:,b1]
        for a2 in range(a1,3):
            b2 = lst[a2]
            if b2 == -1:
                b2 = np.ones_like(axis[:,:,0])
            else:
                b2 = axis[:,:,b2]
            x.append(b1*b2)
    x = np.stack(x,axis=-1)
    #print(x)
    mask = np.sum(x*alpha, axis=-1)
    ma = np.max(x)
    mi = np.min(x)
    if ma == mi:
        ma = mi+1
    mask = (mask-mi)/(ma-mi)
    mask = mask[:,:,np.newaxis]
    
    #尝试让mask尽量接近0
    iters = random.randint(0,5)
    for i in range(iters):
        mask = mask * mask
    
    maximg = np.max(img)
    if maximg > 1:
        img = img / 255.0

    x = np.random.randint(1,3)
    if x == 1:
        img = img * (1-mask)
    elif x == 2:
        color = np.random.rand(1,1,3)
        img = (img + mask*color)/2
    else:
        color = np.random.rand(1,1,3)
        img = img + mask*color
        img = np.clip(img, 0, 1)
    if maximg > 1:
        img = np.round(img * 255)
        img = np.clip(img, 0, 255)
    return img.astype(dtype)


def randlight(img, **params):
    dtype = img.dtype
    h, w = img.shape[:2]
    pos = (random.randint(0, w-1), random.randint(0, h-1))
    img_1 = np.zeros_like(img)
    lightcolor = (random.randint(180,255), random.randint(180,255), random.randint(180,255))
    light_radius = random.randint(3, 15)
    if random.randint(1,2) == 1:
        line_width = -1
    else:
        line_width = random.randint(light_radius//2, light_radius)
    # print(img.shape)
    # print(img_1.shape)
    # print(img_1.dtype)
    img_1 = cv2.circle(img_1.copy(), pos, light_radius, lightcolor,line_width)
    ksize = (random.randint(3,5), random.randint(3,5))
    img_1 = cv2.blur(img_1,ksize)/255.0

    maximg = np.max(img)
    if maximg > 1:
        img = img / 255.0
    img = np.clip(img + img_1, 0, 1)
    if maximg > 1:
        img = np.round(img * 255)
        img = np.clip(img, 0, 255)
    return img.astype(dtype)


def get_tfs_lighter(args):
    if args is None:
        args = tf_args.default_tf_args
    tfs = []
    tfs_occ = []
    if args['CLAHE']:
        tfs += [A.augmentations.transforms.CLAHE()]
    if args['Weather']:
        tfs += [A.OneOf([A.augmentations.transforms.RandomFog(), A.augmentations.transforms.RandomRain(), A.augmentations.transforms.RandomSnow()]),]
    
    if args['ColorJitter']:
        tfs += [A.augmentations.transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2,always_apply=True),
        ]
    if args['ChannelJitter']:
        tfs += [A.augmentations.transforms.ChannelShuffle(),
        A.augmentations.transforms.ChannelDropout(),
        ]
    if args['ColorInvert']:
        if args['Solarize']:
            tfs += [A.OneOf([A.augmentations.transforms.InvertImg(),A.augmentations.transforms.Solarize([100,200])])]
        else:
            tfs += [A.augmentations.transforms.InvertImg()]
    
    if args['GradualJitter']:
        tfs += [A.augmentations.transforms.Lambda(image=gradual_change_lighter, p=1)]
    
    if args['Noise']:
        noise_list = [A.augmentations.transforms.GaussNoise(var_limit=(5,25)),A.augmentations.transforms.ISONoise()]
        if args['CompressionNoise']:
            noise_list += [A.augmentations.transforms.ImageCompression(quality_lower=50)]
        tfs += [A.OneOf(noise_list)]
    
    if args['Sharpen']:
        tfs += [A.augmentations.transforms.Sharpen(alpha=(0.1,0.25), lightness=(0.25,0.5))]
    
    if args['Blur']:
        blur_list = [A.augmentations.transforms.GaussianBlur(blur_limit=(1,5)),A .augmentations.transforms.MotionBlur(blur_limit=5),A.augmentations.transforms.MedianBlur(blur_limit=5)]
        if args['GlassBlur']:
            blur_list += [A.augmentations.transforms.GlassBlur(sigma=0.35, max_delta=2)]
        tfs += [A.OneOf(blur_list, p=0.8)]
    
    if args['Randlight']:
        tfs += [A.augmentations.transforms.Lambda(image=randlight, p=1)]
    
    if args['Occlusion']:
        tfs_occ +=[
            A.OneOf([A.augmentations.transforms.RandomShadow(), A.augmentations.transforms.RandomSunFlare(src_radius=100)], p=0.8),
            A.OneOf([A.augmentations.transforms.GridDropout(random_offset=True,ratio=0.2), A.augmentations.transforms.CoarseDropout(), A.augmentations.transforms.Cutout()])
        ]
    tfs += [A.augmentations.transforms.ToGray()]
    
    return tfs, tfs_occ

class BaseDataset(Dataset):
    def __init__(self, training=True, patch_h = 256, patch_w = 256, augment_args = ['random_pos','random_flip','random_rotate'], tfs_args=None, cross_modal_aug_rate=1.0):
        super(BaseDataset, self).__init__()
        self.cross_modal_aug_rate = cross_modal_aug_rate
        self._train = training
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.augment_args = augment_args
        tfs, tfs_occ = get_tfs_lighter(tfs_args)
        self.transform = A.Compose(tfs)
        self.transform_occ = A.Compose(tfs_occ)
    
    def get_frames(self, f1, f2):
        h, w, _ = f1.shape
        if self._train:
            if h < self.patch_h or w < self.patch_w:
                padding = ((0,max(0,self.patch_h-h)),(0,max(0,self.patch_w-w)),(0,0))
                f1 = np.pad(f1, padding, 'constant')
                f2 = np.pad(f2, padding, 'constant')
                h, w, _ = f1.shape
            _h = np.random.randint(0, h - self.patch_h + 1)
            _w = np.random.randint(0, w - self.patch_w + 1)
            f1 = f1[_h: _h+self.patch_h, _w: _w+self.patch_w]
            f2 = f2[_h: _h + self.patch_h, _w: _w + self.patch_w]
        
            if 'random_flip' in self.augment_args:
                c = np.random.rand()
                if c < 0.5:
                    f1, f2 = f1[::-1], f2[::-1]
                c = np.random.rand()
                if c < 0.5:
                    f1, f2 = f1[:, ::-1], f2[:, ::-1]
                c = np.random.rand()
                if c < 0.5:
                    f1, f2 = f2, f1
            if 'random_rotate' in self.augment_args and self.patch_h == self.patch_w:
                c = np.random.rand()
                if c < 0.5:
                    f1 = np.rot90(f1)
                    f2 = np.rot90(f2)
        f1_o = f1.copy()
        f2_o = f2.copy()
        c = np.random.rand()
        if c < self.cross_modal_aug_rate:
            c = np.random.rand()
            #print(1, f2.shape, f2.dtype)
            if c < 0.5:
                f1 = self.transform(image=f1)['image']
            else:
                f2 = self.transform(image=f2)['image']
            #print(2, type(f2), f2.shape, f2.dtype)
            f2 = self.transform_occ(image=f2.copy())['image']
            #print(fref.shape)

        output_dict = {}

        f1 = torch.from_numpy(f1.copy()).permute(2, 0, 1).float() / 255.0
        f2 = torch.from_numpy(f2.copy()).permute(2, 0, 1).float() / 255.0    
        f1_o = torch.from_numpy(f1_o.copy()).permute(2, 0, 1).float() / 255.0
        f2_o = torch.from_numpy(f2_o.copy()).permute(2, 0, 1).float() / 255.0
        output_dict['f1'] = f1
        output_dict['f2'] = f2
        output_dict['f1_o'] = f1_o
        output_dict['f2_o'] = f2_o
        return output_dict

def random_crop(imgs, cropH, cropW):
    H, W = imgs[0].shape[:2]
    _h = random.randint(0, H - cropH)
    _w = random.randint(0, W - cropW)

    imgs = [img[_h:_h+cropH, _w:_w+cropW] for img in imgs]
    return imgs

class YoutubeVOSDataset(BaseDataset):
    def __init__(self,
                 root_path='D:/interpolation/datasets/youtube_vos', f_range=10,
                 training=True, patch_h = 256, patch_w = 256, augment_args = ['random_pos','random_flip','random_rotate'], tfs_args=None, cross_modal_aug_rate=1.0):
        super(YoutubeVOSDataset, self).__init__(training,patch_h,patch_w,augment_args,tfs_args,cross_modal_aug_rate)
        assert(training)
        self.root_path = root_path
        self.f_range = f_range

        l = []
        for d in os.listdir(os.path.join(root_path, 'train_all_frames','JPEGImages')):
            dir_path = os.path.join(root_path, 'train_all_frames', 'JPEGImages', d)
            imgs = os.listdir(dir_path)
            numbers = [int(img[:5]) for img in imgs]
            max_n = max(numbers)
            min_n = min(numbers)
            assert(max_n - min_n + 1 == len(numbers))
            l.append((os.path.join(dir_path,'%05d.jpg'), min_n, max_n))

        self.l = l
        
    def __len__(self):
        return len(self.l)

    def __getitem__(self, idx):
        path, cnt_min, cnt_max = self.l[idx]
        while True:
            xx = random.randint(cnt_min, cnt_max)
            yy = random.randint(-self.f_range, self.f_range)
            yy = xx + yy
            if yy >= cnt_min and yy <= cnt_max:
                break
        f1 = imread(path % xx)
        f2 = imread(path % yy)
        return self.get_frames(f1, f2)
