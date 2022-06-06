import torch
from torch.autograd.grad_mode import F
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
import numpy as np
import cv2
import os
import random

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

def read_mask_tensor(path):
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.tensor((img > 0).astype(np.float32)).unsqueeze(0)
    except:
        print(path)
        assert False
    return img

def read_img_tensor(path):
    #print(path)
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img/255.0).permute(2,0,1).unsqueeze(0)
    except:
        print(path)
        assert False
    return img

class RgbnirStereo(Dataset):
    def __init__(self, root=r'D:\rgbnir\rgbnir_stereo'):
        list_path = os.path.join(root, 'lists')
        self.gt_path = os.path.join(root, 'data')
        collections = ['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742']

        self.datalist = []
        for i in collections:
            f = open(os.path.join(list_path, i + '.txt'), 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                _, key, _, _, _, _ = line.split()
                self.datalist.append((i, key))
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self, idx):
        i, key = self.datalist[idx]
        img1 = read_img_tensor(os.path.join(self.gt_path, i, 'RGBResize', key + '_RGBResize.png'))
        img2 = read_img_tensor(os.path.join(self.gt_path, i, 'NIRResize', key + '_NIRResize.png'))
        #rgb左图,nir右图（左右指相机位置，与像素位置相反）

        keypoints = []
        f = open(os.path.join(self.gt_path, i, 'Keypoint', key + '_Keypoint.txt'), 'r')
        gts = f.readlines()
        f.close()
        for gt in gts:
            x, y, d, c = gt.split()
            x = round(float(x) * 582) - 1
            x = int(max(0,min(582, x)))
            y = round(float(y) * 429) - 1
            y = int(max(0,min(429, y)))
            d = float(d) * 582
            c = int(c)
            keypoints.append((x, y, d, c))
        return img1[0], img2[0], keypoints

class TriModalHuman(Dataset):
    def __init__(self, root=r'D:\rgbnir\TrimodalDataset', modal='RGB-FIR', subdirs = ['Scene 1', 'Scene 2', 'Scene 3']):
        super(TriModalHuman, self).__init__()
        self.modal0 = modal
        modal = modal.split('-')
        self.modal = modal
        #subdirs = ['Scene 1', 'Scene 2', 'Scene 3']
        self.l = []
        for subdir in subdirs:
            #annot = os.path.join(root, subdir, 'annotations.csv')
            for f in os.listdir(os.path.join(root, subdir, 'rgbMasks')):
                if '.png' not in f:
                    continue
                
                rgb = os.path.join(root,subdir,'SyncRGB',f[:-3]+'jpg')
                rgb_mask = os.path.join(root,subdir,'rgbMasks',f)
                dep = os.path.join(root,subdir,'SyncD',f)
                dep_mask = os.path.join(root,subdir,'depthMasks',f)
                fir = os.path.join(root,subdir,'SyncT',f[:-3]+'jpg')
                fir_mask = os.path.join(root,subdir,'thermalMasks',f)
                
                file_dict = {}
                file_dict['RGB'] = rgb,rgb_mask
                file_dict['FIR'] = fir,fir_mask
                file_dict['DEP'] = dep,dep_mask
                
                img1,mask1 = file_dict[modal[0]]
                img2,mask2 = file_dict[modal[1]]
                self.l.append((img1, mask1, img2, mask2))

    def __len__(self):
        return len(self.l)
    def __getitem__(self, idx):
        img1, mask1, img2, mask2 = self.l[idx]
        
        img1 = read_img_tensor(img1)[0]
        img2 = read_img_tensor(img2)[0]
        mask1 = read_mask_tensor(mask1)[0]
        mask2 = read_mask_tensor(mask2)[0]

        return img1, img2, mask1, mask2


KITTI_ROOT = r'D:\optical_flow\kitti'
KITTI_NIR_ROOT = r'D:\rgbnir\augmentKITTI'
SPLIT_TXT_ROOT = r'D:\rgbnir\augmentKITTI'

def load_arr_from_text(filename):
    filename = os.path.join(SPLIT_TXT_ROOT, filename)
    with open(filename) as f:
        arr = f.readlines()
        arr = [s.strip() for s in arr]
        if arr[-1] == '':
            arr.pop(-1)
    return arr

def getsplit(year='2012'):
    train_file = 'KITTI_split_%s_train.txt'%year
    test_file = 'KITTI_split_%s_test.txt'%year
    #if os.path.exists(train_file):
    train_set = load_arr_from_text(train_file)
    test_set = load_arr_from_text(test_file)
    return train_set, test_set

def list_all_data_crosskitti(year, split_set, f1_nir=False, f2_nir=True, inc_occ=False):
    res = []
    if year == '2012':
        root = os.path.join(KITTI_ROOT, r'2012\data_stereo_flow\training')
        imgdir = os.path.join(root, 'colored_0')
    else:
        root = os.path.join(KITTI_ROOT, r'2015\data_scene_flow\training')
        imgdir = os.path.join(root, 'image_2')
    
    nirdir = os.path.join(KITTI_NIR_ROOT, f'{year}aug\\{split_set}')
    

    train_set, test_set = getsplit(year)
    if split_set == 'train':
        fns = train_set
    else:
        fns = test_set
    if inc_occ:
        flow_dir = os.path.join(root, 'flow_occ')
    else:
        flow_dir = os.path.join(root, 'flow_noc')

    for fn in fns:
        if f1_nir:
            frame1 = os.path.join(nirdir, fn)
        else:
            frame1 = os.path.join(imgdir, fn)
        
        fn1 = fn[:-5]+'1.png'
        if f2_nir:
            frame2 = os.path.join(nirdir, fn1)
        else:
            frame2 = os.path.join(imgdir, fn1)
        
        flow_path = os.path.join(flow_dir, fn)
        res.append((frame1, frame2, flow_path))
    return res

def read_flow_kitti(path):
    flow = cv2.imread(path, -1)
    flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)
    optical_flow = flow[:, :, :2].astype(np.float32)
    optical_flow = (optical_flow - 32768) / 64.0
    mask = (flow[:, :, 2:]).astype(np.bool_).astype(np.float32)
    return optical_flow, mask   

def random_crop(imgs,H,W):
    shape = imgs[0].shape
    #print(shape)
    x = random.randint(0,shape[0]-H)
    y = random.randint(0,shape[1]-W)
    return [img[x:x+H,y:y+W] for img in imgs]

class Cross_KITTI(Dataset):
    def __init__(self, 
                year = '2012', inc_occ=True, test=True):
        super(Cross_KITTI, self).__init__()
        self.test = test
        self.clip_shape = [320, 896]
        split_set = 'test' if test else 'train'
        self.data_list = list_all_data_crosskitti(year, split_set, False, True, inc_occ) + list_all_data_crosskitti(year, split_set, True, False, inc_occ)

    def __len__(self):
        return len(self.data_list)

    def load_img(self, path):
        return imread(path)
    def read_flow(self, path):
        flow, mask = read_flow_kitti(path)
        return flow
    def load_mask(self, path):
        flow, mask = read_flow_kitti(path)
        return mask[:,:,0]

    def __getitem__(self, index):
        (frame1, frame2, flow) = self.data_list[index]
        img1 = imread(frame1)
        img2 = imread(frame2)
        flow, mask = read_flow_kitti(flow)
        #print(self.test)
        if not self.test:
            img1, img2, flow, mask = random_crop([img1, img2, flow, mask], self.clip_shape[0], self.clip_shape[1])
        
        #TODO
        img1 = np.transpose(img1, [2,0,1])
        img2 = np.transpose(img2, [2,0,1])
        flow = np.transpose(flow, [2,0,1])
        mask = np.transpose(mask, [2,0,1])
        #print(img1.shape)
        #print(flow.shape)
        
        output_dict = {}
        output_dict['img1'] = torch.from_numpy(img1)
        output_dict['img2'] = torch.from_numpy(img2)
        output_dict['flow'] = torch.from_numpy(flow)
        output_dict['mask'] = torch.from_numpy(mask)
        return output_dict
