import torch
import numpy as np
from torch.nn import functional as F
from evaluate_dataset import DataLoaderX, Cross_KITTI
import time
import os
from tqdm import tqdm

from get_model import get_model

# def get_epe_and_f1(flows, target, mask):
#     error = torch.norm((flows-target)*mask, p=2, dim=1)
#     target_norm = torch.norm(target*mask, p=2, dim=1)

#     target_norm = torch.max(target_norm, torch.ones_like(target_norm).float()*1e-12)
#     f1_logic = ((error > 3) & (error/target_norm > 0.05)).float()

    
#     mask_sum = mask.sum(dim=3).sum(dim=2).sum(dim=1)
#     mask_sum = torch.max(mask_sum, torch.ones_like(mask_sum).float()*1e-12)

#     epe = error.sum(dim=2).sum(dim=1)/mask_sum
#     f1 = f1_logic.sum(dim=2).sum(dim=1)/mask_sum

#     return epe, f1

# def _test(model, dataset):
#         torch.backends.cudnn.benchmark = True
#         with torch.no_grad():
#             dl = DataLoaderX(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
#             epes = []
#             f1s = []
#             progress = tqdm(desc='testing', total=len(dataset), ncols=75)
#             for data in dl:
#                 img1 = data['img1'].cuda()/255.0
#                 img2 = data['img2'].cuda()/255.0
#                 mask_gt = data['mask'].cuda()
#                 flow_gt = data['flow'].cuda()
                
#                 _,_,h,w = list(img1.size())
#                 div = 64
#                 ph = ((h - 1) // div + 1) * div
#                 pw = ((w - 1) // div + 1) * div
#                 padding = (0, pw - w, 0, ph - h)
#                 img1 = F.pad(img1, padding, mode='replicate')
#                 img2 = F.pad(img2, padding, mode='replicate')
                

#                 output_dict = model(img1, img2)

#                 flow = output_dict['flow'][:,:,:h,:w]
#                 #visualize(img1[:,:,:h,:w],img2[:,:,:h,:w],flow,flow_gt)
#                 epe, f1 = get_epe_and_f1(flow,flow_gt,mask_gt.unsqueeze(1))
#                 #epe = epe.mean().item()
#                 #f1 = f1.mean().item()
#                 #print(epe, f1)
#                 epe = epe.view(-1)
#                 f1 = f1.view(-1)
#                 epes.append(epe.detach().cpu().numpy())
#                 f1s.append(f1.detach().cpu().numpy())
#                 #vis_show(output_dict, data['gt'])
#                 progress.update(1)
#             progress.close()
                
#         avg_epe = np.mean(np.concatenate(epes))
#         avg_f1 = np.mean(np.concatenate(f1s))
#         return avg_epe, avg_f1

def get_epe_and_f1(flow, flow_gt, valid_mask):
    flow = flow[0]
    flow_gt = flow_gt[0]
    valid_mask = valid_mask[0]
    #print(flow.shape)
    #print(flow_gt.shape)
    #print(valid_mask.shape)
    epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
    mag = torch.sum(flow_gt**2, dim=0).sqrt().clip(min=1e-12)

    #print(epe.shape)
    #print(mag.shape)
    #print(valid_mask.shape)
    
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid_mask.view(-1) > 0
    #print(epe.shape)
    #print(val.shape)

    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    
    return epe[val].mean(), out[val]

def _test(model, dataset):
        torch.backends.cudnn.benchmark = True
        with torch.no_grad():
            dl = DataLoaderX(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
            epes = []
            f1s = []
            times = []
            progress = tqdm(desc='testing', total=len(dataset), ncols=75)
            for data in dl:
                img1 = data['img1'].cuda()/255.0
                img2 = data['img2'].cuda()/255.0
                mask_gt = data['mask'].cuda()
                flow_gt = data['flow'].cuda()
                
                _,_,h,w = list(img1.size())
                div = 64
                ph = ((h - 1) // div + 1) * div
                pw = ((w - 1) // div + 1) * div
                padding = (0, pw - w, 0, ph - h)
                img1 = F.pad(img1, padding, mode='replicate')
                img2 = F.pad(img2, padding, mode='replicate')
                
                #torch.cuda.synchronize()
                #start = time.time()
                output_dict = model(img1, img2)
                #torch.cuda.synchronize()
                #end = time.time()
                flow = output_dict['flow'][:,:,:h,:w]
                #visualize(img1[:,:,:h,:w],img2[:,:,:h,:w],flow,flow_gt)
                epe, f1 = get_epe_and_f1(flow,flow_gt,mask_gt)
                epe = epe.mean().item()
                f1 = f1.cpu().numpy()
                #print(epe, f1)
                epes.append(epe)
                f1s.append(f1)
                #t = end - start
                #times.append(t)
                #vis_show(output_dict, data['gt'])
                progress.update(1)
            progress.close()
                
        avg_epe = np.mean(epes)
        #avg_f1 = np.mean(f1s)
        avg_f1 = np.mean(np.concatenate(f1s,axis=-1))
        #avg_time = np.mean(times)
        return avg_epe, avg_f1

def test(model):
    datasets = []

    datasets.append(('Cross_KITTI_noc',Cross_KITTI(inc_occ=False, test=True)))
    datasets.append(('Cross_KITTI_all',Cross_KITTI(inc_occ=True, test=True)))
    
    for dataset in datasets:
        #print(dataset[0])
        avg_epe, avg_f1 = _test(model, dataset[1])
        print(dataset[0], 'Avg. EPE:', avg_epe, 'Avg. F1:', avg_f1)

if __name__ == '__main__':
    model = get_model()
    test(model)