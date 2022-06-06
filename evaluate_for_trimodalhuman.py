import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from evaluate_dataset import DataLoaderX, TriModalHuman
import flow_utils
from get_model import get_model



def _test(mode, model):
    ltas = []
    ious = []
    #dataset = TriModalHuman(modal=mode, subdirs=['Scene 2'], subset=True)
    dataset = TriModalHuman(modal=mode)
    batch_size = 1
    total_steps = (len(dataset) + batch_size-1) // batch_size
    dl = DataLoaderX(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=True, drop_last=False)
            
    progress = tqdm.tqdm(desc='testing', total=total_steps, ncols=75)
    for img1, img2, gt1, gt2 in dl:
        img1 = img1.float().cuda()
        img2 = img2.float().cuda()
        gt1 = gt1.cuda()
        gt2 = gt2.cuda()
        gt1 = gt1.unsqueeze(1)
        gt2 = gt2.unsqueeze(1)
        
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

        #flow = torch.nn.functional.interpolate(flow, (H0, W0), mode='bilinear',align_corners=True) * 2
        
        #tensor_one_hot = torch.zeros(1,12,H0,W0).cuda()
        #tensor_one_hot.scatter_(1,gt2,1)
        #print(gt1.size())
        #gt1 = gt1.unsqueeze(1)
        #gt2 = gt2.unsqueeze(1)
        gt2_warp = flow_utils.backwarp_tensor(gt2, flow)
        #gt2_warp = gt2_warp.argmax(dim=1,keepdim=True)
        
        #valid = (gt1 > 0).float()
        gt2_warp = (gt2_warp > 0.5)
        gt1 = (gt1 > 0)
        valid = gt1.float()
        #if valid.sum() <= 0:
        #    continue
        b = list(gt1.size())[0]
        iou = (gt1 & gt2_warp).float().view(b,-1).sum(dim=-1)/torch.clip((gt1 | gt2_warp).float().view(b,-1).sum(dim=-1),min=1)
        error = (gt1 != gt2_warp).float()

        #imsave(os.path.join('outpics', 'img1.png'), img1[:,:,:h,:w])
        #imsave(os.path.join('outpics', 'img2.png'), img2[:,:,:h,:w])
        #imsave(os.path.join('outpics', 'img1_warp.png'), flow_utils.backwarp_tensor(img2[:,:,:h,:w], flow))
        #imsave(os.path.join('outpics', 'mask_gt.png'), gt1)
        #imsave(os.path.join('outpics', 'mask.png'), gt2_warp)

        #visualize(img1,flow_utils.backwarp_tensor(img2, output_dict['flow']),gt1,gt2_warp)
        lta = (error*valid).view(b,-1).sum(dim=-1)/torch.clip(valid.view(b,-1).sum(dim=-1),min=1)
        valid = valid.view(b,-1).sum(dim=-1)
        #print(lta, iou)
        #errs.append(error.mean().item())
        for i in range(b):
            if valid[i] > 0:
                ious.append(iou[i].item())
                ltas.append(lta[i].item())
        progress.update(1)
    progress.close()
    lta = np.mean(ltas)
    iou = np.mean(ious)
    print(f'Mode: {mode}, 1-lta: {lta}, 1-iou: {1-iou}')

def test(model):
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        _test('RGB-DEP', model)
        _test('RGB-FIR', model)
        _test('DEP-FIR', model)

if __name__ == '__main__':
    model = get_model()
    test(model)