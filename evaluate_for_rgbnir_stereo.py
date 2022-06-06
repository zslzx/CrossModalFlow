import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from evaluate_dataset import DataLoaderX, RgbnirStereo
from get_model import get_model

def test(model):
        dl = DataLoaderX(RgbnirStereo(), batch_size=1, shuffle=False,
                     num_workers=0, pin_memory=True, drop_last=False)
        torch.backends.cudnn.benchmark = True
        ans = [[],[],[],[],[],[],[],[]]
        with torch.no_grad():
            for img1, img2, keypoints in tqdm(dl):
                img1 = img1.float().cuda()
                img2 = img2.float().cuda()
                _,_,h,w = list(img1.size())
                div = 64
                ph = ((h - 1) // div + 1) * div
                pw = ((w - 1) // div + 1) * div
                padding = (0, pw - w, 0, ph - h)
                img1 = F.pad(img1, padding, mode='replicate')
                img2 = F.pad(img2, padding, mode='replicate')
                
                torch.cuda.synchronize()
                #start = time.time()
                output_dict = model(img1, img2)
                torch.cuda.synchronize()
                #end = time.time()
                flow = output_dict['flow'][:,:,:h,:w]
                #print(keypoints.size())
                #print(keypoints[0])
                for x,y,d,c in keypoints:
                    pred_d = -float(flow[0,0,y,x].detach().cpu().numpy())
                    p = max(0, pred_d)
                    ans[c].append((p-d)*(p-d))
                
        rmse = []
        for c in range(8):
            rmse.append(pow(sum(ans[c]) / len(ans[c]), 0.5))
        print('Common    Light     Glass     Glossy  Vegetation   Skin    Clothing    Bag       Mean')
        print('%.4f'%rmse[0], end='')
        for i in range(1,8):
            print('  ', '%.4f'%rmse[i], end='')
        print('  ', '%.4f'%(sum(rmse) / 8.0))
        rmse.append(sum(rmse) / 8.0)
        return rmse

if __name__ == '__main__':
    model = get_model()
    test(model)