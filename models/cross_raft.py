import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .RAFT.update import BasicUpdateBlock, SmallUpdateBlock
from .RAFT.extractor import BasicEncoder, SmallEncoder
from .RAFT.corr import CorrBlock, AlternateCorrBlock
from .RAFT.utils.utils import bilinear_sampler, coords_grid, upflow8
from .basic_blocks import conv

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class Default_args:
    def __init__(self):
        self.mixed_precision = False
        self.small = False
        self.alternate_corr = False

class CrossAdapter(nn.Module):
    def __init__(self):
        super(CrossAdapter, self).__init__()
        self.feat_net = nn.Sequential(
            conv(3, 32, stride=2, instance_norm=True),
            conv(32, 64, stride=2, instance_norm=True),
            conv(64, 128, stride=2, instance_norm=True),
            conv(128, 192, stride=2, instance_norm=True),
            nn.Conv2d(192, 192, kernel_size = 3, stride = 1, dilation = 1, padding = 1, bias=True))
        self.adapool = nn.AdaptiveAvgPool2d(16)
        self.adapter_net = nn.Sequential(
            conv(192+192, 64, stride=2),
            conv(64, 32, stride=2),
            nn.Flatten(),
            nn.Linear(32*4*4, 256*256)
        )
        #self.adapool = nn.AdaptiveMaxPool2d(16)
    def forward(self, img1, img2):
        feat1 = self.feat_net(img1)
        feat2 = self.feat_net(img2)
        #
        feat1 = self.adapool(feat1)
        feat2 = self.adapool(feat2)
        B,C,H,W = list(feat1.size())
        feat1 = feat1.view(B,C,H*W)
        feat1_T = feat1.permute(0,2,1)
        feat2 = feat2.view(B,C,H*W)
        coatt = torch.matmul(feat1_T, feat2)
        coatt_T = coatt.permute(0,2,1)
        feat2_1 = torch.matmul(feat2, torch.softmax(coatt_T,dim=1))
        #feat1_2 = torch.matmul(feat1, torch.softmax(coatt,dim=1))

        latent1 = torch.cat([feat1, feat2_1], dim=1).contiguous().view(B,2*C,H,W)
        #latent2 = torch.cat([feat1, feat1_2], dim=1).contiguous().view(B,2*C,H,W)
        #latent2 = torch.cat([feat2, feat1_2], dim=1).contiguous().view(B,2*C,H,W)


        kernel1 = self.adapter_net(latent1).contiguous().view(B, 256, 256)
        #kernel2 = self.adapter_net(latent2).contiguous().view(B, 256, 256)

        return kernel1, kernel1

class CrossRAFT(nn.Module):
    def __init__(self, load_raft_pretrained = False, adapter = True, iters=12):
        super(CrossRAFT, self).__init__()

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.args = Default_args()
        self.args.corr_levels = 4
        self.args.corr_radius = 4
        self.args.dropout = 0
        self.args.alternate_corr = False
        self.args.mixed_precision = False


        self.iters = iters
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=self.args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
    
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.args.dropout)
        
        if load_raft_pretrained:
            self.load_raft_weight()
        if adapter:
            self.adapter = CrossAdapter()
        else:
            self.adapter = None
    
    def load_raft_weight(self):
        path = 'RAFT/raft-sintel.pth'
        dir_path = os.path.dirname(__file__)
        path = os.path.join(dir_path, path)
        self.load_state_dict({strKey.replace('module.', ''): tenWeight for strKey, tenWeight in
                                     torch.load(path).items()})
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, get_feature=False):
        """ Estimate optical flow between pair of frames """
        #输入修改为0~1

        image1 = 2 * (image1) - 1.0
        image2 = 2 * (image2) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])      
            
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        if get_feature:
            output_dict = {}
            output_dict['f1'] = fmap1
            output_dict['f2'] = fmap2
            

        if self.adapter is not None:
            B,C,H,W = list(fmap1.size())
            adapter_kernel1, adapter_kernel2 = self.adapter(image1, image2)
            fmap1 = torch.matmul(adapter_kernel1, fmap1.view(B,C,-1)).contiguous().view(B,C,H,W)
            fmap2 = torch.matmul(adapter_kernel2, fmap2.view(B,C,-1)).contiguous().view(B,C,H,W)
        
        if get_feature:
            output_dict['f1_a'] = fmap1
            output_dict['f2_a'] = fmap2
            return output_dict

        #print('adapted', fmap1.sum(), fmap2.sum())
        #print(self.args.alternate_corr)
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            #print(net.sum())
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            #ssprint(f'----------- iter {itr} -------------')
            corr = corr_fn(coords1) # index correlation volume
            
            #print(corr.sum())

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                #print(net.sum(), inp.sum(), corr.sum(), flow.sum())
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
                
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            
            if not upsample:
                flow_predictions.append(coords1 - coords0)
                continue
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        output_dict = {}
        output_dict['flow'] = flow_up
        output_dict['flow_seq'] = flow_predictions
            
        return output_dict
