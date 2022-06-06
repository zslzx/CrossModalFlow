import torch
import torch.nn.functional as F
from visdom import Visdom
import numpy as np
from flow_utils import vis_flow

def resize_image(input, scale_factor):
    # return ResizeImageFunction.apply(input, scale_factor)
    with torch.no_grad():
        return F.interpolate(input, scale_factor=scale_factor, mode='bicubic', recompute_scale_factor=True)

def load_state(model, path, state_dict_name='state_dict'):
    print(path)
    pretrained_dict = torch.load(path)[state_dict_name]
    model_dict=model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def vis_flow_tensor(flow):
    #print(flow.shape)
    flow = flow.numpy()
    flow = np.transpose(flow, [1,2,0])
    #print(flow.shape)
    flow = np.transpose(vis_flow(flow), (2, 0, 1))
    flow = torch.from_numpy(flow)
    return flow

def vis_flow_tensor2img(flow):
    flow = flow.numpy()
    flow = np.transpose(flow, [1,2,0])
    return vis_flow(flow)

class VisdomWriter:
    def __init__(self, visdom_port):
        self.viz = Visdom(port=visdom_port)
        self.names = []
    def add_scalar(self, name, val, step):
        try:
            val = val.item()
        except:
            val = float(val)
        if name not in self.names:
            self.names.append(name)
            self.viz.line([val], [step], win=name, opts=dict(title=name))
        else:
            self.viz.line([val], [step], win=name, update='append')
    def add_image(self, name, image, step):
        self.viz.image(image, win=name, opts=dict(title=name))
    def close(self):
        return

class EmptyWriter:
    def __init__(self):
        return
    def add_scalar(self, name, val, step):
        return
    def add_image(self, name, image, step):
        return
    def close(self):
        return

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', alignfactor=8):
        #print(dims)
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // alignfactor) + 1) * alignfactor - self.ht) % alignfactor
        pad_wd = (((self.wd // alignfactor) + 1) * alignfactor - self.wd) % alignfactor
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        #print(self._pad)

    def pad(self, *inputs):
        #print(inputs[0].shape)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
