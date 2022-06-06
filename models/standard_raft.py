import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from .RAFT import raft

class Default_args:
    def __init__(self):
        self.mixed_precision = False
        self.small = False
        self.alternate_corr = False
model_args = Default_args()

class StandardFlowNet(nn.Module):
    def __init__(self):
        super(StandardFlowNet, self).__init__()
        self._model = raft.RAFT(model_args)
        self.load_weights()
    
    @property
    def model(self):
        return self._model

    def forward(self, f0, f1, iters=12):
        x0 = f0
        x1 = f1
        flow = self._model(x0*255, x1*255, iters=iters, test_mode=False)[-1]
        output_dict = {}
        output_dict['flow'] = flow
        return output_dict

    def load_weights(self, path='RAFT/raft-sintel.pth'):
        dir_path = os.path.dirname(__file__)
        path = os.path.join(dir_path, path)
        self._model.load_state_dict({strKey.replace('module.', ''): tenWeight for strKey, tenWeight in
                                     torch.load(path).items()})
