import torch
from models.cross_raft import CrossRAFT

def get_model():
    model = CrossRAFT(adapter=True)
    state_dict = torch.load('pre_trained/net.ckpt')['state_dict']
    #state_dict = torch.load('cross_raft_ckpt/model/checkpoint-10000.ckpt')['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    return model