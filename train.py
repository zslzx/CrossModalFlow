from trainer import Trainer
from models.standard_raft import StandardFlowNet
from models.cross_raft import CrossRAFT
from utils import load_state
from dataset import YoutubeVOSDataset
#from eval_dataset import NIRScene_Flow
import tf_args
import os

def train(output_path):
    model = CrossRAFT(load_raft_pretrained = True, adapter=True)
    teachermodel = StandardFlowNet()
    args = {
        'visdom_port': 8097,
        'learning_rate': 2e-5,
        'batch_size': 4,
        'max_epoch': 10,
        'gamma': 0.8,
        'eval_epoch_interval': 40
    }
    trainer = Trainer(model, output_path, distill=True, teachermodel=teachermodel)
    trainer.update_args(args)


    dataset = YoutubeVOSDataset(patch_h=256, patch_w=256, f_range=20)

    trainer.train(dataset=dataset)

if __name__ == '__main__':
    save_dir = 'cross_raft_ckpt'
    train(save_dir)