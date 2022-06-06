import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import RobustLoss
from dataset import DataLoaderX
from utils import VisdomWriter
from torch.optim.adamax import Adamax
import os
import numpy as np
import random
import tqdm
from utils import resize_image, vis_flow_tensor

torch.backends.cudnn.benchmark = True
default_args = {
    'visdom_port': 8097,
    'model_checkpoint_dir': 'model',
    'optimizer_checkpoint_dir': 'training',
    'batch_size': 8,
    'learning_rate': 1e-2,
    'max_epoch': 100,
    'seed': 2019,
    'gamma': 0.8,
    'iter': 12,
    'step_per_epoch': 1000
}

def get_one_step_data(dl):
    while True:
        for data in dl:
            yield(data)


class Trainer:
    def __init__(self, model, output_path, distill = False, teachermodel = None):
        self.model = model
        if distill:
            assert(teachermodel is not None)
        self.teachermodel = teachermodel.cuda().eval()
        self.args = default_args
        self.args['output_path'] = output_path
        model_checkpoint_path = os.path.join(output_path, self.args['model_checkpoint_dir'])
        # optimizer_checkpoint_path = os.path.join(output_path, self.args['optimizer_checkpoint_dir'])
        if not os.path.exists(model_checkpoint_path):
            os.makedirs(model_checkpoint_path)
        # if not os.path.exists(optimizer_checkpoint_path):
        #     os.makedirs(optimizer_checkpoint_path)
        self.model_checkpoint_path = model_checkpoint_path
        # self.optimizer_checkpoint_path = optimizer_checkpoint_path

    def update_args(self, new_args):
        self.args.update(new_args)

    def reload_checkpoint(self, root_path, epoch, model, optimizer, scheduler):
        model_checkpoint = torch.load(os.path.join(root_path,
                                                'model/checkpoint-%d.ckpt' % epoch))
        train_checkpoint = torch.load(os.path.join(root_path,
                                                'training/checkpoint-%d.ckpt' % epoch))
        model.module.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(train_checkpoint['optimizer'])
        scheduler.step(train_checkpoint['step'])
        return model_checkpoint['epoch'] + 1, train_checkpoint['step']


    def calc_loss_raft_distill(self, loss_fn, output_dict_stu, output_dict_tea):
        if 'flow_seq' not in output_dict_stu.keys():
            return self.calc_loss_pwc_distill(loss_fn,output_dict_stu,output_dict_tea)
        gt_flow = output_dict_tea['flow']
        pred_seq = output_dict_stu['flow_seq']
        n_pred_seq = len(pred_seq)
        flow_loss = 0
        for i in range(n_pred_seq):
            i_weight = self.args['gamma']**(n_pred_seq - i - 1)
            flow_loss += i_weight * loss_fn(pred_seq[i], gt_flow)
        return flow_loss
    def calc_loss_raft(self, loss_fn, output_dict, gt_flow, mask):
        pred_seq = output_dict['flow_seq']
        n_pred_seq = len(pred_seq)
        flow_loss = 0
        for i in range(n_pred_seq):
            i_weight = self.args['gamma']**(n_pred_seq - i - 1)
            flow_loss += i_weight * loss_fn(pred_seq[i], gt_flow, mask)
        return flow_loss
    
    def calc_loss_pwc_distill(self, loss_fn, output_dict_stu, output_dict_tea):
        gt_flow = output_dict_tea['flow']
        pred_flow = output_dict_stu['flow']
        loss = loss_fn(pred_flow, gt_flow)
        return loss


    def train_for_steps(self, total_steps = 1000):
        self.model = self.model.cuda().train()
        total_loss = .0
        progress = tqdm.tqdm(desc='training', total=total_steps, ncols=75)
        for i in range(total_steps):
            self.step += 1
            loss_total = []
            data = next(self.dataset)
            f1 = data['f1'].cuda()
            f2 = data['f2'].cuda()
            f1_o = data['f1_o'].cuda()
            f2_o = data['f2_o'].cuda()
            output_dict = self.model(f1, f2)

            with torch.no_grad():
                output_dict_tea = self.teachermodel(f1_o,f2_o)

            loss_total.append(self.calc_loss_raft_distill(self.loss_fn, output_dict, output_dict_tea))
            
            loss_total = sum(loss_total) / len(loss_total)
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            #self.scheduler.step()
            step_loss = loss_total.cpu().detach().numpy()
            total_loss += step_loss

            if self.step % 50 == 0:
                #self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], self.step)
                self.writer.add_scalar('loss/loss_total', loss_total, self.step)
                self.writer.add_image('img/img1', f1[-1].cpu().detach(), self.step)
                self.writer.add_image('img/img2', f2[-1].cpu().detach(), self.step)
                self.writer.add_image('img/flow12', vis_flow_tensor(output_dict['flow'][-1].cpu().detach()), self.step)
            progress.update(1)
        progress.close()
        print("{} steps total loss: {}".format(total_steps, total_loss / total_steps))
        # saving checkpoint
        checkpoint = {'state_dict': self.model.module.state_dict(), 'step': self.step}
        torch.save(checkpoint, os.path.join(self.model_checkpoint_path, 'checkpoint-%d.ckpt' % self.step))
        # train_checkpoint = {
        #     'optimizer': self.optimizer.state_dict(),
        #     'step': self.step
        # }
        # torch.save(train_checkpoint, os.path.join(self.optimizer_checkpoint_path, 'checkpoint-%d.ckpt' % self.step))
        
    def train(self, dataset):
        self.model.train()
        self.model = nn.DataParallel(self.model, device_ids=[0]).cuda()
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args['learning_rate'], weight_decay=.00005, eps=1e-8)
        #self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.args['learning_rate'], self.args['max_epoch']*self.args['step_per_epoch']+100,
        #pct_start=0, cycle_momentum=False, anneal_strategy='linear')

        self.writer = VisdomWriter(self.args['visdom_port'])

        dl = DataLoaderX(dataset, batch_size=self.args['batch_size'], shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
        self.dataset = get_one_step_data(dl)

        self.loss_fn = RobustLoss()

        current_epoch = 0
        self.step = 0
        
        if 'restore_epoch' in self.args.keys():
            current_epoch, self.step = self.reload_checkpoint(self.args['output_path'], self.args['restore_epoch'], self.model, self.optimizer, self.scheduler)

        for epoch in range(current_epoch, self.args['max_epoch']):
            self.train_for_steps(total_steps=self.args['step_per_epoch'])

        self.writer.close()
