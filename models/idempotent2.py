import torch
import copy
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer_bfp import Buffer
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import copy
from timm.utils.model_ema import ModelEmaV2
import torch.nn as nn
from torch.optim import SGD, Adam
import os

from datasets import get_dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.args import *
from torch.utils.data import DataLoader, TensorDataset

class LogitNormLoss(nn.Module):
    def __init__(self, t=0.1):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x, target, t=10):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms)  / self.t
        return F.cross_entropy(logit_norm, target)
    
def js_divergence(p, q, reduction='none'):
    """Jensen-Shannon divergence"""
    p = p.softmax(-1)
    q = q.softmax(-1)
    m = (p + q) / 2
    
    kl_pm = F.kl_div(p.log(), m, reduction=reduction)
    kl_qm = F.kl_div(q.log(), m, reduction=reduction)
    
    return (kl_pm + kl_qm) / 2

# JS divergence
def js_div(p, q):
    """Function that computes distance between two predictions"""
    p = p+1e-10
    q = q+1e-10
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(torch.log(p), m, reduction='batchmean') +
                  F.kl_div(torch.log(q), m, reduction='batchmean'))

def add_parser(parser):
    parser.add_argument('--weighta', type=float, help='Penalty weight for idempotence distillation.')
    parser.add_argument('--weightb', type=float, help='Penalty weight for current idempotence distillation.')
    parser.add_argument('--weightc', type=float, help='Penalty weight for er.')
    parser.add_argument('--weightmask', type=float, help='Penalty weight for mask ratio.')
    parser.add_argument('--alpha_bfp', type=float, default=0,
				help="Weight of the backward feature projection loss. ")
    parser.add_argument("--class_balance", type=str2bool, default=True,
                        help="If set, the memory buffer will be balanced by class")
    parser.add_argument('--weightema', type=float, default=0,help='Penalty weight for ema.')

    parser.add_argument('--loss_type', type=str, default='mfro', choices=['mse', 'rmse', 'mfro', 'cos'],
				help='How to compute the matching loss on projected features.')
    parser.add_argument("--normalize_feat", action="store_true",
				help="if set, normalize features before computing the matching loss.")
    parser.add_argument("--proj_lr", type=float, default=0.1,
				help="Learning rate for the optimizer on the projectors.")  
    parser.add_argument("--momentum", type=float, default=0.9,
				help="Momentum for SGD.")
    parser.add_argument('--pool_dim', default='hw', type=str, choices=['h', 'w', 'c', 'hw', 'flatten'], 
				help="Pooling before computing BFP loss. If None, no pooling is applied.")
    		
    return parser
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser = add_parser(parser)
    return parser

def pool_feat(f1, f2, pool_dim, normalize_feat=False):
    assert f1.shape == f2.shape, (f1.shape, f2.shape) # (N, C, H, W)

    # If already pooled, return the original features
    if f1.ndim == 2:
        return f1, f2
    
    # Do the pooling and move the channel dim to the last
    if pool_dim == 'hw':
        f1 = f1.mean(dim=(2, 3)) # (N, C)
        f2 = f2.mean(dim=(2, 3)) # (N, C)
    elif pool_dim == 'c':
        f1 = f1.mean(dim=1).reshape(f1.shape[0], -1) # (N, H*W)
        f2 = f2.mean(dim=1).reshape(f2.shape[0], -1) # (N, H*W)
    elif pool_dim == 'h':
        f1 = f1.mean(dim=2).reshape(f1.shape[0], -1) # (N, C*W)
        f2 = f2.mean(dim=2).reshape(f2.shape[0], -1) # (N, C*W)
    elif pool_dim == 'w':
        f1 = f1.mean(dim=3).reshape(f1.shape[0], -1) # (N, C*H)
        f2 = f2.mean(dim=3).reshape(f2.shape[0], -1) # (N, C*H)
    elif pool_dim == 'flatten':
        f1 = f1.transpose(1, 3) # (N, W, H, C)
        f2 = f2.transpose(1, 3) # (N, W, H, C)
    else:
        raise ValueError("Unknown pooling dimension: {}".format(pool_dim))
        
    # Treat each example as an unit
    f1 = f1.reshape(f1.shape[0], -1) # (N, -1)
    f2 = f2.reshape(f2.shape[0], -1) # (N, -1)
    
    # Normalize features if needed
    if normalize_feat:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return f1, f2

def match_loss(f1, f2, loss_type):
    # Compute the loss according to the loss type
    if loss_type == 'mse':
        loss = F.mse_loss(f1, f2)
    elif loss_type == 'rmse':
        loss = F.mse_loss(f1, f2) ** 0.5
    elif loss_type == 'mfro':
        # Mean of Frobenius norm, normalized by the number of elements
        loss = torch.mean(torch.frobenius_norm(f1 - f2, dim=-1)) / (float(f1.shape[-1]) ** 0.5)
    elif loss_type == "cos":
        loss = 1 - F.cosine_similarity(f1, f2, dim=1).mean()
    else:
        raise ValueError("Unknown loss type: {}".format(loss_type))

    return loss
class Idempotent2(ContinualModel):
    NAME = 'idempotent2'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Idempotent2, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device,class_balance = self.args.class_balance)
        self.ft=True
        self.task=0
        self.num_classs = backbone.num_classes
        self.s = backbone.num_classes
        self.first_task = True
        self.old_model=self.deepcopy_model(self.net)
        self.final_d = backbone.final_d
        self.bfp_flag = self.args.alpha_bfp > 0
        self.class_num=0
    def begin_task(self, dataset):
        
        self.class_num = dataset.N_CLASSES_PER_TASK
        
        
        if self.bfp_flag:
            print("Use BFP projector!")
            self.bfp_projector=nn.Linear(self.final_d,self.final_d)
            self.bfp_projector.to(self.device)
            self.opt_proj = SGD(
			self.bfp_projector.parameters(), 
			lr=self.args.proj_lr, momentum=self.args.momentum)
        else:
            print("Dont use BFP projector!")
    def compute_loss(self, feats, feats_old, mask_new, mask_old):
        bfp_loss = 0.0
			
		# After pooling, feat and feat_old have shape (n, d)
        feat, feat_old = pool_feat(feats, feats_old, self.args.pool_dim, self.args.normalize_feat)
        feat_proj = self.bfp_projector(feat) # (N, C)
        bfp_loss += self.args.alpha_bfp * match_loss(feat_proj, feat_old, self.args.loss_type)
        loss = bfp_loss
        loss_dict = {
			'match_loss': bfp_loss,
		}
        return loss, loss_dict

    def observe(self, inputs, labels, not_aug_inputs):
        batch_size, _, H, W = inputs.shape

        #x1 = torch.cat([inputs], dim=1)

        #inputs = torch.cat([x1], dim=0)
        #targets = torch.cat([targets], dim=0)

        self.opt.zero_grad()
        mask_current = torch.rand(1) > self.args.weightmask
        y_0_current = F.one_hot(labels, self.num_classs).float() if mask_current else torch.ones(batch_size, self.num_classs).to(self.device) /self.s
        z_current = self.net.f1(inputs)
        y_1_current, z1_current = self.net.f2(z_current, y_0_current)
        #y_2_current, z2_current = self.net.f2(z_current + z1_current[..., None, None], y_1_current.softmax(-1))
        y_2_current, z2_current = self.net.f2(z_current , y_1_current.softmax(-1))
        loss_supervised_1 = self.loss(y_1_current, labels)
        loss_supervised_2 = self.loss(y_2_current, labels)

        loss = 0.5*(loss_supervised_1 + loss_supervised_2)
      
        if self.args.weightb!=0 and self.task>0:
            y_current_mask = torch.ones(batch_size, self.num_classs).to(self.device) /self.s
            z = self.net.f1(inputs)
            y_1, z1 = self.net.f2(z, y_current_mask)
            z_old = self.old_model.f1(inputs)
            y_2, z2 = self.old_model.f2(z_old, y_1.softmax(-1))
            #loss_unsupervised_y = F.mse_loss(y_1, y_2)
            loss += self.args.weightb*F.mse_loss(y_1, y_2)

            

        
        if not self.buffer.is_empty() and self.args.weightc !=0:
            buf_inputs,  buf_labels,_,_,_ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            batch_size, _, H, W = buf_inputs.shape
            mask = torch.rand(1) > self.args.weightmask
            y_0_buf = F.one_hot(buf_labels, self.num_classs).float() if mask else torch.ones(batch_size, self.num_classs).to(self.device) /self.s
            z_buf = self.net.f1(buf_inputs)
            y_1_buf, z1_buf = self.net.f2(z_buf, y_0_buf)
            #y_2_buf, z2_buf = self.net.f2(z_buf + z1_buf[..., None, None], y_1_buf.softmax(-1))
            y_2_buf, z2_buf = self.net.f2(z_buf , y_1_buf.softmax(-1))
            loss_supervised_1_buf = self.loss(y_1_buf, buf_labels)
            loss_supervised_2_buf = self.loss(y_2_buf, buf_labels)

            loss += self.args.weightc*(loss_supervised_1_buf + loss_supervised_2_buf)


        if not self.buffer.is_empty() and self.task>0 and self.args.weighta!=0:
            buf_inputs,  buf_labels,_,_,_ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            batch_size, _, H, W = buf_inputs.shape
           
            y_0 = torch.ones(batch_size, self.num_classs).to(self.device) /self.s
            
            z = self.net.f1(buf_inputs)
            y_1, z1 = self.net.f2(z, y_0)
         
            z_old = self.old_model.f1(buf_inputs)
            y_2, z2 = self.old_model.f2(z_old, y_1.softmax(-1))
            loss_unsupervised_y = F.mse_loss(y_1, y_2)
            loss += self.args.weighta * loss_unsupervised_y
     

        if not self.buffer.is_empty() and self.task > 0 and self.bfp_flag:
            buf_inputs, buf_labels,_,_,_= self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs_comb = torch.cat((inputs, buf_inputs), dim=0)
            labels_comb = torch.cat((labels, buf_labels), dim=0)
            B,_,H,W =inputs_comb.shape
            y_0 = torch.ones(B, self.num_classs).to(self.device) /self.s
            z = self.net.f1(inputs_comb)
            y_1,feats_comb = self.net.f2(z, y_0,returnt2='features')
            mask_old = labels_comb < self.task * self.class_num
            mask_new = labels_comb >= self.task * self.class_num
            with torch.no_grad():
                self.old_model.eval()
                z_old = self.old_model.f1(inputs_comb)

                _,feats_old = self.old_model.f2(z_old, y_0,returnt2='features')
            bfp_loss_all, bfp_loss_dict = self.compute_loss(
                    feats_comb, feats_old, mask_new, mask_old)
            loss +=  bfp_loss_all
        
  
        if  self.bfp_flag : self.opt_proj.zero_grad()
        loss.backward()
        if  self.bfp_flag : self.opt_proj.step()
        
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels,logits=y_1_current.data, logits2=y_2_current.data,mask=y_0_current)

        return loss.item()
    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        print(self.task)  

        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net).to(self.device)
        else:
            self.old_model = self.deepcopy_model(self.net).to(self.device)
            
    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy
