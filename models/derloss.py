# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import copy
from timm.utils.model_ema import ModelEmaV2

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Derloss(ContinualModel):
    NAME = 'derloss'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Derloss, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.ft=True
        self.s = 100
        self.c = 0
        self.task=0

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        batch_size, _, H, W = inputs.shape
        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
        x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
        targets_2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_2 = targets_2 * labels.reshape(-1, 1, 1, 1) + 1
        x2 = torch.cat([inputs, targets_2 / self.s], dim=1)
        inputs_sum = torch.cat([x1, x2], dim=0)
        labels_sum = torch.cat([labels, labels], dim=0)
       
        self.opt.zero_grad()

        #outputs1 = self.net(x1)
    
        #outputs2 = self.net(x2)
        outputs = self.net(inputs_sum)
      
        loss = self.loss(outputs, labels_sum)



        #loss = 1/2*(self.loss(outputs1, labels)+self.loss(outputs2, labels))
      

        if not self.buffer.is_empty() :
            buf_inputs, buf_labels,buf_logits,buf_logits2= self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            B,_,H,W =buf_inputs.shape

            
            targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            #buf_outputs1 = self.net(xbuf1)
            #max_logits = torch.max(buf_logits, buf_logits2)
           # loss +=  0.6* F.mse_loss(buf_outputs1, buf_logits)
            
            #loss +=  0.5*F.mse_loss(buf_outputs1, buf_logits2)
            targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
            targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
            x_bufs= torch.cat([xbuf1, xbuf2], dim=0)
            x_ouputs=self.net(x_bufs)
            buf_logits_sum=torch.cat([buf_logits, buf_logits2], dim=0)
            loss +=  0.3* F.mse_loss(buf_logits_sum, x_ouputs)
           # buf_outputs2 = self.net(xbuf2)
            #loss +=  0.6* F.mse_loss(buf_outputs2, buf_logits2)
            #loss +=  0.5*F.mse_loss(buf_outputs2, buf_logits)
            
            #xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
            #print("logits2")
            #print(buf_logits2)
            #predictions = torch.nn.functional.softmax(buf_logits2, dim=1)
           # pred_labels = predictions.argmax(dim=1).to(inputs.device)
           # print(pred_labels)
            #print(F.mse_loss(buf_outputs1, buf_logits2))
            
            #loss += 0.3 * F.mse_loss(buf_outputs1, buf_logits)
            #print("logits1")
           # print(buf_logits)
            #predictions = torch.nn.functional.softmax(buf_logits, dim=1)
            #pred_labels = predictions.argmax(dim=1).to(inputs.device)
            #print(pred_labels)
            #print(F.mse_loss(buf_outputs1, buf_logits))

            
            
            '''
            #targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
            #targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            #xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
            #buf_buf_inputs_sum = torch.cat([xbuf1,xbuf2], dim=0)
            buf_logits_sum = torch.cat([buf_logits, buf_logits2], dim=0)
            buf_inputs_sum = torch.cat([buf_inputs,buf_inputs], dim=0)
            #buf_outputs2 = self.net(xbuf2)
            #loss += 0.3 * F.mse_loss(buf_outputs2, buf_logits2)
            #buf_outputs=self.net(buf_buf_inputs_sum)
            #loss += 0.3 * F.mse_loss(buf_outputs, buf_logits_sum)
            predictions = torch.nn.functional.softmax(buf_logits_sum, dim=1)
            pred_labels = predictions.argmax(dim=1).to(inputs.device)
            targets_buf_sum = torch.ones(2*B, 1, H, W).to(inputs.device)
            targets_buf_sum = targets_buf_sum * pred_labels.reshape(-1, 1, 1, 1) + 1
            x_buf2 = torch.cat([buf_inputs_sum, targets_buf_sum / self.s], dim=1)
            buf_outputs_sum=self.net(x_buf2)
            
            
            loss += 0.5*F.mse_loss(buf_outputs_sum, buf_logits_sum)

            '''
            """
            targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
            targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
            inputs_buf = torch.cat([xbuf1, xbuf2], dim=0)
            labels_buf = torch.cat([buf_labels, buf_labels], dim=0)
            outputs_buf = self.net(inputs_buf)
            loss += self.loss(outputs_buf, labels_buf)d
            """
            
            '''
            predictions = torch.nn.functional.softmax(outputs_buf[:B], dim=1)
            pred_labels = predictions.argmax(dim=1).to(inputs.device)
            targets_buf = torch.ones(B, 1, H, W).to(inputs.device)
            targets_buf = targets_buf * pred_labels.reshape(-1, 1, 1, 1) + 1
            x_buf = torch.cat([buf_inputs, targets_buf / self.s], dim=1)
            buf_outputs_sum=self.net(x_buf)
            loss += 0.5*F.mse_loss(buf_outputs_sum, outputs_buf[:B])
            '''

            

        loss.backward()
      
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:batch_size],logits=outputs[:batch_size].data,logits2= outputs[batch_size:].data)

        return loss.item()
    
    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        print(self.task)
        self.net.eval()
        features=[]
        labels_sum=[]
        for k, test_loader in enumerate(dataset.test_loaders):
            if k>1:
                break
            for data in test_loader:
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batch_size, _, H, W = inputs.shape
                    targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
                    x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
                    outputs = self.net(x1, returnt='features')
                    features.append(outputs.cpu().numpy()) 
                    labels_sum.append(labels.cpu().numpy())
        features = np.concatenate(features)
        labels_sum = np.concatenate(labels_sum)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(features)
        print("t-SNE complete!")
        plt.figure(figsize=(10, 8))
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        colors = [
           '#FF0000',  # 红
           '#00FF00',  # 绿
           '#0000FF',  # 蓝
           '#FFD700',  # 金
           '#FF1493',  # 粉
           '#00FFFF',  # 青
           '#FF8C00',  # 橙
           '#8A2BE2',  # 紫
           '#32CD32',  # 绿
           '#FF69B4'   # 粉红
           ]
        for i, class_name in enumerate(class_names):
            idx = labels_sum == i
            plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], 
                       c=[colors[i]], label=class_name, alpha=0.6)
        plt.title('t-SNE Visualization of CIFAR-10')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'tsne_visualization derloss task:{self.task}.png', dpi=300, bbox_inches='tight')
        plt.close()



        
       
     
        
        print('end_task call')