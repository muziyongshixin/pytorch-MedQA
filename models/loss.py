#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liyz'

import torch
import torch.nn.functional as F
from IPython import embed

class Embedding_reg_L21_Loss(torch.nn.modules.loss._Loss):
    def __init__(self,c=1):
        super(Embedding_reg_L21_Loss, self).__init__()
        self.c=c
    def forward(self, y_pred, y_true):
        torch.nn.modules.loss._assert_no_grad(y_true)
        weight_change = y_pred-y_true #size=(355921,200)
        loss=weight_change**2 #size=(355921,200)
        loss=torch.sum(loss,1)#size=(355921,1)
        loss+=(1e-8)
        loss=loss**0.5#size=(355921,1)
        loss=torch.sum(loss)#size=(1)
        loss=loss*self.c
        return loss

class delta_embedding_Loss(torch.nn.modules.loss._Loss):
    def __init__(self,c=1):
        super(delta_embedding_Loss, self).__init__()
        self.c=c
    def forward(self,weight_change):
        loss=weight_change**2 #size=(355921,200)
        loss=torch.sum(loss,1)#size=(355921,1)
        loss += (1e-8)
        loss=loss**0.5#size=(355921,1)
        loss=torch.mean(loss)#size=(1)
        loss=loss*self.c
        return loss

class delta_embedding_sum_Loss(torch.nn.modules.loss._Loss):
    def __init__(self,c=1):
        super(delta_embedding_sum_Loss, self).__init__()
        self.c=c
    def forward(self,weight_change):
        loss=weight_change**2 #size=(355921,200)
        loss=torch.sum(loss,1)#size=(355921,1)
        loss += (1e-12)
        loss=loss**0.5#size=(355921,1)
        loss=torch.sum(loss)#size=(1)
        loss=loss*self.c
        return loss
# return max(false_score-true_score+delta , 0)
class SVM_loss(torch.nn.modules.loss._Loss):
    def __init__(self,delta=0.5,mean=True):
        super(SVM_loss,self).__init__()
        self.delta=delta
        self.mean=mean
    def forward(self,pred_score,real_label):
        # pred_score size is 16,5
        # real_label size is 16 where element is between 0 and 4
        batch_size=pred_score.size()[0]
        class_nums=pred_score.size()[1]
        true_score=pred_score[range(batch_size),real_label] # batch
        true_score=true_score.unsqueeze(1).repeat(1,class_nums) #batch*5

        score_gap=pred_score-true_score
        score_gap=score_gap+self.delta

        score_gap[score_gap<0]=0
        loss=torch.sum(score_gap)
        if self.mean:
            loss=loss/batch_size
        return loss


class gate_Loss(torch.nn.modules.loss._Loss):
    def __init__(self,c=0.1,t=0.7):
        super(gate_Loss, self).__init__()
        self.c=c
        self.t=t
    def forward(self,mean_gate_val):
        return self.c*max(mean_gate_val-self.t,0)


class MyNLLLoss(torch.nn.modules.loss._Loss):
    """
    a standard negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    Shape:
        - y_pred: (batch, answer_len, prob)
        - y_true: (batch, answer_len)
        - output: loss
    """
    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def forward(self, y_pred, y_true):
        torch.nn.modules.loss._assert_no_grad(y_true)

        y_pred_log = torch.log(y_pred)
        loss = []
        for i in range(y_pred.shape[0]):
            tmp_loss = F.nll_loss(y_pred_log[i], y_true[i], reduce=False)
            one_loss = tmp_loss[0] + tmp_loss[1]
            loss.append(one_loss)

        loss_stack = torch.stack(loss)
        return torch.mean(loss_stack)


class RLLoss(torch.nn.modules.loss._Loss):
    """
    a reinforcement learning loss. f1 score

    Shape:
        - y_pred: (batch, answer_len)
        - y_true: (batch, answer_len)
        - output: loss
    """
    def __init__(self):
        super(RLLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-6):
        return NotImplementedError

        torch.nn.modules.loss._assert_no_grad(y_true)

        assert y_pred.shape[1] == 2

        same_left = torch.stack([y_true[:, 0], y_pred[:, 0]], dim=1)
        same_left, _ = torch.max(same_left, dim=1)

        same_right = torch.stack([y_true[:, 1], y_pred[:, 1]], dim=1)
        same_right, _ = torch.min(same_right, dim=1)

        same_len = same_right - same_left + 1   # (batch_size,)
        same_len = torch.stack([same_len, torch.zeros_like(same_len)], dim=1)
        same_len, _ = torch.max(same_len, dim=1)

        same_len = same_len.type(torch.float)

        pred_len = (y_pred[:, 1] - y_pred[:, 0] + 1).type(torch.float)
        true_len = (y_true[:, 1] - y_true[:, 0] + 1).type(torch.float)

        pre = same_len / (pred_len + eps)
        rec = same_len / (true_len + eps)

        f1 = 2 * pre * rec / (pre + rec + eps)

        return -torch.mean(f1)
