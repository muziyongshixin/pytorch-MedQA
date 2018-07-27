#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time

import torch
from IPython import  embed

logger = logging.getLogger(__name__)
gpu_nums=torch.cuda.device_count()
def eval_on_model_5c(model, criterion, batch_data, epoch, device, eval_dataset=''):
    """
    evaluate on a specific trained model
    :param enable_char:
    :param batch_char_func: transform word id to char id representation
    :param model: model with weight loaded
    :param criterion:
    :param batch_data: test data with batches
    :param epoch:
    :param device:
    :return: (em, f1, sum_loss)
    """

    epoch_loss = AverageMeter()
    epoch_problem_acc = AverageMeter()
    batch_cnt = len(batch_data)
    start_time=time.time()
    for i, batch in enumerate(batch_data, 0):
        # batch data
        contents, question_ans, sample_labels, sample_ids, sample_categorys, sample_logics = batch

        if len(sample_ids)%(gpu_nums*5)!=0:
            logger.info("batch num is incorrect, ignore this batch")
            continue
        contents = contents.to(device)
        question_ans = question_ans.to(device)
        sample_labels = sample_labels.to(device)
        sample_labels=torch.argmax(sample_labels.resize_(int(sample_labels.size()[0]/5),5),dim=1)
        sample_logics = sample_logics.to(device)

        # contents:batch_size*10*200,  question_ans:batch_size*100  ,sample_labels=batchsize
        # forward
        pred_labels = model.forward(contents, question_ans, sample_logics)  # pred_labels size=(batch,2)

        # get task loss
        task_loss = criterion[0].forward(pred_labels, sample_labels)

        # # gate_loss
        # # gate_loss = criterion[1].forward(mean_gate_val)
        # gate_loss=0
        #
        # # embedding regularized loss
        embedding_loss=criterion[2].forward(model.delta_embedding.embedding_layer.weight)

        loss = task_loss + embedding_loss
        # logging
        batch_loss = loss.item()
        epoch_loss.update(batch_loss, len(sample_ids))

        problem_acc = compute_problems_accuracy_5c(pred_labels, sample_labels, sample_ids)
        epoch_problem_acc.update(problem_acc, int(len(sample_ids) / 5))

        logger.info('epoch=%d, batch=%d/%d, embedding_loss=%.5f loss=%.5f  problem_acc=%.4f' % (
            epoch, i, batch_cnt, embedding_loss,batch_loss, problem_acc))

        # manual release memory, todo: really effect?
        del contents, question_ans, sample_labels, sample_ids
        del pred_labels, loss
        # torch.cuda.empty_cache()

    eval_time=time.time()-start_time
    logger.info(
        '=====dataset=%s epoch=%d, batch_count=%d, epoch_average_loss=%.5f, avg_problem_acc=%.4f, eval_time=%.1f====' % (
            eval_dataset, epoch, batch_cnt, epoch_loss.avg, epoch_problem_acc.avg,eval_time))

    return epoch_loss.avg, epoch_problem_acc.avg


def compute_binary_accuracy(pred_labels, real_labels):
    pred_labels = torch.argmax(pred_labels, dim=1)  # 得到一个16*1的矩阵，
    difference = torch.abs(pred_labels - real_labels)
    accuracy = 1.0 - torch.mean(difference.float())
    return accuracy


def compute_problems_accuracy_5c(pred_labels, real_labels, sample_ids):
    check_flag = True
    problem_num = int(len(sample_ids) / 5)
    # 检查是不是每5个sample都是属于一个问题的
    for i in range(problem_num):
        problem_id = sample_ids[i * 5].split("_")[0]
        for j in range(5):
            cur_pro_id = sample_ids[i * 5 + j].split("_")[0]
            if (cur_pro_id != problem_id):
                check_flag = False
                break
    if check_flag:
        pred_label_index=torch.argmax(pred_labels,dim=1) # 10*1
        difference_index=pred_label_index-real_labels # 10*1 正确的题是0，错误的地方非零
        correct_count=difference_index.eq(0).sum() #正确题目的数量
        accuracy=correct_count.float().item()/problem_num
        return accuracy
    else:
        logging.info("check problem groups failed")
        return 0

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
