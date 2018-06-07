#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import torch
import logging
from dataset.preprocess_data import PreprocessData

logger = logging.getLogger(__name__)

def eval_on_model(model, criterion, batch_data, epoch, device, enable_char, batch_char_func):
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
    epoch_binary_acc = AverageMeter()
    epoch_problem_acc = AverageMeter()
    batch_cnt = len(batch_data)
    start_time=time.time()
    for i, batch in enumerate(batch_data, 0):
        # batch data
        contents, question_ans, sample_labels, sample_ids = batch
        contents = contents.to(device)
        question_ans = question_ans.to(device)
        sample_labels = sample_labels.to(device)
        # contents:batch_size*10*200,  question_ans:batch_size*100  ,sample_labels=batchsize
        # forward
        pred_labels = model.forward(contents, question_ans)  # pred_labels size=(batch,1)
        # get loss
        loss = criterion.forward(pred_labels, sample_labels)
        # logging
        batch_loss = loss.item()
        epoch_loss.update(batch_loss, len(sample_ids))

        binary_acc = compute_binary_accuracy(pred_labels, sample_labels)
        problem_acc = compute_problems_accuracy(pred_labels, sample_labels, sample_ids)

        epoch_binary_acc.update(binary_acc, len(sample_ids))
        epoch_problem_acc.update(problem_acc, int(len(sample_ids) / 5))

        logger.info('epoch=%d, batch=%d/%d, loss=%.5f binary_acc=%.4f problem_acc=%.4f' % (
            epoch, i, batch_cnt, batch_loss, binary_acc, problem_acc))

        # manual release memory, todo: really effect?
        del contents, question_ans, sample_labels, sample_ids
        del pred_labels, loss
        # torch.cuda.empty_cache()

    eval_time=time.time()-start_time
    logger.info(
        '===== epoch=%d, batch_count=%d, epoch_average_loss=%.5f, avg_binary_acc=%.4f, avg_problem_acc=%.4f, eval_time=%.1f====' % (
            epoch, batch_cnt, epoch_loss.avg, epoch_binary_acc.avg, epoch_problem_acc.avg,eval_time))

    return epoch_loss.avg, epoch_binary_acc.avg, epoch_problem_acc.avg


def compute_binary_accuracy(pred_labels, real_labels):
    pred_labels = torch.argmax(pred_labels, dim=1)  # 得到一个16*1的矩阵，
    difference = torch.abs(pred_labels - real_labels)
    accuracy = 1.0 - torch.mean(difference.float())
    return accuracy


def compute_problems_accuracy(pred_labels, real_labels, sample_ids):
    check_flag = True
    problem_num = int(real_labels.size()[0] / 5)
    # 检查是不是每5个sample都是属于一个问题的
    for i in range(problem_num):
        problem_id = sample_ids[i * 5].split("_")[0]
        for j in range(5):
            cur_pro_id = sample_ids[i * 5 + j].split("_")[0]
            if (cur_pro_id != problem_id):
                check_flag = False
                break

    if check_flag:
        softmax_score = torch.nn.functional.softmax(pred_labels, dim=1)
        confidence = softmax_score[:, 1] - softmax_score[:, 0]
        problem_wise_confidence = confidence.resize_(problem_num, 5)
        max_idx = torch.argmax(problem_wise_confidence, dim=1)
        new_labels = torch.zeros(problem_num, 5).to(torch.device("cuda"))
        new_labels[range(problem_num), max_idx] = 1
        real_labels = real_labels.resize_(problem_num, 5).float()
        error = torch.abs(real_labels - new_labels)
        accuracy = 1.0 - (torch.mean(error.float()) * 5 / 2)
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
