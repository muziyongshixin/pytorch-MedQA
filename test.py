#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from torch.nn import CrossEntropyLoss

from dataset.MedQA_dataset import MedQADataset

__author__ = 'liyz'

import json
import os
import torch
import logging
import argparse
from dataset.squad_dataset import SquadDataset
from models import *
from utils.load_config import init_logging, read_config
from models.loss import MyNLLLoss, gate_Loss, Embedding_reg_L21_Loss
from utils.eval import eval_on_model
from train import AverageMeter
import csv
from IPython import  embed

logger = logging.getLogger(__name__)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(str(net))
    logger.info('Total number of parameters: %d' % num_params)


def test(config_path, experiment_info):
    logger.info('------------MedQA v1.0 Evaluate--------------')
    logger.info('============================loading config file... print config file =========================')
    global_config = read_config(config_path)
    logger.info(open(config_path).read())
    logger.info('^^^^^^^^^^^^^^^^^^^^^^   config file info above ^^^^^^^^^^^^^^^^^^^^^^^^^')

    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    enable_cuda = global_config['test']['enable_cuda']
    device = torch.device("cuda" if enable_cuda else "cpu")
    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        raise ValueError("CUDA is not abaliable, please unable CUDA in config file")

    torch.set_grad_enabled(False)  # make sure all tensors below have require_grad=False,

    ############################### 获取数据集 ############################
    logger.info('reading MedQA h5file dataset...')
    dataset = MedQADataset(global_config)

    logger.info('constructing model...')
    model_choose = global_config['test']['model']
    logger.info("model choose is:   " + model_choose)
    dataset_h5_path = global_config['test']['dataset_h5']
    if model_choose == 'SeaReader':
        model = SeaReader(dataset_h5_path, device)
    elif model_choose == 'SimpleSeaReader':
        model = SimpleSeaReader(dataset_h5_path, device)
    else:
        raise ValueError('model "%s" in config file not recoginized' % model_choose)

    print_network(model)
    logger.info('dataParallel using %d GPU.....' % torch.cuda.device_count())
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()  # let training = False, make sure right dropout

    global init_embedding_weight
    init_embedding_weight = model.state_dict()['module.embedding.embedding_layer.weight']

    # criterion
    task_criterion = CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device)).to(device)
    gate_criterion = gate_Loss().to(device)
    embedding_criterion = Embedding_reg_L21_Loss().to(device)
    all_criterion = [task_criterion, gate_criterion, embedding_criterion]

    # testing arguments
    logger.info('get test data loader ...')
    test_batch_size = global_config['test']['test_batch_size']
    batch_test_data = dataset.get_dataloader_test(test_batch_size, shuffle=False)

    # load model weight
    logger.info('loading model weight...')
    model_weight_path = global_config['data']['model_path']
    assert os.path.exists(model_weight_path), "not found model weight file on '%s'" % model_weight_path

    weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    if enable_cuda:
        weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage.cuda())
    if not global_config['test']['keep_embedding']:
        del weight['module.embedding.embedding_layer.weight']  # 删除掉embedding层的参数 ，避免尺寸不对的问题
    model.load_state_dict(weight, strict=False)

    # forward
    logger.info('evaluate forwarding...')

    out_path=global_config['test']['output_file_path']+experiment_info+"_result.csv"
    # to just evaluate score or write answer to file
    if out_path is not None:
        predict_on_model(model=model,batch_data=batch_test_data,device=device,out_path=out_path)

    logging.info('finished.')


def predict_on_model(model, batch_data, device,out_path):
    epoch_problem_acc = AverageMeter()
    epoch_binary_acc=AverageMeter()
    batch_cnt = len(batch_data)
    start_time = time.time()
    for bnum, batch in enumerate(batch_data):
        # batch data
        contents, question_ans, sample_labels, sample_ids, sample_categorys, sample_logics = batch
        contents = contents.to(device)
        question_ans = question_ans.to(device)
        sample_labels = sample_labels.to(device)
        sample_logics = sample_logics.to(device)
        # contents:batch_size*10*200,  question_ans:batch_size*100  ,sample_labels=batchsize
        # forward
        pred_labels = model.forward(contents, question_ans, sample_logics)  # pred_labels size=(batch,2)

        binary_acc = compute_binary_accuracy(pred_labels, sample_labels)
        problem_acc = compute_problems_accuracy(pred_labels, sample_labels, sample_ids)

        epoch_problem_acc.update(problem_acc.item(), int(len(sample_ids) / 5))
        epoch_binary_acc.update(binary_acc.item(),len(sample_ids))

        logger.info('batch=%d/%d, binary_acc=%.4f  problem_acc=%.4f' % (bnum, batch_cnt,binary_acc, problem_acc))
        # save result to csv file
        save_test_result_to_csv(sample_ids,pred_labels,sample_labels,sample_categorys,sample_logics,out_path)

    test_time = time.time() - start_time
    logger.info('===== test completed, avg_problem_acc=%.4f, eval_time=%.1f====' % (epoch_problem_acc.avg, test_time))

    return 0

def save_test_result_to_csv(sample_ids,pred_labels,real_labels,sample_categorys,sample_logics,csv_file_path):
    problem_num = int(len(sample_ids) / 5)
    is_first_batch = True
    if os.path.exists(csv_file_path):
        is_first_batch = False
    out = open(csv_file_path, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    if is_first_batch:
        head_info = ['problem_id', 'real_labels', 'pred_labels', 'neg_probability', 'pos_probability', 'binary_correct',
                     'sample_category', 'sample_logics', '正确选项', '预测选项', '是否做对']
        csv_write.writerow(head_info)

    softmax_score = torch.nn.functional.softmax(pred_labels, dim=1)  # 得到一个20*2的矩阵
    pred_labels = torch.argmax(pred_labels, dim=1)  # 得到一个20*1的矩阵

    confidence = softmax_score[:, 1] - softmax_score[:, 0]# 得到一个20*1的矩阵
    problem_wise_confidence = confidence.resize_(problem_num, 5)# 得到一个4*5的矩阵
    max_idx = torch.argmax(problem_wise_confidence, dim=1) # 得到一个4*1的矩阵
    candidates=["A","B","C","D","E"]

    for problem_count in range(problem_num):
        cur_problem_correct_idx=0
        for candidate in range(5):
            cur_index = problem_count * 5 + candidate
            if real_labels[cur_index]==1:
                cur_problem_correct_idx=candidate
            if real_labels[cur_index]==pred_labels[cur_index]:
                cur_binary_correct="True"
            else:
                cur_binary_correct="False"
            cur_row_info=[sample_ids[cur_index],real_labels[cur_index].item(),pred_labels[cur_index].item(),softmax_score[cur_index][0].item(),softmax_score[cur_index][1].item(),cur_binary_correct,"","","","",""]
            csv_write.writerow(cur_row_info)
        if cur_problem_correct_idx==max_idx[problem_count]:
            cur_problem_is_correct="True"
        else:
            cur_problem_is_correct="False"
        cur_problem_row=[sample_ids[problem_count*5].split("_")[0],"","","","","",sample_categorys[problem_count*5],sample_logics[problem_count*5].item(),candidates[cur_problem_correct_idx],candidates[max_idx[problem_count]],cur_problem_is_correct]
        csv_write.writerow(cur_problem_row)

    out.flush()
    out.close()



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
        real_labels = real_labels.clone().resize_(problem_num, 5).float()
        error = torch.abs(real_labels - new_labels)
        accuracy = 1.0 - (torch.mean(error.float()) * 5 / 2)
        return accuracy
    else:
        logging.info("check problem groups failed")
        return 0


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="evaluate on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
    parser.add_argument('--output', '-o', required=False, dest='out_path')
    args = parser.parse_args()

    test(config_path=args.config_path, out_path=args.out_path)
