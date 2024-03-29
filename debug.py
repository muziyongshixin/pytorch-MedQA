#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

__author__ = 'liyz'

import os
import torch
import logging
import argparse
import torch.optim as optim
from dataset.squad_dataset import SquadDataset
from dataset.MedQA_dataset import MedQADataset
from models import *
from models.loss import gate_Loss, Embedding_reg_L21_Loss
from utils.load_config import init_logging, read_config
from utils.eval import eval_on_model
from utils.functions import pop_dict_keys
from IPython import embed
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(str(net))
    logger.info('Total number of parameters: %d' % num_params)


def debug(config_path, experiment_info):
    logger.info('------------MedQA v1.0 Train--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    enable_cuda = global_config['train']['enable_cuda']
    device = torch.device("cuda" if enable_cuda else "cpu")
    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        raise ValueError("CUDA is not abaliable, please unable CUDA in config file")

    ############################### 获取数据集 ############################
    logger.info('reading MedQA h5file dataset...')
    dataset = MedQADataset(global_config)

    logger.info('constructing model...')
    model_choose = global_config['global']['model']
    dataset_h5_path = global_config['data']['dataset_h5']
    logger.info('Using dataset path is : %s'%dataset_h5_path)
    logger.info('### Using model is: %s ###'%model_choose)
    if model_choose == 'SeaReader':
        model = SeaReader(dataset_h5_path, device)
    elif model_choose == 'SimpleSeaReader':
        model = SimpleSeaReader(dataset_h5_path, device)
    elif model_choose == 'SeaReader_v2':
        model = SeaReader_v2(dataset_h5_path, device)
    elif model_choose == 'match-lstm':
        model = MatchLSTM(dataset_h5_path)
    elif model_choose == 'match-lstm+':
        model = MatchLSTMPlus(dataset_h5_path)
    elif model_choose == 'r-net':
        model = RNet(dataset_h5_path)
    else:
        raise ValueError('model "%s" in config file not recoginized' % model_choose)

    print_network(model)
    logger.info('dataParallel using %d GPU.....'%torch.cuda.device_count())
    model=torch.nn.DataParallel(model)
    model = model.to(device)


    task_criterion = CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device)).to(device)
    gate_criterion= gate_Loss().to(device)
    embedding_criterion= Embedding_reg_L21_Loss().to(device)
    all_criterion=[task_criterion,gate_criterion,embedding_criterion]

    # optimizer
    optimizer_choose = global_config['train']['optimizer']
    optimizer_lr = global_config['train']['learning_rate']
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_choose == 'adamax':
        optimizer = optim.Adamax(optimizer_param)
    elif optimizer_choose == 'adadelta':
        optimizer = optim.Adadelta(optimizer_param)
    elif optimizer_choose == 'adam':
        optimizer = optim.Adam(optimizer_param, lr=optimizer_lr)
    elif optimizer_choose == 'sgd':
        optimizer = optim.SGD(optimizer_param,
                              lr=optimizer_lr)
    else:
        raise ValueError('optimizer "%s" in config file not recoginized' % optimizer_choose)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

    # check if exist model weight
    weight_path = global_config['data']['model_path']
    if os.path.exists(weight_path) and global_config['train']['continue']:
        logger.info('loading existing weight............')
        if enable_cuda:
            weight = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda())
        else:
            weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
        # weight = pop_dict_keys(weight, ['pointer', 'init_ptr_hidden'])  # partial initial weight
        # todo 之后的版本可能不需要这句了
        del weight['module.embedding.embedding_layer.weight'] #删除掉embedding层的参数 ，避免尺寸不对的问题
        model.load_state_dict(weight, strict=False)

    # training arguments
    logger.info('start training............................................')
    train_batch_size = 10
    valid_batch_size = 10
    test_batch_size=10

    batch_train_data = dataset.get_dataloader_train(train_batch_size, shuffle=True)
    batch_dev_data = dataset.get_dataloader_dev(valid_batch_size, shuffle=False)
    global batch_test_data
    batch_test_data = dataset.get_dataloader_test(test_batch_size,shuffle=False )


    clip_grad_max = global_config['train']['clip_grad_norm']
    enable_char = False

    best_valid_acc = None
    # every epoch
    for epoch in range(1):
        # train
        model.train()  # set training = True, make sure right dropout
        train_avg_loss, train_avg_binary_acc = train_on_model(model=model,
                                                              criterion=all_criterion,
                                                              optimizer=optimizer,
                                                              batch_data=batch_dev_data,
                                                              epoch=epoch,
                                                              clip_grad_max=clip_grad_max,
                                                              device=device,
                                                              enable_char=enable_char,
                                                              batch_char_func=dataset.gen_batch_with_char)

        # evaluate
        # with torch.no_grad():
        #     model.eval()  # let training = False, make sure right dropout
        #     val_avg_loss, val_avg_binary_acc, val_avg_problem_acc = eval_on_model(model=model,
        #                                                                           criterion=all_criterion,
        #                                                                           batch_data=batch_dev_data,
        #                                                                           epoch=epoch,
        #                                                                           device=device,
        #                                                                           enable_char=enable_char,
        #                                                                           batch_char_func=dataset.gen_batch_with_char,
        #                                                                           init_embedding_weight=init_embedding_weight)

            # test_avg_loss, test_avg_binary_acc, test_avg_problem_acc=eval_on_model(model=model,
            #                                                                       criterion=all_criterion,
            #                                                                       batch_data=batch_test_data,
            #                                                                       epoch=epoch,
            #                                                                       device=device,
            #                                                                       enable_char=enable_char,
            #                                                                       batch_char_func=dataset.gen_batch_with_char,
            #                                                                       init_embedding_weight=init_embedding_weight)

        # # save model when best f1 score
        # if best_valid_acc is None or val_avg_problem_acc > best_valid_acc:
        #     epoch_info = 'epoch=%d, val_binary_acc=%.4f, val_problem_acc=%.4f' % (
        #         epoch, val_avg_binary_acc, val_avg_problem_acc)
        #     save_model(model,
        #                epoch_info=epoch_info,
        #                model_weight_path=global_config['data']['model_weight_dir']+experiment_info+"_model_weight.pt",
        #                checkpoint_path=global_config['data']['checkpoint_path']+experiment_info+"_save.log")
        #     logger.info("=========  saving model weight on epoch=%d  =======" % epoch)
        #     best_valid_acc = val_avg_problem_acc


        # tensorboard_writer.add_scalar("train/problem_acc", train_avg_problem_acc, epoch)
        # tensorboard_writer.add_scalar("val/avg_loss", val_avg_loss, epoch)
        # tensorboard_writer.add_scalar("val/binary_acc", val_avg_binary_acc, epoch)
        # tensorboard_writer.add_scalar("val/problem_acc", val_avg_problem_acc, epoch)

        # tensorboard_writer.add_scalar("test/avg_loss", test_avg_loss, epoch)
        # tensorboard_writer.add_scalar("test/binary_acc", test_avg_binary_acc, epoch)
        # tensorboard_writer.add_scalar("test/problem_acc", test_avg_problem_acc, epoch)

        #  adjust learning rate
        scheduler.step(train_avg_loss)

    logger.info('finished.................................')



def train_on_model(model, criterion, optimizer, batch_data, epoch, clip_grad_max, device, enable_char, batch_char_func):
    """
    train on every batch
    :param enable_char:
    :param batch_char_func:
    :param model:
    :param criterion:
    :param batch_data:
    :param optimizer:
    :param epoch:
    :param clip_grad_max:
    :param device:
    :return:
    """
    epoch_loss = AverageMeter()
    epoch_binary_acc = AverageMeter()
    epoch_problem_acc = AverageMeter()
    batch_cnt = len(batch_data)
    for i, batch in enumerate(batch_data, 0):
        optimizer.zero_grad()
        # batch data
        # bat_context, bat_question, bat_context_char, bat_question_char, bat_answer_range = batch_char_func(batch, enable_char=enable_char, device=device)
        contents, question_ans, sample_labels, sample_ids ,sample_categorys, sample_logics = batch
        contents = contents.to(device)
        question_ans = question_ans.to(device)
        sample_labels = sample_labels.to(device)
        sample_logics = sample_logics.to(device)
        # contents:batch_size*10*200,  question_ans:batch_size*100  ,sample_labels=batchsize
        # forward
        pred_labels = model.forward(contents, question_ans,sample_logics,sample_ids)  # pred_labels size=(batch,2)

        logger.info('epoch=%d, batch=%d/%d, loss=%.5f binary_acc=%.4f ' % (epoch, i, batch_cnt, 0, 0))

    logger.info('===== epoch=%d, batch_count=%d, epoch_average_loss=%.5f, avg_binary_acc=%.4f ====' % (epoch, batch_cnt, epoch_loss.avg, epoch_binary_acc.avg))

    return epoch_loss.avg, epoch_binary_acc.avg


def save_model(model, epoch_info, model_weight_path, checkpoint_path):
    """
    save model weight without embedding
    :param model:
    :param epoch:
    :param model_weight_path:
    :param checkpoint_path:
    :return:
    """
    # save model weight
    model_weight = model.state_dict()
    # del model_weight['embedding.embedding_layer.weight']

    torch.save(model_weight, model_weight_path)
    with open(checkpoint_path, 'w') as checkpoint_f:
        checkpoint_f.write(epoch_info + "\n")


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


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="train on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
    args = parser.parse_args()

    debug(args.config_path)
