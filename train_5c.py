#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
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
from models.loss import gate_Loss, Embedding_reg_L21_Loss,delta_embedding_Loss,SVM_loss
from utils.load_config import init_logging, read_config
from utils.eval_5c import eval_on_model_5c
from utils.functions import pop_dict_keys
from IPython import embed
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from torch.nn import init
import random

logger = logging.getLogger(__name__)


# 保存本次实验的部分代码
def save_current_codes(des_path, global_config):
    if not os.path.exists(des_path):
        os.mkdir(des_path)

    train_file_path = os.path.realpath(__file__)  # eg：/n/liyz/videosteganography/train.py
    cur_work_dir, trainfile = os.path.split(train_file_path)  # eg：/n/liyz/videosteganography/

    new_train_path = os.path.join(des_path, trainfile)
    shutil.copyfile(train_file_path, new_train_path)

    config_file_path = cur_work_dir + "/config/global_config.yaml"
    config_file_name = 'global_config.yaml'
    new_config_file_path = os.path.join(des_path, config_file_name)
    shutil.copyfile(config_file_path, new_config_file_path)

    model_choose = global_config['global']['model']
    model_file_name = model_choose + ".py"
    model_file_path = cur_work_dir + "/models/" + model_file_name
    new_model_file_path = os.path.join(des_path, model_file_name)
    shutil.copyfile(model_file_path, new_model_file_path)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(str(net))
    logger.info('Total number of parameters: %d' % num_params)


# custom weights initialization called on model
def weights_init(m):
    for name, params in m.named_parameters():
        if name.find('decision_layer.4') != -1 or name.find('decision_layer.0') != -1:
            embed()
            init.xavier_uniform_(params, gain=init.calculate_gain('relu'))
        elif name.find('conv') != -1:
            pass
        elif name.find('norm') != -1:
            pass


def train_5c(config_path, experiment_info, thread_queue):
    logger.info('------------MedQA v1.0 Train--------------')
    logger.info('============================loading config file... print config file =========================')
    global_config = read_config(config_path)
    logger.info(open(config_path).read())
    logger.info('^^^^^^^^^^^^^^^^^^^^^^   config file info above ^^^^^^^^^^^^^^^^^^^^^^^^^')
    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    global gpu_nums, init_embedding_weight, batch_test_data,batch_dev_data, tensorboard_writer, test_epoch,embedding_layer_name,val_epoch,global_config,best_valid_acc
    test_epoch = 0
    val_epoch = 0

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
    logger.info('Using dataset path is : %s' % dataset_h5_path)
    logger.info('### Using model is: %s ###' % model_choose)
    if model_choose == 'SeaReader':
        model = SeaReader(dataset_h5_path, device)
    elif model_choose == 'SimpleSeaReader':
        model = SimpleSeaReader(dataset_h5_path, device)
    elif model_choose == 'TestModel':
        model = TestModel(dataset_h5_path, device)
    elif model_choose == 'cnn_model':
        model = cnn_model(dataset_h5_path, device)
    elif model_choose == 'SeaReader_5c':
        model = SeaReader_5c(dataset_h5_path, device)
    elif model_choose == 'SeaReader_v2':
        model = SeaReader_v2(dataset_h5_path, device)
    elif model_choose == 'SeaReader_v3':
        model =SeaReader_v3(dataset_h5_path,device)
    elif model_choose=='SeaReader_v4':
        model=SeaReader_v4(dataset_h5_path,device)
    elif model_choose=='No_content_model':
        model= No_content_model(dataset_h5_path)
    else:
        raise ValueError('model "%s" in config file not recognized' % model_choose)
    print_network(model)

    gpu_nums = torch.cuda.device_count()
    logger.info('dataParallel using %d GPU.....' % gpu_nums)
    if gpu_nums > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    # weights_init(model)

    # embedding_layer_name = 'module.embedding.embedding_layer.weight'
    # for  name in model.state_dict().keys():
    #     if 'embedding_layer.weight' in name:
    #         embedding_layer_name=name
    #         break
    # init_embedding_weight = model.state_dict()[embedding_layer_name].clone()

    task_criterion = CrossEntropyLoss().to(device)
    gate_criterion = gate_Loss().to(device)
    embedding_criterion = delta_embedding_Loss(c=1).to(device)
    all_criterion = [task_criterion, gate_criterion, embedding_criterion]

    # optimizer
    optimizer_choose = global_config['train']['optimizer']
    optimizer_lr = global_config['train']['learning_rate']
    optimizer_eps = float(global_config['train']['eps'])
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_choose == 'adamax':
        optimizer = optim.Adamax(optimizer_param)
    elif optimizer_choose == 'adadelta':
        optimizer = optim.Adadelta(optimizer_param)
    elif optimizer_choose == 'adam':
        optimizer = optim.Adam(optimizer_param, lr=optimizer_lr, eps=optimizer_eps)
    elif optimizer_choose == 'sgd':
        optimizer = optim.SGD(optimizer_param,
                              lr=optimizer_lr)
    else:
        raise ValueError('optimizer "%s" in config file not recoginized' % optimizer_choose)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

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
        if not global_config['train']['keep_embedding']:
            del weight['module.embedding.embedding_layer.weight']  # 删除掉embedding层的参数 ，避免尺寸不对的问题
        model.load_state_dict(weight, strict=False)

    # training arguments
    logger.info('start training............................................')
    train_batch_size = global_config['train']['batch_size']
    valid_batch_size = global_config['train']['valid_batch_size']
    test_batch_size = global_config['train']['test_batch_size']

    batch_train_data = dataset.get_dataloader_train(train_batch_size, shuffle=False)
    batch_dev_data = dataset.get_dataloader_dev(valid_batch_size, shuffle=False)
    batch_test_data = dataset.get_dataloader_test(test_batch_size, shuffle=False)

    clip_grad_max = global_config['train']['clip_grad_norm']

    # tensorboardX writer
    tensorboard_writer = SummaryWriter(log_dir=os.path.join('tensorboard_logdir', experiment_info))

    save_cur_experiment_code_path = "savedcodes/" + experiment_info
    save_current_codes(save_cur_experiment_code_path, global_config)

    best_valid_acc = None
    # every epoch
    for epoch in range(global_config['train']['epoch']):
        # train
        model.train()  # set training = True, make sure right dropout
        train_avg_loss, train_avg_problem_acc = train_on_model(model=model,
                                                               criterion=all_criterion,
                                                               optimizer=optimizer,
                                                               batch_data=batch_train_data,
                                                               epoch=epoch,
                                                               clip_grad_max=clip_grad_max,
                                                               device=device,
                                                               thread_queue=thread_queue,
                                                               experiment_info=experiment_info)

        # # evaluate
        # with torch.no_grad():
        #     model.eval()  # let training = False, make sure right dropout
        #     val_avg_loss, val_avg_problem_acc = eval_on_model_5c(model=model,
        #                                                          criterion=all_criterion,
        #                                                          batch_data=batch_dev_data,
        #                                                          epoch=epoch,
        #                                                          device=device,
        #                                                          init_embedding_weight=init_embedding_weight,
        #                                                          eval_dataset='dev')

            # test_avg_loss, test_avg_binary_acc, test_avg_problem_acc=eval_on_model(model=model,
            #                                                                       criterion=all_criterion,
            #                                                                       batch_data=batch_test_data,
            #                                                                       epoch=epoch,
            #                                                                       device=device,
            #                                                                       enable_char=enable_char,
            #                                                                       batch_char_func=dataset.gen_batch_with_char,
            #                                                                       init_embedding_weight=init_embedding_weight)
        tensorboard_writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        tensorboard_writer.add_scalar("train/avg_loss", train_avg_loss, epoch)
        tensorboard_writer.add_scalar("train/problem_acc", train_avg_problem_acc, epoch)
        # tensorboard_writer.add_scalar("test/avg_loss", test_avg_loss, epoch)
        # tensorboard_writer.add_scalar("test/binary_acc", test_avg_binary_acc, epoch)
        # tensorboard_writer.add_scalar("test/problem_acc", test_avg_problem_acc, epoch)

        #  adjust learning rate
        scheduler.step(train_avg_loss)

    logger.info('finished.................................')
    tensorboard_writer.close()


def train_on_model(model, criterion, optimizer, batch_data, epoch, clip_grad_max, device, thread_queue,experiment_info):
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
    global test_epoch
    global val_epoch
    global best_valid_acc
    epoch_loss = AverageMeter()
    epoch_problem_acc = AverageMeter()
    batch_cnt = len(batch_data)
    for i, batch in enumerate(batch_data, 0):
        optimizer.zero_grad()
        model.train()
        # batch data
        # bat_context, bat_question, bat_context_char, bat_question_char, bat_answer_range = batch_char_func(batch, enable_char=enable_char, device=device)
        # contents, question_ans, sample_labels, sample_ids, sample_categorys, sample_logics = batch
        contents, question_ans, question, ans, sample_ids, sample_labels, sample_categorys, sample_logics= batch
        # # 通过随机跳过实现 shuffle效果
        # if random.randint(0, 10000) % 2 == 0 and i!=0:
        #     continue

        if len(sample_ids) % (5 * gpu_nums) != 0:
            logger.info("batch num is incorrect, ignore this batch")
            continue
        contents = contents.to(device)
        question_ans = question_ans.to(device)
        question=question.to(device)
        ans=ans.to(device)
        sample_labels = sample_labels.to(device)
        sample_labels = torch.argmax(sample_labels.resize_(int(sample_labels.size()[0] / 5), 5), dim=1)
        sample_logics = sample_logics.to(device)
        # contents:batch_size*10*200,  question_ans:batch_size*100  ,sample_labels=batchsize
        # forward
        pred_labels = model.forward(contents, question_ans, sample_logics,question,ans)  # pred_labels size=(batch,2)
        # get task loss
        task_loss = criterion[0].forward(pred_labels, sample_labels)
        # gate_loss
        # gate_loss=criterion[1].forward(mean_gate_val)
        gate_loss = 0
        # embedding regularized loss
        embedding_loss = criterion[2].forward(model.delta_embedding.embedding_layer.weight)
        # loss=task_loss+gate_loss+embedding_loss
        loss = task_loss + embedding_loss
        # loss=task_loss
        loss.backward()

        # print(model.embedding.embedding_layer.weight.grad)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max)  # fix gradient explosion
        optimizer.step()  # update parameters

        # logging
        batch_loss = loss.item()
        epoch_loss.update(batch_loss, len(sample_ids))

        # binary_acc = compute_binary_accuracy(pred_labels.data, sample_labels.data)
        problem_acc = compute_problems_accuracy_5c(pred_labels.data, sample_labels.data, sample_ids)

        # epoch_binary_acc.update(binary_acc.item(), len(sample_ids))
        epoch_problem_acc.update(problem_acc, int(len(sample_ids) / 5))

        logger.info('epoch=%d, batch=%d/%d, embedding_loss=%.5f loss=%.5f problem_acc=%.4f ' % (
        epoch, i, batch_cnt,embedding_loss, batch_loss, epoch_problem_acc.val))

        # 线程间通信，用于存放时间
        if thread_queue.qsize() != 0:
            thread_queue.queue.clear()
        thread_queue.put(time.time())
        # manual release memory, todo: really effect?
        del contents, question_ans, sample_labels, sample_ids
        del pred_labels, loss

        #do test
        if i % 200 == 0:
            model.eval()
            with torch.no_grad():
                test_avg_loss, test_avg_problem_acc = eval_on_model_5c(model=model,
                                                                       criterion=criterion,
                                                                       batch_data=batch_test_data,
                                                                       epoch=test_epoch,
                                                                       device=device,
                                                                       eval_dataset='test')
                tensorboard_writer.add_scalar("test/avg_loss", test_avg_loss, test_epoch)
                tensorboard_writer.add_scalar("test/problem_acc", test_avg_problem_acc, test_epoch)
                test_epoch += 1
        # do evaluate
        if (batch_cnt>2000 and i%2000==0 and i!=0) or (batch_cnt<2000 and i%2000==0) :
            with torch.no_grad():
                model.eval()  # let training = False, make sure right dropout
                val_avg_loss, val_avg_problem_acc = eval_on_model_5c(model=model,
                                                                     criterion=criterion,
                                                                     batch_data=batch_dev_data,
                                                                     epoch=val_epoch,
                                                                     device=device,
                                                                     eval_dataset='dev')
                tensorboard_writer.add_scalar("val/avg_loss", val_avg_loss, val_epoch)
                tensorboard_writer.add_scalar("val/problem_acc", val_avg_problem_acc, val_epoch)
                # save model when best f1 score
                if best_valid_acc is None or val_avg_problem_acc > best_valid_acc:
                    epoch_info = 'epoch=%d, time=%s, val_problem_acc=%.4f' % (
                        val_epoch, time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime()), val_avg_problem_acc)
                    save_model(model,
                               epoch_info=epoch_info,
                               model_weight_path=global_config['data']['model_weight_dir'] + experiment_info + "_model_weight.pt",
                               checkpoint_path=global_config['data']['checkpoint_path'] + experiment_info + "_save.log")
                    logger.info("=============================  saving model weight, saving info=%s ===================================" % epoch_info)
                    best_valid_acc = val_avg_problem_acc
                val_epoch+=1
    logger.info('===== epoch=%d, batch_count=%d, epoch_average_loss=%.5f, avg_problem_acc=%.4f ====' % (
    epoch, batch_cnt, epoch_loss.avg, epoch_problem_acc.avg))

    return epoch_loss.avg, epoch_problem_acc.avg


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
    with open(checkpoint_path, 'a') as checkpoint_f:
        checkpoint_f.write(epoch_info + "\n")


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
        pred_label_index = torch.argmax(pred_labels, dim=1)  # 10*1
        difference_index = pred_label_index - real_labels  # 10*1 正确的题是0，错误的地方非零
        correct_count = difference_index.eq(0).sum()  # 正确题目的数量
        accuracy = correct_count.float().item() / problem_num
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

    train(args.config_path)
