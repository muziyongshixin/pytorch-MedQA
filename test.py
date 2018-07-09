#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

logger = logging.getLogger(__name__)

# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(str(net))
    logger.info('Total number of parameters: %d' % num_params)

def test(config_path, out_path):
    logger.info('------------MedQA v1.0 Evaluate--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

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
    model_choose = global_config['global']['model']
    logger.info("model choose is:   "+model_choose)
    dataset_h5_path = global_config['data']['dataset_h5']
    if model_choose == 'SeaReader':
        model = SeaReader(dataset_h5_path, device)
    elif model_choose == 'SimpleSeaReader':
        model = SimpleSeaReader(dataset_h5_path, device)
    elif model_choose == 'base':
        model_config = read_config('config/base_model.yaml')
        model = BaseModel(dataset_h5_path, model_config)
    elif model_choose == 'match-lstm':
        model = MatchLSTM(dataset_h5_path)
    elif model_choose == 'match-lstm+':
        model = MatchLSTMPlus(dataset_h5_path)
    elif model_choose == 'r-net':
        model = RNet(dataset_h5_path)
    else:
        raise ValueError('model "%s" in config file not recoginized' % model_choose)

    print_network(model)
    logger.info('dataParallel using %d GPU.....' % torch.cuda.device_count())
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()  # let training = False, make sure right dropout

    global init_embedding_weight
    init_embedding_weight = model.state_dict()['module.embedding.embedding_layer.weight']

    #criterion
    task_criterion = CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device)).to(device)
    gate_criterion = gate_Loss().to(device)
    embedding_criterion = Embedding_reg_L21_Loss().to(device)
    all_criterion = [task_criterion, gate_criterion, embedding_criterion]

    # training arguments
    logger.info('get test data loader ...')
    test_batch_size = global_config['train']['test_batch_size']
    batch_test_data = dataset.get_dataloader_test(test_batch_size, shuffle=False)

    # load model weight
    logger.info('loading model weight...')
    model_weight_path = global_config['data']['model_path']
    assert os.path.exists(model_weight_path), "not found model weight file on '%s'" % model_weight_path

    weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    if enable_cuda:
        weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(weight, strict=False)

    # forward
    logger.info('evaluate forwarding...')

    enable_char = False
    if model_choose == 'match-lstm+' or model_choose == 'r-net' or (
            model_choose == 'base' and model_config['encoder']['enable_char']):
        enable_char = True
    batch_size = global_config['test']['batch_size']
    # batch_dev_data = dataset.get_dataloader_dev(batch_size)
    batch_dev_data = list(dataset.get_batch_dev(batch_size))

    # to just evaluate score or write answer to file
    if out_path is None:
        test_avg_loss, test_avg_binary_acc, test_avg_problem_acc = eval_on_model(model=model,
                                                                                 criterion=all_criterion,
                                                                                 batch_data=batch_test_data,
                                                                                 epoch=0,
                                                                                 device=device,
                                                                                 enable_char=enable_char,
                                                                                 batch_char_func=dataset.gen_batch_with_char,
                                                                                 init_embedding_weight=init_embedding_weight)
        logger.info("test: test_avg_loss=%.4f, test_avg_binary_acc=%.4f, test_avg_problem_acc=%.4f" % (test_avg_loss, test_avg_binary_acc, test_avg_problem_acc))
    else:
        predict_ans = predict_on_model(model=model,
                                       batch_data=batch_dev_data,
                                       device=device,
                                       enable_char=enable_char,
                                       batch_char_func=dataset.gen_batch_with_char,
                                       id_to_word_func=dataset.sentence_id2word)
        samples_id = dataset.get_all_samples_id_dev()
        ans_with_id = dict(zip(samples_id, predict_ans))

        logging.info('writing predict answer to file %s' % out_path)
        with open(out_path, 'w') as f:
            json.dump(ans_with_id, f)

    logging.info('finished.')


def predict_on_model(model, batch_data, device, enable_char, batch_char_func, id_to_word_func):
    batch_cnt = len(batch_data)
    answer = []

    for bnum, batch in enumerate(batch_data):

        # batch data
        bat_context, bat_question, bat_context_char, bat_question_char, bat_answer_range = \
            batch_char_func(batch, enable_char=enable_char, device=device)

        _, tmp_ans_range, _ = model.forward(bat_context, bat_question, bat_context_char, bat_question_char)
        tmp_context_ans = zip(bat_context.cpu().data.numpy(),
                              tmp_ans_range.cpu().data.numpy())
        tmp_ans = [' '.join(id_to_word_func(c[a[0]:(a[1] + 1)])) for c, a in tmp_context_ans]
        answer += tmp_ans

        logging.info('batch=%d/%d' % (bnum, batch_cnt))

        # manual release memory, todo: really effect?
        del bat_context, bat_question, bat_answer_range, bat_context_char, bat_question_char
        del tmp_ans_range
        # torch.cuda.empty_cache()

    return answer


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="evaluate on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
    parser.add_argument('--output', '-o', required=False, dest='out_path')
    args = parser.parse_args()

    test(config_path=args.config_path, out_path=args.out_path)
