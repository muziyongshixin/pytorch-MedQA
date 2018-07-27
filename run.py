#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import time

__author__ = 'han'

import logging
import argparse
from test import test
from debug import debug
from train import train
from train_5c import train_5c
from utils.load_config import init_logging, read_config
from dataset.preprocess_data import PreprocessData
from queue import Queue

training_info=Queue()

def preprocess(config_path):
    logger.info('------------Preprocess SQuAD dataset--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

    logger.info('preprocess data...')
    pdata = PreprocessData(global_config)
    pdata.run()


parser = argparse.ArgumentParser(description="preprocess/train/test the model")
parser.add_argument('mode', help='preprocess or train or test')
parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
parser.add_argument('--output', '-o', required=False, dest='out_path')
parser.add_argument('--remark', required=False, dest='remark',default="")
args = parser.parse_args()


cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
host_name = socket.gethostname()
experiment_info = host_name + "_" + cur_time+"_"+args.remark

init_logging(experiment_info)
logger = logging.getLogger(__name__)
logger.info('========================  %s  ================================='%experiment_info)
if args.mode == 'preprocess':
    preprocess(args.config_path)
elif args.mode == 'train':
    train(args.config_path,experiment_info,training_info)
elif args.mode=='debug':
    debug(args.config_path,experiment_info)
elif args.mode == 'test':
    test(args.config_path, experiment_info)
elif args.mode == 'train_5c':
    train_5c(args.config_path,experiment_info,training_info)
else:
    raise ValueError('Unrecognized mode selected.')

