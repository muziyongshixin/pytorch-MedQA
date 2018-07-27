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
import time
from queue import Queue
import traceback
from DL_Thread import DL_Thread
import itchat
from queue import Queue
import sys

training_info =Queue()


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
parser.add_argument('--itchat', '-i', required=False, dest='itchat',default=False)
parser.add_argument('--remark', required=False, dest='remark',default="")
args = parser.parse_args()


cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
host_name = socket.gethostname()
experiment_info = host_name + "_" + cur_time+"_"+args.remark

init_logging(experiment_info)
logger = logging.getLogger(__name__)
logger.info('========================  %s  ================================='%experiment_info)
if not args.itchat:
    if args.mode == 'preprocess':
        preprocess(args.config_path)
    elif args.mode == 'train':
        train(args.config_path,experiment_info)
    elif args.mode=='debug':
        debug(args.config_path,experiment_info)
    elif args.mode == 'test':
        test(args.config_path, experiment_info)
    elif args.mode == 'train_5c':
        train_5c(args.config_path,experiment_info)
    else:
        raise ValueError('Unrecognized mode selected.')
else:
    try:
        if args.mode =="train":
            dl_thread=DL_Thread(train,args.config_path,experiment_info,training_info)
        elif args.mode == 'train_5c':
            dl_thread = DL_Thread(train_5c,args.config_path, experiment_info,training_info)
        else:
            raise ValueError("itchat mode doesn\'t support %s mode"%args.mode)
        dl_thread.start()

        logger.info("dl_thread started, try to login wechat...")
        itchat.auto_login(hotReload=True, enableCmdQR=2)
        username=itchat.search_friends(nickName="zzzz")[0]["UserName"]
        itchat.send("training task begin... experiment_info:%s"%experiment_info,username)
        last_update_time= time.time()
        while dl_thread.exitcode == 0:
            if not training_info.empty():
                last_update_time=training_info.get()
            logger_info="itchat thread check==== train thread last update time=%s "%str(last_update_time)
            logger.info(logger_info)
            if time.time()-last_update_time>600:
                warning_info="itchat thread check==== train thread seems to go wrong, last update time is %s"%str(last_update_time)
                logger.info(warning_info)
                itchat.send(warning_info,username)
                break
            time.sleep(500)

        dl_thread.join()
        if dl_thread.finished:
            itchat.send("deep learning task finished",username)
            logger.info("itchat send finished info ")
        else:
            error_info=dl_thread.exc_traceback
            itchat.send(error_info,username)
            logger.info("itchat send error info==================\n %s" % error_info)
    except Exception as e:
        error_info = "dl_thread error info="+dl_thread.exc_traceback
        itchat.send(error_info, username)
        logger.info(error_info)
        main_thread_error_info= ''.join(traceback.format_exception(*sys.exc_info()))
        itchat.send(main_thread_error_info, username)
        logger.info(main_thread_error_info)
    finally:
        itchat.logout()





