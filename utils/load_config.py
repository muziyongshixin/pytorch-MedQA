#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import sys

import time
import yaml
import logging.config
from IPython import  embed


def init_logging(log_file_name, config_path='config/logging_config.yaml' ):
    """
    initial logging module with config
    :param config_path:
    :return:
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read())

        experiment_info = "logs/"+log_file_name
        config["handlers"]["info_file_handler"]["filename"]=experiment_info+".debug_log"
        config["handlers"]["time_file_handler"]["filename"]=experiment_info+".debug_log"

        logging.config.dictConfig(config)
    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        logging.basicConfig(level=logging.DEBUG)


def read_config(config_path='config/global_config.yaml'):
    """
    store the global parameters in the project
    :param config_path:
    :return:
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read())
        return config

    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        exit(-1)
