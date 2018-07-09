#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liyz'

import json
import logging
import os

import h5py
import numpy as np

from utils.functions import pad_sequences

logger = logging.getLogger(__name__)

question_ans_longer_100_count=0

class PreprocessData:
    """
    preprocess dataset and glove embedding to hdf5 files
    """
    padding = '__padding__'  # id = 0
    padding_idx = 0          # also same to char level padding values
    answer_padding_idx = -1

    __compress_option = dict(compression="gzip", compression_opts=9, shuffle=False)

    def __init__(self, global_config):
        # data config
        self.__dev_path = ''
        self.__train_path = ''
        self.__export_medqa_path = ''
        self.__embedding_path = ''
        self.__embedding_size = 200
        self.__ignore_max_len = 200
        self._ignore_max_ques_ans_len=100
        self.__load_config(global_config)

        self.__question_ans_longer_100_count = 0
        self.__content_longer_200_count = 0

        # preprocess config
        self.__max_context_token_len = 0
        self.__max_question_ans_token_len = 0

        # self.__max_answer_len = 0

        # temp data
        self.__word2id = {self.padding: 0}
        self.__char2id = {self.padding: 0}  # because nltk word tokenize will replace '"' with '``'
        self.__word2vec = {self.padding: [0. for i in range(self.__embedding_size)]}
        self.__oov_num = 0

        # data need to store in hdf5 file
        self.__meta_data = {'id2vec': [[0. for i in range(self.__embedding_size)]],
                            'id2word': [self.padding],
                            'id2char': [self.padding]}
        self.__data = {}
        self.__attr = {}

    def __load_config(self, global_config):
        """
        load config from a dictionary, such as dataset path
        :param global_config: dictionary
        :return:
        """
        data_config = global_config['data']
        self.__train_path = data_config['dataset']['train_path']
        self.__dev_path = data_config['dataset']['dev_path']
        self.__test_path = data_config['dataset']['test_path']
        self.__export_medqa_path = data_config['dataset_h5']
        self.__embedding_path = data_config['embedding_path']
        self.__ignore_max_len = data_config['ignore_max_len']
        self._ignore_max_ques_ans_len = data_config['ignore_max_ques_ans_len']
        self.__embedding_size = int(global_config['model']['encoder']['word_embedding_size'])

    #将一个文件里的所有的sample合并到一个大的list里，list里的每一个元素是一个题目的问题、答案以及content
    def __read_json(self, path):

        """
        read json format file from raw json  text
        :param path: squad file path
        :return:
        """
        contexts_qas=[]
        with open(path, 'r') as f:
            for line in f:
                cur_data=json.loads(line)
                contexts_qas.append(cur_data)
        return contexts_qas

    def __build_data(self, contexts_qas, training):
        """
        handle raw data to (question+answer,[content1,content2....content10]) with word id representation
        :param contexts_qas: 一个问题，5个选项，每个选项10个content
        :return:
        """
        candidates=["A","B","C","D","E"]

        contents_wids=[]    # 里面的每一个元素是一个list A，A里有10个元素Bi，Bi的每一个元素的一个content的所有单词的id表示
        question_ans_wids=[] # 里面的每一个元素是一个list A，A的每一个元素是question和answer连起来之后的句子的所有的词的id表示
        samples_ids = []     # 里面的每一个元素的一个字符串，例如56be4db0acb8001400a502ec_A,表示一个问题的A选项的训练样例
        samples_labels=[]   # 里面每一个元素是0或者1,0表示错误的问题和答案样例，1表示正确的

        for question_grp in contexts_qas:
            cur_question=question_grp["question"]
            cur_quesion_id=question_grp["id"]
            cur_correct_answer=question_grp["correct"]

            for candidate in candidates:
                if cur_correct_answer==candidate:
                    cur_label=1 ##
                else:
                    cur_label=0  ##

                cur_sample_id=cur_quesion_id+"_"+candidate ##

                # cur_answer_text=question_grp[candidate]["text"]
                # cur_question_ans=cur_question+" "+cur_answer_text ##
                cur_question_ans =question_grp[candidate]["text"] ## 因为这批数据的text里已经加了question
                cur_facts=question_grp[candidate]["facts"]

                contents_ids=[] ##  #一个选项里的10个content的ids list的一个总的list
                for fact in cur_facts:
                    cur_content=fact["content"]
                    content_words=cur_content.split(" ")


                    if training and len(content_words)>self.__ignore_max_len:
                        content_words=content_words[0:self.__ignore_max_len]
                        self.__content_longer_200_count+=1

                    self.__max_context_token_len = max(self.__max_context_token_len, len(content_words))
                    cur_content_id=self.__sentence_to_id(content_words) # 一个content的words的id表示
                    contents_ids.append(cur_content_id)

                cur_question_ans_words=cur_question_ans.split(" ")
                if training and len(cur_question_ans_words)>self._ignore_max_ques_ans_len:
                    cur_question_ans_words=cur_question_ans_words[0:self._ignore_max_ques_ans_len]
                    self.__question_ans_longer_100_count+=1

                self.__max_question_ans_token_len=max(self.__max_question_ans_token_len,len(cur_question_ans_words))
                cur_question_ans_id = self.__sentence_to_id(cur_question_ans_words) ##  #question和选项一起的id表示

                contents_wids.append(contents_ids)
                question_ans_wids.append(cur_question_ans_id)
                samples_ids.append(cur_sample_id)
                samples_labels.append(cur_label)

        return {'contents': contents_wids,
                'question_ans': question_ans_wids,
                'samples_ids': samples_ids,
                'samples_labels' : samples_labels}

    def __sentence_to_id(self, sentence):
        """
        transform a sentence to word index id representation
        :param sentence: tokenized sentence
        :return: word ids
        """
        ids = [] #保存一个句子的所有单词的id  etc。[1,2,4,1,2,4,2,1]
        for word in sentence:
            if word not in self.__word2id: #说明word不在词典中，因为word2id已经是全词典大小
                # self.__word2id[word] = len(self.__word2id)
                # self.__meta_data['id2word'].append(word)
                # whether OOV
                # if word in self.__word2vec:   #__word2vec是由word2vec_embedding文件生成的
                #     self.__meta_data['id2vec'].append(self.__word2vec[word])
                # else:
                self.__oov_num += 1
                logger.debug('No.%d OOV word %s' % (self.__oov_num, word))
                # self.__meta_data['id2vec'].append([0. for i in range(self.__embedding_size)])  #如果不在vocabulary里面用0向量表示
                ids.append(0)
            else:
                ids.append(self.__word2id[word])

        return ids

    def __update_to_char(self, sentence):
        """
        update char2id
        :param sentence: raw sentence
        """
        for ch in sentence:
            if ch not in self.__char2id:
                self.__char2id[ch] = len(self.__char2id)
                self.__meta_data['id2char'].append(ch)

    def __handle_word2vec(self):
        """
        handle word2vec embeddings, restore embeddings with dictionary
        :return:
        """
        logger.debug("read pertrained word2vec file from text file %s" % self.__embedding_path)
        if not os.path.exists(self.__embedding_path):
            raise ValueError('word2vec file "%s" not found' % self.__embedding_path)
        word_num = 0
        with open(self.__embedding_path,"r") as ebf:
            for line in ebf:
                line_split = line.split(' ')
                cur_word=line_split[0]
                self.__word2vec[cur_word] = [float(x) for x in line_split[1:]]
                self.__word2id[cur_word]=word_num+1 #word 的id 从1开始，0留给OOV和padding
                self.__meta_data["id2vec"].append([float(x) for x in line_split[1:]])
                self.__meta_data['id2word'].append(cur_word)
                word_num += 1
                if word_num % 10000 == 0:
                    logger.debug('handle word No.%d' % word_num)
        logger.debug("pertrained word2vec file reading completed")
        # with zipfile.ZipFile(self.__glove_path, 'r') as zf:
        #     if len(zf.namelist()) != 1:
        #         raise ValueError('glove file "%s" not recognized' % self.__glove_path)
        #
        #     glove_name = zf.namelist()[0]
        #
        #     word_num = 0
        #     with zf.open(glove_name) as f:
        #         for line in f:
        #             line_split = line.decode('utf-8').split(' ')
        #             self.__word2vec[line_split[0]] = [float(x) for x in line_split[1:]]
        #
        #             word_num += 1
        #             if word_num % 10000 == 0:
        #                 logger.debug('handle word No.%d' % word_num)




    def __pad_contents_sequences(self,all_contents):
        new_all_contents=[]
        for contents in all_contents:
            new_contents=pad_sequences(contents,
                                     maxlen=self.__max_context_token_len,
                                     padding='post',
                                     value=self.padding_idx)
            new_all_contents.append(new_contents)
        result=np.stack(new_all_contents)
        return result

    def run(self):
        """
        main function to generate hdf5 file
        :return:
        """
        logger.info('handle word2vec file...')
        self.__handle_word2vec() #读取word2vec_embedding文件，获得word to vector字典，etc. {a:[0.99,0.23,-0.12,0.33]}

        logger.info('read train/dev/test json file...')
        train_context_qas = self.__read_json(self.__train_path)
        logger.info('train json file loading completed')
        dev_context_qas = self.__read_json(self.__dev_path)
        logger.info('dev json file loading completed')
        test_context_qas = self.__read_json(self.__test_path)
        logger.info('test json file loading completed')

        logger.info('transform word to id...')
        train_cache_nopad = self.__build_data(train_context_qas, training=True)
        dev_cache_nopad = self.__build_data(dev_context_qas, training=True)
        test_cache_nopad = self.__build_data(test_context_qas, training=True)

        self.__attr['train_size'] = len(train_cache_nopad['samples_labels'])
        self.__attr['dev_size'] = len(dev_cache_nopad['samples_labels'])
        self.__attr['test_size'] = len(test_cache_nopad['samples_labels'])

        self.__attr['word_dict_size'] = len(self.__word2id)
        self.__attr['char_dict_size'] = len(self.__char2id)
        self.__attr['embedding_size'] = self.__embedding_size
        self.__attr['oov_word_num'] = self.__oov_num
        self.__attr['max_question_ans_token_len']=self.__max_question_ans_token_len,
        self.__attr['max_context_token_len'] = self.__max_context_token_len,

        logger.debug("self.__question_ans_longer_100_count======="+str(self.__question_ans_longer_100_count))
        logger.debug("self.__content_longer_200_count======="+str(self.__content_longer_200_count))

        logger.info('padding id vectors...')

        logger.info('padding test id vectors...')
        self.__data['test'] = {
            'contents': self.__pad_contents_sequences(test_cache_nopad['contents']),
            'question_ans': pad_sequences(test_cache_nopad['question_ans'],
                                          maxlen=self.__max_question_ans_token_len,
                                          padding='post',
                                          value=self.padding_idx),
            'samples_ids': np.array(test_cache_nopad['samples_ids']),
            'samples_labels': np.array(test_cache_nopad['samples_labels'])}

        logger.info('padding train id vectors...')
        self.__data['train'] = {
            'contents': self.__pad_contents_sequences(train_cache_nopad['contents']),
            'question_ans': pad_sequences(train_cache_nopad['question_ans'],
                                      maxlen=self.__max_question_ans_token_len,
                                      padding='post',
                                      value=self.padding_idx),
            'samples_ids':    np.array(train_cache_nopad['samples_ids']),
            'samples_labels': np.array(train_cache_nopad['samples_labels'])}

        logger.info('padding dev id vectors...')
        self.__data['dev'] = {
            'contents':       self.__pad_contents_sequences(dev_cache_nopad['contents']),
            'question_ans':   pad_sequences(dev_cache_nopad['question_ans'],
                                      maxlen=self.__max_question_ans_token_len,
                                      padding='post',
                                      value=self.padding_idx),
            'samples_ids':    np.array(dev_cache_nopad['samples_ids']),
            'samples_labels': np.array(dev_cache_nopad['samples_labels'])}

        logger.info('export to hdf5 file...')
        self.__export_squad_hdf5()

        logger.info('finished.')



    def __export_squad_hdf5(self):
        """
        export squad dataset to hdf5 file
        :return:
        """
        f = h5py.File(self.__export_medqa_path, 'w')
        str_dt = h5py.special_dtype(vlen=str)

        # attributes
        for attr_name in self.__attr:
            f.attrs[attr_name] = self.__attr[attr_name]
            print(attr_name,  self.__attr[attr_name])


        # meta_data
        id2word = np.array(self.__meta_data['id2word'], dtype=np.str)
        id2char = np.array(self.__meta_data['id2char'], dtype=np.str)
        id2vec = np.array(self.__meta_data['id2vec'], dtype=np.float32)
        f_meta_data = f.create_group('meta_data')

        meta_data = f_meta_data.create_dataset('id2word', id2word.shape, dtype=str_dt, **self.__compress_option)
        meta_data[...] = id2word

        meta_data = f_meta_data.create_dataset('id2char', id2char.shape, dtype=str_dt, **self.__compress_option)
        meta_data[...] = id2char

        meta_data = f_meta_data.create_dataset('id2vec', id2vec.shape, dtype=id2vec.dtype, **self.__compress_option)
        meta_data[...] = id2vec

        # data
        f_data = f.create_group('data')
        for key, value in self.__data.items():
            data_grp = f_data.create_group(key)
            for sub_key, sub_value in value.items():
                logger.debug(str(key)+" "+str(sub_key)+" "+str(sub_value.shape))
                cur_dtype = str_dt if sub_value.dtype.type is np.str_ else sub_value.dtype
                data = data_grp.create_dataset(sub_key, sub_value.shape, dtype=cur_dtype,**self.__compress_option)
                data[...] = sub_value

        f.flush()
        f.close()
