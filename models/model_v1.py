# -*- coding:utf-8 -*-

import time, sys, operator, math, random, argparse, json, ntpath, os, shutil, heapq
from functools import reduce
from pathlib import Path

import jieba
from gensim.models import word2vec
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers

# ----- utilities

class perf_checker(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.last_perf_counter = time.perf_counter()

        return self

    def __exit__(self, type, value, traceback):
        pass

    def mark_time(self):
        self.last_perf_counter = time.perf_counter()

    def time_used(self):
        return time.perf_counter() - self.last_perf_counter

    def str_time_used(self):
        return 'used {0:.7f} sec'.format(self.time_used())

    def print_time_used(self):
        print(self.str_time_used())

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_tfrecords_count(filename):
    count = 0
    for record in tf.python_io.tf_record_iterator(filename):
        count += 1
    return count

def set_global_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

class stat_helper():
    def __init__(self):
        self.data = {}
        pass

    def add_data(self, data_dict):
        if self.data:
            for k in self.data.keys():
                self.data[k].append(data_dict[k])
        else:
            for k, v in data_dict.items():
                self.data[k] = [v]

    def get_weighted_avg(self, weight_key=None):
        result = {}
        if weight_key is None:
            for k, v in self.data.items():
                result[k] = np.average(v)
        else:
            weights = self.data[weight_key]
            for k, v in self.data.items():
                if k != weight_key:
                    result[k] = np.average(v, weights=weights)

        return result

    def get_max(self, key):
        if key in self.data:
            return np.max(self.data[key])
        else:
            return -1

    def get_min(self, key):
        if key in self.data:
            return np.min(self.data[key])
        else:
            return -1

def rm_r(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
    else:
        shutil.rmtree(path)


class saver_model_helper(object):
    def __init__(self, path, max_keep = -1):
        self.path = path
        self.max_keep = max_keep

        self.minheap = []
        self.metric_map = {}

    def prepare_cur_dir(self, epoch, metrics):

        dir = '{0}model-{1}'.format(self.path, epoch)
        rm_r(dir)

        if self.max_keep >= 0:

            if epoch in self.metric_map:
                return dir

            if len(self.minheap) >= self.max_keep and metrics < self.minheap[0][0]:
                return None

            heapq.heappush(self.minheap, (metrics, epoch))
            self.metric_map[epoch] = metrics

            self_purge_flag = False

            while len(self.minheap) > self.max_keep:
                purge_epoch = self.minheap[0][1]
                if epoch == purge_epoch:
                    self_purge_flag = True
                purge_dir = '{0}model-{1}'.format(self.path, purge_epoch)

                index_file = purge_dir + '.index'
                rm_r(index_file)

                data_file = purge_dir + '.data-00000-of-00001'
                rm_r(data_file)

                rm_r(purge_dir)

                heapq.heappop(self.minheap)
                self.metric_map.pop(purge_epoch, None)

            if self_purge_flag:
                return None

        return dir

    def prepare_final_dir(self, final_suffix = 'model-latest'):

        # final_suffix could be 'model-latest'
        # dir = '{0}model-latest'.format(self.path, self.seq_id)
        dir = '{0}{1}'.format(self.path, final_suffix)

        rm_r(dir)

        return dir

# ----- test procs

def parse_tf(example_proto):
    features = {'id': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=''),
                'label': tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64, default_value=0, allow_missing=True),
                'feature': tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64, default_value=0, allow_missing=True),
                'ins_f': tf.FixedLenSequenceFeature(shape=(), dtype=tf.float32, default_value=0, allow_missing=True),
                }
    parsed_features = tf.parse_single_example(example_proto, features)

    return parsed_features['id'], parsed_features['label'], parsed_features['feature'], parsed_features['ins_f']

def test():
    #fn = '../data/tfrecords/train_0706_312k_zjall_e297k_es0625.tfrecords'
    fn = '../data/tfrecords/val_0704_1k_e297k_es0625.tfrecords'
    out_fn = '../data/tfrecords/test.tfrecords'

    tf.reset_default_graph()
    read_batch_size = tf.placeholder(tf.int64, shape=[], name='read_batch_size')
    read_data = tf.data.TFRecordDataset([fn]).map(lambda x: parse_tf(x)).batch(read_batch_size)
    read_iter = read_data.make_initializable_iterator()

    data_handle = tf.placeholder(tf.string, shape=[], name='data_handle')
    iterator = tf.data.Iterator.from_string_handle(data_handle,
                                                   (tf.string, tf.int64, tf.int64, tf.float32),
                                                   (tf.TensorShape([None]),
                                                    tf.TensorShape([None, 5]),
                                                    tf.TensorShape([None, 5500]),
                                                    tf.TensorShape([None, 2]))
                                                   )
    batch_id, batch_l, batch_f, batch_if = iterator.get_next()

    data = []
    batch_size = 1024

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        read_handle = sess.run(read_iter.string_handle())

        sess.run(read_iter.initializer, feed_dict={read_batch_size: batch_size})

        while True:
            try:
                b_id, b_l, b_f, b_if = sess.run([batch_id, batch_l, batch_f, batch_if], feed_dict={data_handle: read_handle})

                batch = [[item[0], item[1], item[2], item[3]] for item in zip(b_id, b_l, b_f, b_if)]

                data.extend(batch)
            except tf.errors.OutOfRangeError:
                break

    print(len(data))

    writer = tf.python_io.TFRecordWriter(out_fn)

    for item in data:
        example = tf.train.Example(features=
        tf.train.Features(feature={
            'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item[0]])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=item[1])),
            'feature': tf.train.Feature(int64_list=tf.train.Int64List(value=item[2])),
            'ins_f': tf.train.Feature(float_list=tf.train.FloatList(value=item[3])),
        }))
        writer.write(example.SerializeToString())

    # important! ensuring file closed correctly
    writer.close()

    # count = 0
    #
    # sess = tf.Session()
    #
    # for record in tf.python_io.tf_record_iterator(fn):
    #     count += 1
    #     data = sess.run(parse_tf(record))
    #
    #     print(count)
    #
    # return count

# --------------------------------------------------

class MedExamData(object):
    '''
    class for converting data to tfrecord file
    or load es_file and return numpy array
    '''
    def __init__(self, args, default_unkown_index=0):

        if args.wordcut_file is not None:
            self.need_cut = True
            jieba.load_userdict(args.wordcut_file)
        else:
            self.need_cut = False

        self.num_vocab, self.embed_dim, self.embed_dict, self.embed_vec = self.load_embedding(args.embedding_file)

        self.default_unknown_index = default_unkown_index # default unknown word index

        self.norm_flag = args.cut_norm

        pass

    # load embedding file, add 0 as unknown embedding
    def load_embedding(self, embedding_file):
        # first check embedding file existence
        if not Path(embedding_file).is_file():
            print('Could not load embedding file!')  # ERROR!
            return

        # load actual embedding, generate word mapping and actual embedding value
        with perf_checker() as pc:

            embedding_dict = {}
            embedding_vec = []
            embedding_dim = 0
            num_vocab = 0

            with open(embedding_file, 'r', encoding='utf-8') as embed:
                i = 0
                for line in embed:
                    if i == 0:  # load data stat from file head line
                        num_vocab, embedding_dim, *temp = line.rstrip('\n').split(' ')
                        num_vocab = int(num_vocab)
                        embedding_dim = int(embedding_dim)
                        print('Embedding has', num_vocab, 'words with', embedding_dim, 'dim embedding')

                        # add unknown embedding
                        embedding_vec.append([0] * embedding_dim)
                    else:
                        w, *w_embed_vec = line.rstrip('\n').split(' ')
                        embedding_dict[w] = i  # add_mapping
                        embedding_vec.append([float(val) for val in w_embed_vec])

                    i += 1

            print('Loaded embedding in {0:.7f} sec'.format(pc.time_used()))

        return num_vocab, embedding_dim, embedding_dict, np.reshape(embedding_vec, newshape=(num_vocab + 1, embedding_dim))

    def get_term_index(self, term):

        if term in self.embed_dict:
            return self.embed_dict[term]
        else:
            # print('Un-recognized term: "{0}"'.format(term))
            return self.default_unknown_index

    def write_tfrecord(self, data, filename):
        writer = tf.python_io.TFRecordWriter(filename)

        for item in data:
            example = tf.train.Example(features=
            tf.train.Features(feature={
                'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item[0].encode('utf-8')])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=item[1])),
                'feature': tf.train.Feature(int64_list=tf.train.Int64List(value=item[2])),
                'ins_f': tf.train.Feature(float_list=tf.train.FloatList(value=item[3])),
            }))
            writer.write(example.SerializeToString())

        # important! ensuring file closed correctly
        writer.close()

    # combine a list of es files, do train/val/test splitting and convert them to tfrecord files (smaller in file size)
    def convert_es_to_tfrecord(self, es_file_list, train_out_file, validate_out_file, test_out_file, split_stratified, shuffle_seed, validate_ratio=0.1, test_ratio=0.0, text_truncate_length=100, topn=10):

        data_list = []
        if split_stratified is True:
            for f in es_file_list:
                data_f, data_if, data_l, data_id = self.load_data_from_es_new(f, text_truncate_length, topn) # here extra data is not needed

                data_list.append(list(zip(data_id, data_l, data_f, data_if)))
        else:
            data_f, data_if, data_l, data_id = self.load_bulk_data_from_es_new(es_file_list, text_truncate_length, topn)

            data_list.append(list(zip(data_id, data_l, data_f, data_if)))

        # ensure repeatability
        rnd = random.Random(shuffle_seed)
        eps = 1e-6

        train_data = []
        val_data = []
        test_data = []

        for data, f in zip(data_list, es_file_list):
            rnd.shuffle(data)
            data_len = len(data)
            all_ratio = 1 + validate_ratio + test_ratio

            val_len = 0
            if validate_ratio > eps: # whether we need validation
                val_len = int(round(data_len * validate_ratio / all_ratio))


            test_len = 0
            if test_ratio > eps: # whether we need test
                test_len = int(round(data_len * test_ratio / all_ratio))

            train_len = data_len - val_len - test_len

            print('file {0}: train {1}, validate {2}, test {3}'.format(f, train_len, val_len, test_len))

            train_data.extend(data[:train_len])
            val_data.extend(data[train_len:train_len+val_len])
            test_data.extend(data[train_len+val_len:])

        if len(train_data) > 0 and train_out_file:
            print('Train data: {0} records'.format(len(train_data)))
            self.write_tfrecord(train_data, train_out_file)
        else:
            print('Train data empty or no filename specified! No data is written to disk!')

        if len(val_data) > 0 and validate_out_file:
            print('Validate data: {0} records'.format(len(val_data)))
            self.write_tfrecord(val_data, validate_out_file)
        else:
            print('Validate data empty or no filename specified! No data is written to disk!')

        if len(test_data) > 0 and test_out_file:
            print('Test data: {0} records'.format(len(test_data)))
            self.write_tfrecord(test_data, test_out_file)
        else:
            print('Test data empty or no filename specified! No data is written to disk!')

    # load bulk data from a list of es files and return numpy arrays
    def load_bulk_data_from_es_new(self, es_file_list, text_truncate_length=100, topn=10):
        all_feature = []
        all_label = []
        all_id = []
        all_ins_feature = []

        for f in es_file_list:
            data_f, data_if, data_l, data_id = self.load_data_from_es_new(f, text_truncate_length, topn)

            all_feature.extend(data_f)
            all_label.extend(data_l)
            all_id.extend(data_id)
            all_ins_feature.extend(data_if)

        return all_feature, all_ins_feature, all_label, all_id

    def load_data_from_es_new(self, es_file, text_truncate_length, topn):
        with perf_checker() as pc:
            with open(es_file, 'r', encoding='utf-8') as inputf:
                all_feature = []
                all_label = []
                all_id= []
                # ----- extra data for special purpose -----
                all_ins_feature = []

                print('reading file {0}...'.format(es_file))

                i = 0

                for line in inputf:
                    content = json.loads(line.rstrip('\n'))

                    i+=1

                    if i % 5000==0:
                        print('line {0}'.format(i))

                    # get feature
                    line_feature = []

                    es_facts_all = content['es_research_facts']
                    choices = ['Q+A', 'Q+B', 'Q+C', 'Q+D', 'Q+E']

                    for choice in choices:
                        fact_detail = es_facts_all[choice]

                        text = fact_detail['text']
                        line_feature.extend(self.get_text_index_feature(text, text_truncate_length))

                        facts_list = fact_detail['facts'] # array
                        facts_len = len(facts_list)

                        for seq_id in range(topn):
                            if seq_id < facts_len:
                                line_feature.extend(self.get_text_index_feature(facts_list[seq_id]['content'], text_truncate_length))
                            else:
                                line_feature.extend(self.get_text_index_feature('', text_truncate_length)) # empty default feature

                    if 'answer' in content:
                        if len(content['answer'])>1:
                            print("Invalid answer!", content['answer'])
                            continue

                        answer = ord(content['answer'].strip()) - ord('A')
                        if answer < 0 or answer > 4:
                            print("Invalid answer!", content['answer'])
                            continue
                            #raise Exception("Invalid answer!", content['answer'])

                        line_label = [int(i==answer) for i in range(5)]
                    else:
                        # answer not found, we give all zero invalid label
                        line_label = [0] * 5

                    # get id, for easier debugging
                    if 'id' in content:
                        line_id = content['id']
                    else:
                        line_id = '{0}_{1}'.format(path_leaf(es_file), i)

                    # line_id = '{0}_{1}'.format(path_leaf(es_file), i)

                    # get instance features
                    ins_f = self.get_instance_feature(content)

                    # add current data
                    all_feature.append(line_feature)
                    all_label.append(line_label)
                    all_id.append(line_id)

                    all_ins_feature.append(ins_f)

            print('used {0:.7f} sec'.format(pc.time_used()))

            # return data
            return all_feature, all_ins_feature, all_label, all_id

    def get_text_index_feature(self, text, text_truncate_length):
        if not text: # text is empty
            return [self.default_unknown_index] * text_truncate_length

        feature = []

        text_split = []
        if self.need_cut:
            # actually when we have 0x01 as q, a separator may perform better
            if self.norm_flag:
                text_split = [s for s in jieba.cut(text.replace('\u0001', ' ')) if not s.isspace()]
            else:
                text_split = [s for s in jieba.cut(text) if not s.isspace()]
        else:
            text_split = text.split(' ')

        len_text = len(text_split)

        for i in range(text_truncate_length):
            if i < len_text:
                feature.append(self.get_term_index(text_split[i]))
            else:
                feature.append(self.default_unknown_index)

        # sum_f = np.sum(feature)
        #
        # if sum_f != 0:
        #     print(sum_f)

        return feature

    def get_instance_feature_len(self):
        return 2

    def get_instance_feature(self, json_obj):
        ins_feature = []

        # feature value
        # get knowledge score
        if 'question_category' in json_obj:
            knw_score = float(json_obj['question_category']['knw'])
        else:
            knw_score = -1.0

        ins_feature.append(knw_score)

        if 'logic' in json_obj:
            logic = json_obj['logic']
            if logic[0] == 0:
                ins_feature.append(1.0)
            else:
                ins_feature.append(-1.0)
        else:
            ins_feature.append(1.0) # default we think question is positive

        return ins_feature

    def parse_record(self, example_proto):
        features = {'id': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=''),
                    'label': tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64, default_value=0, allow_missing=True),
                    'feature': tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64, default_value=0, allow_missing=True),
                    'ins_f': tf.FixedLenSequenceFeature(shape=(), dtype=tf.float32, default_value=0, allow_missing=True),
                    }
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features['id'], parsed_features['label'], parsed_features['feature']

    def validate_tfrecord_file(self, filename):

        count = 0
        for record in tf.python_io.tf_record_iterator(filename):
            count += 1

        print('There are {0} records in {1}'.format(count, filename))

        tf.reset_default_graph()

        filenames = tf.placeholder(tf.string, shape=[None])

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(lambda x: self.parse_record(x)).batch(256)

        data_iter = dataset.make_initializable_iterator()

        batch_id, batch_l, batch_f = data_iter.get_next()

        # batch_l = data_iter.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            sess.run(data_iter.initializer, feed_dict={filenames: [filename]})

            print('Readf actual data in {0}, bacthsize: {1}'.format(filename, 256))

            print('checking batch ', end = '')

            for i in range(int(math.ceil(count/256))):
                id, l, f = sess.run([batch_id, batch_l, batch_f])

                print('{0}...'.format(i), end = '')

                #print(l, f, [x.decode('utf-8') for x in id])

            print()

# --------------------------------------------------

def boolean_flag(parser, name, default=False, help=None):
    dest = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    group.add_argument("--no-" + name, action="store_false", dest=dest)

class MedExamModelParamParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.init_params()

    def init_params(self):
        self.parser.add_argument('--embedding-file', help='Embedding file', type=str, default='')
        self.parser.add_argument('--wordcut-file', help='Word cut file for Jieba', type=str, default='')
        self.parser.add_argument('--global-seed', help='Global seed used by model', type=int, default=0)
        self.parser.add_argument('--truncate-length', help='Text truncate length', type=int, default=100)
        self.parser.add_argument('--topn', help='Number of fatcs per ES search', type=int, default=10)
        boolean_flag(self.parser, 'cut-norm', help='Whether to normalize data before cut', default=False)
        boolean_flag(self.parser, 'debug-info', help='Whether to show debug info', default=False) # use '--debug-info' to set debug, '--no-debug-info' to turn off debug

        self.parser.add_argument('--lr', help='Model learning rate', type=float, default=0.001)
        self.parser.add_argument('--hidden-size', help='LSTM hidden layer size', type=int, default=128)
        self.parser.add_argument('--layer-num', help='Total number of LSTM layers', type=int, default=1)
        self.parser.add_argument('--keep-prob', help='1 - LSTM dropout rate', type=float, default=0.8)
        self.parser.add_argument('--rnd-sigma', help='Std-dev of gaussian random numbers of decision gating layer', type=float, default=0.1)
        self.parser.add_argument('--gate-loss-scale', help='The scale of gate loss', type=float, default=1.0)
        self.parser.add_argument('--gate-loss-th', help='The threshold when we need to add penalty for large average gate values', type=float, default=0.7)
        boolean_flag(self.parser, 'use-delta-embedding', help='Whether shoud we use delta embedding in training', default=False)
        self.parser.add_argument('--delta-embed-loss-scale', help='The scale of delta embedding loss', type=float, default=1.0)
        self.parser.add_argument('--soft-eps', help='Soft epsilon to avoid gradient explode for L2 norm', type=float, default=1e-6)
        boolean_flag(self.parser, 'use-instance-feature', help='Whether shoud we use instance feature in model', default=False)
        boolean_flag(self.parser, 'scale-dot-products', help='Whether shoud we make dot products smaller', default=False)
        boolean_flag(self.parser, 'use-orthogonal-init', help='Whether shoud we use orthogonal initialization for LSTM', default=False)
        boolean_flag(self.parser, 'use-mlp-vscale-init', help='Whether shoud we use variance scaling initialization for MLP', default=False)
        boolean_flag(self.parser, 'use-grad-clip', help='Whether shoud we use gradient clip', default=True)
        self.parser.add_argument('--grad-clip-norm', help='Global gradient clip norm', type=float, default=5.0)

        self.parser.add_argument('--epoch-limit', help='Max limit of epochs', type=int, default=100)
        self.parser.add_argument('--train-batch-per-epoch', help='Number of training batch per epoch', type=int, default=100)
        self.parser.add_argument('--train-batch-size', help='Number of training instances per batch', type=int, default=20)
        #self.parser.add_argument('--validate-batch-per-epoch', help='Number of validating batch per epoch', type=int, default=10) # now we go through all validation set
        self.parser.add_argument('--validate-batch-size', help='Number of validating instances per batch', type=int, default=50)
        self.parser.add_argument('--test-batch-size', help='Number of testing instances per batch', type=int, default=50)

        self.parser.add_argument('--save-model-dir', help='The path to save model', type=str, default='./models/')
        self.parser.add_argument('--final-model-suffix', help='The final suffix for saving model', type=str, default='model-latest')
        self.parser.add_argument('--model-prefix', help='The model suffix for loading model', type=str, default='./models/model-latest')
        self.parser.add_argument('--model-keep', help='Max number of models to keep when >0, or keep all models when <=0', type=int, default=3)
        self.parser.add_argument('--model-max-metrics', help='The metrics for model comparison', type=str, default='test_acc')
        self.parser.add_argument('--model-metrics-scale', help='The scale to multiply model metrics, could be -1 to apply min metrics', type=float, default=1.0)
        boolean_flag(self.parser, 'need-last-model', help='Whether to save last model after maxinum epochs elapsed, default yes', default=True)
        self.parser.add_argument('--summary-dir', help='The path to store checkpoints', type=str, default='./checkpoint/')

    def do_parse(self, sys_args):
        return self.parser.parse_args(sys.argv[1:])

# --------------------------------------------------

class MedExamModel(object):

    def __init__(self, args):
        if not isinstance(args, argparse.Namespace):
            raise ValueError("MedExam model only accept argparse.Namespace as init parameter")

        self.global_seed = args.global_seed

        # load embedding
        self.data = MedExamData(args)

        # after data loaded, we add some stat
        self.embedding_dim = self.data.embed_dim

        # set data dependent parameters
        self.truncate_length = args.truncate_length
        self.topn = args.topn

        self.input_types = (tf.int64, tf.int64, tf.float32)
        self.input_shapes = (tf.TensorShape([None, 5]),
                             tf.TensorShape([None, 5, self.topn + 1, self.truncate_length]),
                             tf.TensorShape([None, self.data.get_instance_feature_len()]))

        self.input_types_with_id = (tf.int64, tf.int64, tf.float32, tf.string)
        self.input_shapes_with_id = (tf.TensorShape([None, 5]),
                                     tf.TensorShape([None, 5, self.topn + 1, self.truncate_length]),
                                     tf.TensorShape([None, self.data.get_instance_feature_len()]),
                                     tf.TensorShape([None]))

        self.output = None

    # return a seed when an TF operator needs
    # use functions for flexibility
    def get_seed(self):
        return self.global_seed

    # function to parse single tfrecord
    # remember that 'id'field is only use for debugging
    def parse_medexam_tfrecord(self, example_proto):
        features = {'id': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=''),
                    'label': tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64, default_value=0, allow_missing=True),
                    'feature': tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64, default_value=0, allow_missing=True),
                    'ins_f': tf.FixedLenSequenceFeature(shape=(), dtype=tf.float32, default_value=0, allow_missing=True),
                    }
        parsed_features = tf.parse_single_example(example_proto, features)

        return parsed_features['label'], tf.reshape(parsed_features['feature'], [5, self.topn+1, self.truncate_length]), parsed_features['ins_f']

    def build_train_scenario_tfrecord_ds(self, train_tfrecord_file, validate_tfrecord_file = None, test_tfrecord_file = None, test_small = -1):
        if not train_tfrecord_file or not Path(train_tfrecord_file).is_file():
            print('Could not load train record file!')  # ERROR!
            return

        train_size = get_tfrecords_count(train_tfrecord_file)

        # batch_size is a variable now, which will be initialized before using
        self.batch_size = tf.placeholder(tf.int64, shape=[], name='batch_size')

        # use latest tf.data API and zip to input both features and labels
        train_data = tf.data.TFRecordDataset([train_tfrecord_file]).map(lambda x: self.parse_medexam_tfrecord(x))

        # train_data shape batch_size * ((), (topn+1, text_len))
        # we use repeat to produce wrap-around data
        # every time when we read all data, we will do re-shuffle, and repeat infinitely, with controlled batch_size
        train_data = train_data\
            .take(test_small)\
            .shuffle(train_size, seed = self.get_seed())\
            .repeat()\
            .batch(self.batch_size)

        self.train_iter = train_data.make_initializable_iterator()

        if validate_tfrecord_file and Path(validate_tfrecord_file).is_file():
            # validate
            val_all_size = get_tfrecords_count(validate_tfrecord_file)

            if test_small > 0:
                self.val_size = min(val_all_size, test_small)
            else:
                self.val_size = val_all_size

            # batch_size is a variable now, which will be initialized before using
            self.val_batch_size = tf.placeholder(tf.int64, shape=[], name='val_batch_size')

            val_data = tf.data.TFRecordDataset([validate_tfrecord_file]).map(lambda x: self.parse_medexam_tfrecord(x))

            # validate_data: val_size * ((), (topn+1, text_len))
            # use batch to handle big data
            # val_data = tf.data.Dataset.zip((val_data_l, val_data_f)).batch(self.batch_size)
            # val_data = val_data\
            #     .take(test_small)\
            #     .repeat()\
            #     .batch(self.batch_size)
            val_data = val_data \
                .take(test_small) \
                .batch(self.val_batch_size)

            self.val_iter = val_data.make_initializable_iterator()
        else:
            self.val_iter = None

        if test_tfrecord_file and Path(test_tfrecord_file).is_file():
            # test
            test_all_size = get_tfrecords_count(test_tfrecord_file)

            if test_small > 0:
                self.test_size = min(test_all_size, test_small)
            else:
                self.test_size = test_all_size

            # batch_size is a variable now, which will be initialized before using
            self.test_batch_size = tf.placeholder(tf.int64, shape=[], name='test_batch_size')

            # use latest tf.data API and zip to input both features and labels
            test_data = tf.data.TFRecordDataset([test_tfrecord_file]).map(lambda x: self.parse_medexam_tfrecord(x))

            # test_data: test_size * ((), (topn+1, text_len))
            # use batch to handle big data
            test_data = test_data.take(test_small) \
                .batch(self.test_batch_size)

            self.test_iter = test_data.make_initializable_iterator()
        else:
            self.test_iter = None

    def build_data_handle(self):
        # handle is used to save state, switch data source and resume from where we left
        self.handle = tf.placeholder(tf.string, shape=[], name='data_handle')
        # all three data source have same type and shape
        iterator = tf.data.Iterator.from_string_handle(self.handle, self.input_types, self.input_shapes)
        # batch_l: batch_size * 1, batch_f: batch_size * topn+1 * text_len, batch_if: batch_size * ins_feature_len
        batch_l, batch_f, batch_if = iterator.get_next()

        return batch_l, batch_f, batch_if

    def parse_medexam_tfrecord_with_id(self, example_proto):
        features = {'id': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=''),
                    'label': tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64, default_value=0, allow_missing=True),
                    'feature': tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64, default_value=0, allow_missing=True),
                    'ins_f': tf.FixedLenSequenceFeature(shape=(), dtype=tf.float32, default_value=0, allow_missing=True),
                    }
        parsed_features = tf.parse_single_example(example_proto, features)

        return parsed_features['label'], tf.reshape(parsed_features['feature'], [5, self.topn + 1, self.truncate_length]), tf.cast(parsed_features['ins_f'], tf.float32), parsed_features['id']

    def build_test_es_for_train_ds(self, test_es_data, test_small=-1):

        test_feature_data, test_ins_feature_data, test_label_data, *temp = test_es_data

        test_feature_data = np.reshape(test_feature_data, [-1, 5, self.topn+1, self.truncate_length])
        test_ins_feature_data = np.reshape(test_ins_feature_data, [-1, self.data.get_instance_feature_len()])
        test_label_data = np.array(test_label_data)

        if test_small > 0:
            self.test_size = min(test_feature_data.shape[0], test_small)
        else:
            self.test_size = test_feature_data.shape[0]

        # test feature, since it could very big, we also use batch to prevent memory break
        # if actual validate data size is small, then we could input is as a whole
        # we use zip to input both features and labels
        test_data_f = tf.data.Dataset.from_tensor_slices(test_feature_data)
        test_data_if = tf.data.Dataset.from_tensor_slices(tf.cast(test_ins_feature_data, tf.float32))
        test_data_l = tf.data.Dataset.from_tensor_slices(test_label_data)

        # batch_size is a variable now, which will be initialized before using
        self.test_batch_size = tf.placeholder(tf.int64, shape=[], name='test_batch_size')

        # test_data: test_size * ((), (topn+1, text_len))
        # use batch to handle big data
        test_data = tf.data.Dataset.zip((test_data_l, test_data_f, test_data_if)).take(test_small).batch(self.test_batch_size)

        self.test_iter = test_data.make_initializable_iterator()

    def build_test_es_ds(self, test_es_data, test_small=-1):

        test_feature_data, test_ins_feature_data, test_label_data, test_id_data, *temp = test_es_data

        test_feature_data = np.reshape(test_feature_data, [-1, 5, self.topn+1, self.truncate_length])
        test_ins_feature_data = np.reshape(test_ins_feature_data, [-1, self.data.get_instance_feature_len()])
        test_label_data = np.array(test_label_data)

        if test_small > 0:
            self.test_size = min(test_feature_data.shape[0], test_small)
        else:
            self.test_size = test_feature_data.shape[0]

        # test feature, since it could very big, we also use batch to prevent memory break
        # if actual validate data size is small, then we could input is as a whole
        # we use zip to input both features and labels
        test_data_f = tf.data.Dataset.from_tensor_slices(test_feature_data)
        test_data_if = tf.data.Dataset.from_tensor_slices(tf.cast(test_ins_feature_data, tf.float32))
        test_data_l = tf.data.Dataset.from_tensor_slices(test_label_data)
        test_data_id = tf.data.Dataset.from_tensor_slices(test_id_data)

        # batch_size is a variable now, which will be initialized before using
        self.test_batch_size = tf.placeholder(tf.int64, shape=[], name='test_batch_size')

        # test_data: test_size * ((), (topn+1, text_len))
        # use batch to handle big data
        test_data = tf.data.Dataset.zip((test_data_l, test_data_f, test_data_if, test_data_id)).take(test_small).batch(self.test_batch_size)

        self.test_iter = test_data.make_initializable_iterator()

    def build_test_tfrecord_ds(self, test_tfrecord_file, test_small=-1):
        if not test_tfrecord_file or not Path(test_tfrecord_file).is_file():
            print('Could not load test tf record file!')  # ERROR!
            return

        test_all_size = get_tfrecords_count(test_tfrecord_file)

        if test_small > 0:
            self.test_size = min(test_all_size, test_small)
        else:
            self.test_size = test_all_size

        # batch_size is a variable now, which will be initialized before using
        self.test_batch_size = tf.placeholder(tf.int64, shape=[], name='test_batch_size')

        # use latest tf.data API and zip to input both features and labels
        test_data = tf.data.TFRecordDataset([test_tfrecord_file]).map(lambda x: self.parse_medexam_tfrecord_with_id(x))

        # test_data: test_size * ((), (topn+1, text_len))
        # use batch to handle big data
        test_data = test_data.take(test_small)\
            .batch(self.test_batch_size)

        self.test_iter = test_data.make_initializable_iterator()

    def build_data_handle_with_id(self):
        # handle is used to save state, switch data source and resume from where we left
        self.handle = tf.placeholder(tf.string, shape=[], name='data_handle')
        # all three data source have same type and shape
        iterator = tf.data.Iterator.from_string_handle(self.handle, self.input_types_with_id, self.input_shapes_with_id )
        # batch_l: batch_size * 1, batch_f: batch_size * topn+1 * text_len
        batch_l, batch_f, batch_if, batch_id = iterator.get_next()

        return batch_l, batch_f, batch_if, batch_id

    def build_embedding_ds(self):
        self.embedding_ds = tf.placeholder(dtype=self.data.embed_vec.dtype,
                                           shape=self.data.embed_vec.shape,
                                           name='embedding_ds')

        self.embedding_input = tf.get_variable(name='embedding_input',
                                               initializer=self.embedding_ds,
                                               trainable=False,
                                               collections=[])
        if self.run_args.use_delta_embedding:
            # with tf.variable_scope('embed', reuse=True):
            self.delta_embedding = tf.get_variable(name='delta_embedding',
                                                   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=self.get_seed()),
                                                   shape=self.data.embed_vec.shape,
                                                   dtype=tf.float32,
                                                   trainable=True)

    def get_lstm_cell_layer(self, lstm_size, keep_prob, random_seed):
        # rnn_size is the output dimension
        lstm_cell = None
        if self.run_args.use_orthogonal_init is True:
            lstm_cell = rnn.LSTMCell(num_units=lstm_size,
                                     initializer=tf.orthogonal_initializer(gain=1.0, seed=random_seed),
                                     forget_bias=1.0,
                                     state_is_tuple=True)
        else:
            lstm_cell = rnn.LSTMCell(num_units=lstm_size,
                                     initializer=tf.truncated_normal_initializer(0, 0.1, seed=random_seed),
                                     forget_bias=1.0,
                                     state_is_tuple=True)

        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,
                                       input_keep_prob=1.0,
                                       output_keep_prob=keep_prob,
                                       seed=random_seed)
        return lstm_cell

    def get_bi_lstm(self, input, scope, layer_num, hidden_size, keep_prob):
        # use scope to avoid name confict
        with tf.variable_scope(scope):

            mlstm_cell_fw = rnn.MultiRNNCell(
                [self.get_lstm_cell_layer(hidden_size, keep_prob, self.get_seed()) for l_ind in range(layer_num)],
                state_is_tuple=True)
            mlstm_cell_bw = rnn.MultiRNNCell(
                [self.get_lstm_cell_layer(hidden_size, keep_prob, self.get_seed()) for l_ind in range(layer_num)],
                state_is_tuple=True)

            # using bidirectional_dynamic_rnn()
            # When time_major==False
            # outputs.shape = [batch_size, timestep_size, hidden_size]
            # state.shape = [layer_num, 2, batch_size, hidden_size]
            # if output is last output [batch_size, hidden_size], we could use h_state = outputs[:, -1, :], or use h_state = state[-1][1]

            (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(mlstm_cell_fw,
                                                                            mlstm_cell_bw,
                                                                            input,
                                                                            dtype = tf.float32,
                                                                            time_major = False,
                                                                            # sequence_length=timestep_size,
                                                                            )

            return output_fw, output_bw, state

    def build_model_targets(self, batch_l, batch_f, batch_if):
        if self.truncate_length is None or self.topn is None:
            print('Error! Please load data first before you build model!')
            return

        # important flag!
        # during training phase, to imporve training performance, we may use dropout, batch normalization, noisy gating, etc.
        # however, during testing phase, randomness need to be removed for reproducible results
        # and some layers are required to behave differently
        # we make it as model-wise flag
        self.is_training = tf.placeholder_with_default(True, shape=())

        actual_keep_prob = tf.cond(tf.equal(self.is_training, tf.constant(True)), lambda: tf.constant(self.run_args.keep_prob), lambda: tf.constant(1.0))

        # explicitly reshape, batch*5*(topn+1) * truncate_length
        x = tf.reshape(batch_f, [-1, self.truncate_length])

        # batch_embedding: batch_size*5*(topn+1) * truncate_length * embedding_dim
        xe = tf.nn.embedding_lookup(tf.cast(self.embedding_input, tf.float32), x)

        if self.run_args.use_delta_embedding:
            xed = tf.nn.embedding_lookup(self.delta_embedding, x)
            xe = xe + xed

        # do bi-directional LSTM with drop-out
        output_fw, output_bw, *temp = self.get_bi_lstm(xe, 'context_lstm', self.run_args.layer_num, self.run_args.hidden_size, actual_keep_prob)

        # combine forward and backward state output
        # outshape is batch*5*(topn+1) * truncate_length * hidden_size
        output = tf.add(output_fw, output_bw)

        # build attention layer
        rq_all, rd_prime, qpool_all, dpool_all, qa, facts = self.build_attention_layer(contextual_embedding_batch = output,
                                                                                       input_dim = self.run_args.hidden_size)

        # build reasoning layer and get q/fact support and average gate value, which is used in loss function
        support, g_avg = self.build_reason_layer(qa,
                                                 rq_all,
                                                 qpool_all,
                                                 facts,
                                                 rd_prime,
                                                 dpool_all,
                                                 input_dim = self.run_args.hidden_size,
                                                 hidden_size = self.run_args.hidden_size,
                                                 layer_num = self.run_args.layer_num,
                                                 keep_prob = actual_keep_prob)

        # build decision layer
        logits = self.build_decision_layer(support, batch_if)

        self.logits = logits

        self.label = batch_l

        self.prob = tf.nn.softmax(logits, axis=-1, name = 'probability')

        self.actual_batch_size = tf.identity(tf.shape(logits)[0], name = 'actual_batch_size')

        self.prediction  = tf.argmax(logits, 1, name = 'prediction')

        self.ce_loss = tf.losses.softmax_cross_entropy(onehot_labels=batch_l, logits = logits, reduction = tf.losses.Reduction.MEAN)

        self.gate_reg_loss = self.run_args.gate_loss_scale * tf.nn.relu(g_avg - self.run_args.gate_loss_th)

        self.delta_e_loss = tf.cast(0.0, tf.float32)
        if self.run_args.use_delta_embedding:
            soft_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.delta_embedding), axis=-1) + self.run_args.soft_eps)

            self.delta_e_loss = self.run_args.delta_embed_loss_scale * tf.reduce_mean(soft_l2_norm)

        #self.loss = tf.add(self.ce_loss, self.gate_reg_loss, name = 'loss')
        self.loss = tf.add_n([self.ce_loss, self.gate_reg_loss, self.delta_e_loss], name='loss')

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(batch_l, 1), self.prediction), tf.float32), name = 'accuracy')

        self.debug = tf.equal(tf.argmax(batch_l, 1), self.prediction)

        # train
        self.optimizer = tf.train.AdamOptimizer(self.run_args.lr, epsilon=1e-6)
        gs, vs = zip(*self.optimizer.compute_gradients(self.loss))
        self.gn = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in gs]))
        if args.use_grad_clip is True:
            # This is the correct way to perform gradient clipping
            gs, _ = tf.clip_by_global_norm(gs, clip_norm=args.grad_clip_norm, use_norm=self.gn)

        self.gnclip = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in gs]))

        self.train_op = self.optimizer.apply_gradients(zip(gs, vs))

        # self.train_op = tf.train.AdamOptimizer(self.run_args.lr, epsilon = 1e-6).minimize(self.loss)
        ## self.train_op = tf.train.GradientDescentOptimizer(lr, name = 'train_op').minimize(self.loss)

    def build_attention_layer(self, contextual_embedding_batch, input_dim):

        topn = self.topn
        textlen = self.truncate_length

        # here batch will be batch * 5
        batch = tf.reshape(contextual_embedding_batch, [-1, topn + 1, textlen, input_dim])

        qlen = textlen
        dlen = textlen

        qa = tf.reshape(batch[:, 0, :, :], [-1, qlen, input_dim])

        rqlist = []
        rdlist = []
        qpool_list = []
        dpool_list = []

        facts = tf.reshape(batch[:, 1:, :, :], [-1, topn, dlen, input_dim])

        for i in range(topn):
            fact = tf.reshape(batch[:, i + 1, :, :], [-1, dlen, input_dim])

            # outshape is batch * qlen * dlen
            sim_mat = tf.map_fn(lambda x: tf.tensordot(x[0], x[1], [[-1], [-1]]), (qa, fact), dtype=tf.float32)

            if self.run_args.scale_dot_products is True:
                sim_mat = sim_mat / np.sqrt(input_dim)

            # outshape is batch * qlen * [dlen], where softmax is performed on last index
            alpha_q = tf.nn.softmax(sim_mat, axis=-1)

            # outshape is batch * qlen * dim
            rq = tf.map_fn(lambda x: tf.matmul(x[0], x[1]), (alpha_q, fact), dtype=tf.float32)

            rqlist.append(rq)

            # outshape is batch * qlen
            q_maxpool = tf.reduce_max(sim_mat, axis=-1)
            q_meanpool = tf.reduce_mean(sim_mat, axis=-1)

            # outshape is batch * qlen * 2
            qpool_list.append(tf.concat([tf.expand_dims(q_maxpool, -1),
                                        tf.expand_dims(q_meanpool, -1)], axis=-1))

            # outshape is batch * [qlen] * dlen
            alpha_d = tf.nn.softmax(sim_mat, axis=-2)

            # outshape is batch * dlen * dim
            rd = tf.map_fn(lambda x: tf.tensordot(x[0], x[1], [[0], [0]]), (alpha_d, qa), dtype=tf.float32)

            rdlist.append(rd)

            # outshape is batch * dlen
            d_maxpool = tf.reduce_max(sim_mat, axis=-2)
            d_meanpool = tf.reduce_mean(sim_mat, axis=-2)

            # outshape is batch * dlen * 2
            dpool_list.append(tf.concat([tf.expand_dims(d_maxpool, -1),
                                         tf.expand_dims(d_meanpool, -1)], axis=-1))

        # outshape is batch * topn * qlen * dim
        rq_all = tf.stack(rqlist, axis=1)

        # outshape is batch * topn * qlen * 2
        qpool_all = tf.stack(qpool_list, axis=1)

        # outshape is batch * topn * dlen * dim
        rd_all = tf.stack(rdlist, axis=1)

        #facts = tf.reshape(batch[:, 1:, :, :], [-1, topn, dlen, dim])

        # outshape is batch * topn * dlen * (2*dim)
        cross_doc = tf.concat([rd_all, facts], axis=-1)

        # outshape is batch * topn * dlen * topn * dlen
        cross_doc_sim = tf.map_fn(lambda x: tf.tensordot(x[0], x[1], [[-1], [-1]]), (cross_doc, cross_doc),
                                  dtype=tf.float32)

        if self.run_args.scale_dot_products is True:
            cross_doc_sim = cross_doc_sim / np.sqrt(2 * input_dim)

        # outshape is batch * topn * dlen * [topn * dlen]
        # alpha_cross_doc = tf.nn.softmax(cross_doc_sim, axis = (3, 4))
        # use trick to avoid NaN in softmax
        normalized_cross_doc_sim = cross_doc_sim - tf.reduce_max(cross_doc_sim, [-2, -1], keepdims=True)
        exp_val = tf.exp(normalized_cross_doc_sim)
        alpha_cross_doc = exp_val / tf.reduce_sum(exp_val, [-2, -1], keepdims=True)

        # outshape is batch * topn * dlen * (2*dim)
        rd_prime = tf.map_fn(lambda x: tf.tensordot(x[0], x[1], [[-2, -1], [0, 1]]), (alpha_cross_doc, cross_doc),
                             dtype=tf.float32)

        # outshape is batch * topn * dlen * 2
        dpool_all = tf.stack(dpool_list, axis=1)

        # batch * topn * qlen * dim
        # batch * topn * dlen * (2*dim)
        # batch * topn * qlen * 2
        # batch * topn * dlen * 2
        # batch * qlen * dim
        # batch * topn * dlen * dim
        return rq_all, rd_prime, qpool_all, dpool_all, qa, facts

    def get_dense_layer(self, input, output_size):

        if self.run_args.use_mlp_vscale_init is True:
            return tf.layers.dense(input, output_size, kernel_initializer=tf.variance_scaling_initializer(seed=self.get_seed()))
        else:
            return tf.layers.dense(input, output_size, kernel_initializer=layers.xavier_initializer(seed=self.get_seed()))

    def build_reason_layer(self, q_context_batch, rq, qpool, d_context_batch, rd_prime, dpool_all, input_dim, hidden_size, layer_num, keep_prob):
        qlen = self.truncate_length
        dlen = self.truncate_length

        # q-part
        #q_context_batch size=(batch,1,100,128)
        q_x = tf.reshape(q_context_batch, [-1, input_dim]) # 100*batch,128
        q_x_gate = self.get_dense_layer(input = q_x, output_size = 1)
        q_x_gate = tf.reshape(tf.nn.sigmoid(q_x_gate), [-1, 1, qlen, 1]) # batch,1,100,1

        q_g_avg = tf.reduce_mean(q_x_gate)

        # outshape is batch * topn * qlen * (dim+2)
        q_gated = tf.multiply(tf.concat([rq, qpool], axis = -1), q_x_gate)
        q_gated = tf.reshape(q_gated, [-1, qlen, input_dim + 2])

        # gated value go though bi-directional LSTM
        q_out_fw, q_out_bw, *temp = self.get_bi_lstm(q_gated, 'reason_lstm_q', layer_num, hidden_size, keep_prob)

        # combine forward and backword state output
        # outshape is (batch*topn) * qlen * hidden_size
        q_out = tf.add(q_out_fw, q_out_bw)

        # max pooling
        # outshape is (batch*topn) * 、
        q_out = tf.reduce_max(q_out, axis = -2)

        # restore shape
        q_out = tf.reshape(q_out, [-1, self.topn, hidden_size])

        # ---------- doc part ----------

        # d-part
        d_x = tf.reshape(d_context_batch, [-1, input_dim])
        d_x_gate = self.get_dense_layer(input = d_x, output_size = 1)
        d_x_gate = tf.reshape(tf.nn.sigmoid(d_x_gate), [-1, self.topn, dlen, 1])

        d_g_avg = tf.reduce_mean(d_x_gate)

        # outshape is batch * topn * dlen * (2*dim+2)
        d_gated = tf.multiply(tf.concat([rd_prime, dpool_all], axis=-1), d_x_gate)
        d_gated = tf.reshape(d_gated, [-1, dlen, 2 * input_dim + 2])

        # gated value go though bi-directional LSTM
        d_out_fw, d_out_bw, *temp = self.get_bi_lstm(d_gated, 'reason_lstm_d', layer_num, hidden_size, keep_prob)

        # combine forward and backword state output
        # outshape is (batch*topn) * dlen * hidden_size
        d_out = tf.add(d_out_fw, d_out_bw)

        # max pooling
        # outshape is (batch*topn) * hidden_size
        d_out = tf.reduce_max(d_out, axis=-2)

        # restore shape
        d_out = tf.reshape(d_out, [-1, self.topn, hidden_size])

        # average of gate value, for loss function
        g_avg = (q_g_avg + d_g_avg) / 2

        # concat, shape: batch * topn * (2*hidden_size)
        support = tf.concat([q_out, d_out], axis = -1)  #16*10*256

        return support, g_avg

    def build_decision_layer(self, support, batch_if):
        # outshape: batch * topn * 1
        support_gate = self.get_dense_layer(input = support, output_size = 1)  #####################################################support 16*10*256
        support_gate = tf.sigmoid(support_gate)

        # outshape: batch * topn * (2*hidden_size)
        gated = tf.multiply(support_gate, support)

        noise = tf.cond(tf.equal(self.is_training, tf.constant(True)),
                        lambda: tf.random_normal(shape=tf.shape(gated), stddev=self.run_args.rnd_sigma),
                        lambda: tf.zeros(shape=tf.shape(gated))) # noise is disabled for testing phase

        # noisy gate
        gated = gated + noise

        # outshape: batch * (2*hidden_size)
        gated_maxpool = tf.reduce_max(gated, axis = -2)
        gated_meanpool = tf.reduce_mean(gated, axis = -2)

        # final outout
        x = tf.concat([gated_maxpool, gated_meanpool], axis = -1)
        x = self.get_dense_layer(input = x, output_size = 1)

        # restore 5 choices
        logits = tf.reshape(x, [-1, 5])

        if self.run_args.use_instance_feature:
            sign = tf.reshape(batch_if[:, 1], [-1, 1]) # logic[0]
            logits = logits * sign

        return logits

    def check_model(self):
        # this function is used only for debuggin purpose
        #if self.prediction is None or self.loss is None or self.accuracy is None:
        #    print('Error! Please build the model first!')

        with tf.Session() as sess:

            train_handle = sess.run(self.train_iter.string_handle())

            gi = tf.global_variables_initializer()

            sess.run(gi)

            sess.run(self.embedding_input.initializer, feed_dict={self.embedding_ds: self.data.embed_vec})

            cur_batch_size = 3

            file_writer = tf.summary.FileWriter('./checkpoint/', sess.graph)

            # initialize train data set once, and identify bacth_size, since train is repeat, we do not need to initialize again
            sess.run(self.train_iter.initializer, feed_dict={self.batch_size: cur_batch_size})

            check, *result = sess.run([self.test, self.prediction, self.loss, self.accuracy], feed_dict={self.handle: train_handle})
            print(result)

            #print(np.max(np.abs(check)), np.min(np.abs(check)))
            print(np.max(check), np.min(check))
            #test = np.exp(check)/np.sum(np.sum(np.exp(check), axis = -1, keepdims=True), axis = -2, keepdims=True)
            #print(np.max(test), np.min(test))
            #
            # loop_num = int(math.ceil(self.train_size/cur_batch_size))
            #
            # for _ in range(loop_num):
            #
            #     result = sess.run([self.actual_batch_size], feed_dict={self.handle: train_handle})
            #     # result = sess.run(self.debug, feed_dict={self.handle: train_handle})
            #
            #     #print(result.dtype)
            #     #print(result.shape)
            #     print(result)

            # for epoch in range(5):
            #
            #     for train_loop in range(2):
            #         lval, *val = sess.run([loss, next_l_batch, next_f_batch], feed_dict={handle: train_handle})
            #         print(lval, val)
            #
            #     # every time we need to iterate over all validation sets
            #     sess.run(val_iter.initializer)
            #     # sess.run(val_iter.initializer, feed_dict = {batch_size: fake_val_size})
            #     for _ in range(1):
            #         lval, *val = sess.run([op, next_l_batch, next_f_batch], feed_dict={handle: val_handle})
            #         print(lval, val)

    def build_train_model_tfrecord(self, train_tfrecord_file, train_phase_seed, args, validate_tfrecord_file=None, test_tfrecord_file=None, test_small=-1):
        if self.data is None:
            print('Error! Please load data first before you build model!')
            return

        tf.reset_default_graph()

        set_global_random_seed(train_phase_seed)

        self.run_args = args # pass arguments to class instance

        self.build_train_scenario_tfrecord_ds(train_tfrecord_file, validate_tfrecord_file, test_tfrecord_file, test_small)

        self.build_embedding_ds()

        batch_label, batch_feature, batch_ins_feature = self.build_data_handle()

        self.build_model_targets(batch_label, batch_feature, batch_ins_feature)

    def build_train_model_test_es(self, train_tfrecord_file, train_phase_seed, args, test_es_data=None, test_small=-1, validate_tfrecord_file=None,):
        if self.data is None:
            print('Error! Please load data first before you build model!')
            return

        tf.reset_default_graph()

        set_global_random_seed(train_phase_seed)

        self.run_args = args  # pass arguments to class instance

        self.build_train_scenario_tfrecord_ds(train_tfrecord_file, validate_tfrecord_file, None, test_small)

        self.build_test_es_for_train_ds(test_es_data, test_small)

        self.build_embedding_ds()

        batch_label, batch_feature, batch_ins_feature = self.build_data_handle()

        self.build_model_targets(batch_label, batch_feature, batch_ins_feature)

    def build_test_model_es(self, test_es_data, test_phase_seed, args, test_small=-1):
        if self.data is None:
            print('Error! Please load data first before you build model!')
            return

        tf.reset_default_graph()

        set_global_random_seed(test_phase_seed)

        self.run_args = args  # pass arguments to class instance

        self.build_test_es_ds(test_es_data, test_small)

        self.build_embedding_ds()

        batch_label, batch_feature, batch_ins_feature, batch_id = self.build_data_handle_with_id()

        self.build_model_targets(batch_label, batch_feature, batch_ins_feature)

        self.ids = batch_id

    def build_test_model_tfrecord(self, test_tfrecord_file, test_phase_seed, args, test_small=-1):
        if self.data is None:
            print('Error! Please load data first before you build model!')
            return

        tf.reset_default_graph()

        set_global_random_seed(test_phase_seed)

        self.run_args = args  # pass arguments to class instance

        self.build_test_tfrecord_ds(test_tfrecord_file, test_small)

        self.build_embedding_ds()

        batch_label, batch_feature, batch_ins_feature, batch_id = self.build_data_handle_with_id()

        self.build_model_targets(batch_label, batch_feature, batch_ins_feature)

        self.ids = batch_id

    def load_and_train_model_tfrecord(self, train_tfrecord_file, train_phase_seed, args, train_stat_file, validate_tfrecord_file=None, test_tfrecord_file=None, test_small=-1):

        print('--------------------Load and Train--------------------')
        print('Train tfrecord file: {0}'.format(train_tfrecord_file))
        print('Validate tfrecord file: {0}'.format(validate_tfrecord_file))
        print('Test tfrecord file: {0}'.format(test_tfrecord_file))
        print('test_small:', test_small)
        print(args)
        print()

        with perf_checker() as pc:
            self.build_train_model_tfrecord(train_tfrecord_file, train_phase_seed, args, validate_tfrecord_file, test_tfrecord_file, test_small)

            print('build training model used {0:.7f} sec'.format(pc.time_used()))

        with open(train_stat_file, 'w', newline='', encoding='utf-8') as stat_writer:
            stat = self.train_model(args, stat_writer)

        return stat

    def load_and_train_model_test_es(self, train_tfrecord_file, train_phase_seed, args, train_stat_file, test_es_file=None, test_small=-1, validate_tfrecord_file=None):

        print('--------------------Load and Train--------------------')
        print('Train tfrecord file: {0}'.format(train_tfrecord_file))
        print('Validation tfrecord file: {0}'.format(validate_tfrecord_file))
        print('Test es file: {0}'.format(test_es_file))
        print('Train random seed: {0}'.format(train_phase_seed))
        print('test_small:', test_small)
        print(args)
        print()

        # result should not change, because testing does not depend on random numbers
        with perf_checker() as pc:
            test_data = self.data.load_data_from_es_new(test_es_file, self.truncate_length, self.topn)

            self.build_train_model_test_es(train_tfrecord_file, train_phase_seed, args, test_data, test_small, validate_tfrecord_file)

            print('build training model used {0:.7f} sec'.format(pc.time_used()))

        with open(train_stat_file, 'w', newline='', encoding='utf-8') as stat_writer:
            stat = self.train_model(args, stat_writer)

        return stat

    # test phase is not compatible with training, so we only do validation
    def train_model(self, args, writer):

        all_stat = stat_helper()

        print('----- train -----')

        if args.debug_info is True:
            tf_config = tf.ConfigProto(log_device_placement=True)
            tf_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        else:
            tf_config = tf.ConfigProto()
            tf_options = tf.RunOptions()
        tf_metadata = tf.RunMetadata()

        best_test_accuracy = -1.0
        best_val_accuracy = -1.0

        # --------------------
        # builder = tf.profiler.ProfileOptionBuilder
        # opts = builder(builder.time_and_memory()).order_by('bytes').build()
        # --------------------

        with tf.contrib.tfprof.ProfileContext('./prof/',
                                      trace_steps=[],
                                      dump_steps=[]) as pctx,\
                tf.Session(config=tf_config) as sess:

            file_writer = None
            if args.debug_info is True:
                file_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)
            #meta_helper = run_stat_helper(file_writer, args.debug_info)
            #meta_helper.add(tf_metadata, 'self.embedding_input.initializer')

            # -------------------- prepare models --------------------
            with perf_checker() as pc:

                sess.run(tf.global_variables_initializer(), options=tf_options, run_metadata=tf_metadata)

                train_handle = sess.run(self.train_iter.string_handle(), options=tf_options, run_metadata=tf_metadata)

                val_handle = None
                if self.val_iter is not None:
                    val_handle = sess.run(self.val_iter.string_handle(), options=tf_options, run_metadata=tf_metadata)

                test_handle = None
                if self.test_iter is not None:
                    test_handle = sess.run(self.test_iter.string_handle(), options=tf_options, run_metadata=tf_metadata)

                # embedding
                sess.run(self.embedding_input.initializer, feed_dict={self.embedding_ds: self.data.embed_vec}, options=tf_options, run_metadata=tf_metadata)

                # initialize train data set once, and set bacth_size,
                # training data is repeat, we do not need to initialize again
                sess.run(self.train_iter.initializer, feed_dict={self.batch_size: args.train_batch_size}, options=tf_options, run_metadata=tf_metadata)

                # if self.val_iter is not None:
                #
                #     sess.run(self.val_iter.initializer, feed_dict={self.val_batch_size: args.validate_batch_size}, options=tf_options, run_metadata=tf_metadata)

                print('prepare model used {0:.7f} sec'.format(pc.time_used()))

            if args.save_model_dir is not None:
                dir_helper = saver_model_helper(args.save_model_dir, args.model_keep)

            # -------------------- Preparation done, start to do training --------------------
            # get all model variables for model load and save
            model_variables = {val.name : val for val in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}

            # print('We have following model variables:')
            # for val_name,val in model_variables.items():
            #     print('{0} -> {1}'.format(val_name, val.shape))

            with perf_checker() as pc_all_epoch:

                stat_keys = []

                for epoch in range(args.epoch_limit):
                    print('\n[start epoch {0}]'.format(epoch))
                    # clear per epoch stat
                    epoch_stat = {}

                    # -------------------- epoch: train phase --------------------
                    print('\ntrain ', end='')
                    sys.stdout.flush()
                    train_epoch_stat = stat_helper()

                    #debug_node = [print(n.name) for n in sess.graph.as_graph_def().node]TensorArrayScatterV3

                    with perf_checker() as pc:
                        for batch in range(args.train_batch_per_epoch):
                            print('{0} '.format(batch), end='')
                            sys.stdout.flush()

                            # pctx.trace_next_step()
                            # pctx.dump_next_step()

                            _, gn, gnclip, ce_loss, g_loss, de_loss, accuracy, weight = sess.run([self.train_op, self.gn, self.gnclip, self.ce_loss, self.gate_reg_loss, self.delta_e_loss, self.accuracy, self.actual_batch_size],
                                                                                     feed_dict={self.handle: train_handle, self.is_training: True},
                                                                                     options=tf_options,
                                                                                     run_metadata=tf_metadata)

                            # debug_node = [print(n.name) for n in sess.graph.as_graph_def().node if 'TensorArrayScatterV3' in n.name]
                            # pctx.profiler.profile_operations(options=opts)
                            #meta_helper.add(tf_metadata, 'train_epoch_{0}_batch_{1}'.format(epoch, batch))

                            train_epoch_stat.add_data({'ce_loss': ce_loss, 'g_loss': g_loss, 'de_loss': de_loss, 'accuracy': accuracy, 'weight': weight, 'gnorm': gn, 'gnormclip': gnclip})

                        pc.print_time_used()

                    train_epoch_stat_avg = train_epoch_stat.get_weighted_avg(weight_key='weight')
                    epoch_stat.update({'train_ce_loss': train_epoch_stat_avg['ce_loss'],
                                       'train_g_loss': train_epoch_stat_avg['g_loss'],
                                       'train_de_loss': train_epoch_stat_avg['de_loss'],
                                       'train_acc': train_epoch_stat_avg['accuracy'],
                                       'avg_gnorm': train_epoch_stat_avg['gnorm'],
                                       'min_gnorm': train_epoch_stat.get_min('gnorm'),
                                       'max_gnorm': train_epoch_stat.get_max('gnorm'),
                                       'avg_gnormclip': train_epoch_stat_avg['gnormclip'],
                                       'min_gnormclip': train_epoch_stat.get_min('gnormclip'),
                                       'max_gnormclip': train_epoch_stat.get_max('gnormclip')
                                       })

                    print('----->>>>> train ce loss: {0:.7f}, g loss: {1:.7f}, de loss: {2:.7f}, accuracy: {3:.7f}'.format(
                        epoch_stat['train_ce_loss'], epoch_stat['train_g_loss'], epoch_stat['train_de_loss'], epoch_stat['train_acc']))
                    print('----->>>>> train grad_norm: [{0:.7f}, {1:.7f}], avg {2:.7f}, clipped grad_norm: [{3:.7f}, {4:.7f}], avg {5:.7f}'.format(
                        epoch_stat['min_gnorm'], epoch_stat['max_gnorm'], epoch_stat['avg_gnorm'],
                        epoch_stat['min_gnormclip'], epoch_stat['max_gnormclip'], epoch_stat['avg_gnormclip']))

                    # -------------------- epoch: validate phase --------------------
                    if self.val_iter is not None:

                        # init val batch
                        sess.run(self.val_iter.initializer, feed_dict={self.val_batch_size: args.validate_batch_size})

                        val_loop_num = int(math.ceil(self.val_size/args.validate_batch_size))

                        print('\nvalidate ', end='')
                        sys.stdout.flush()
                        val_epoch_stat = stat_helper()

                        with perf_checker() as pc:
                            #for val_loop_index in range(args.validate_batch_per_epoch):
                            for val_loop_index in range(val_loop_num):
                                print('{0} '.format(val_loop_index), end='')
                                sys.stdout.flush()

                                ce_loss, g_loss, de_loss, accuracy, weight = sess.run([self.ce_loss, self.gate_reg_loss, self.delta_e_loss, self.accuracy, self.actual_batch_size],
                                                                                      feed_dict={self.handle: val_handle, self.is_training: False}, options=tf_options, run_metadata=tf_metadata)

                                #meta_helper.add(tf_metadata, 'validate_epoch_{0}_batch_{1}'.format(epoch, val_loop_index))
                                val_epoch_stat.add_data({'ce_loss': ce_loss, 'g_loss': g_loss, 'de_loss': de_loss, 'accuracy': accuracy, 'weight': weight})

                            pc.print_time_used()

                        val_epoch_stat_avg = val_epoch_stat.get_weighted_avg(weight_key='weight')
                        epoch_stat.update({'val_ce_loss': val_epoch_stat_avg['ce_loss'],
                                           'val_g_loss': val_epoch_stat_avg['g_loss'],
                                           'val_de_loss': val_epoch_stat_avg['de_loss'],
                                           'val_acc': val_epoch_stat_avg['accuracy']})

                        print('>>>>> validation ce loss: {0:.7f}, g loss: {1:.7f}, de loss: {2:.7f}, accuracy: {3:.7f}'.format(epoch_stat['val_ce_loss'], epoch_stat['val_g_loss'], epoch_stat['val_de_loss'], epoch_stat['val_acc']))

                    # -------------------- epoch: test phase --------------------
                    if self.test_iter is not None:

                        # init test batch
                        sess.run(self.test_iter.initializer, feed_dict={self.test_batch_size: args.test_batch_size})

                        test_loop_num = int(math.ceil(self.test_size / args.test_batch_size))

                        print('\ntest ', end='')
                        sys.stdout.flush()
                        test_epoch_stat = stat_helper()

                        with perf_checker() as pc:
                            for test_loop_index in range(test_loop_num):
                                print('{0} '.format(test_loop_index), end='')
                                sys.stdout.flush()

                                ce_loss, g_loss, de_loss, accuracy, weight = sess.run([self.ce_loss, self.gate_reg_loss, self.delta_e_loss, self.accuracy, self.actual_batch_size],
                                                                                      feed_dict={self.handle: test_handle, self.is_training: False})

                                test_epoch_stat.add_data({'ce_loss': ce_loss, 'g_loss': g_loss, 'de_loss': de_loss, 'accuracy': accuracy, 'weight': weight})

                            pc.print_time_used()

                        test_epoch_stat_avg = test_epoch_stat.get_weighted_avg(weight_key='weight')
                        epoch_stat.update({'test_ce_loss': test_epoch_stat_avg['ce_loss'],
                                           'test_g_loss': test_epoch_stat_avg['g_loss'],
                                           'test_de_loss': test_epoch_stat_avg['de_loss'],
                                           'test_acc': test_epoch_stat_avg['accuracy']})

                        print('>>>>> test ce loss: {0:.7f}, g loss: {1:.7f}, de loss: {2:.7f}, accuracy: {3:.7f}'.format(epoch_stat['test_ce_loss'], epoch_stat['test_g_loss'], epoch_stat['test_de_loss'], epoch_stat['test_acc']))

                    # -------------------- epoch: model saving --------------------
                    if args.save_model_dir is not None:
                        next_dir = dir_helper.prepare_cur_dir(epoch, args.model_metrics_scale * epoch_stat[args.model_max_metrics])

                        if next_dir is not None:
                            print('\n----- write model for epoch {0}...'.format(epoch), end='')
                            # builder = tf.saved_model.builder.SavedModelBuilder(next_dir)
                            # builder.add_meta_graph_and_variables(sess, ['med_qa_model'])
                            # builder.save()

                            saver = tf.train.Saver(model_variables)
                            saver.save(sess, next_dir, write_meta_graph=False, write_state=False)

                            print(' finished -----')

                    # get current best
                    if 'val_acc' in epoch_stat and epoch_stat['val_acc'] > best_val_accuracy:
                        best_val_accuracy = epoch_stat['val_acc']
                    if epoch_stat['test_acc'] > best_test_accuracy:
                        best_test_accuracy = epoch_stat['test_acc']

                    print('\n current best val acc: {0:.7f}, best test acc: {1:.7f}'.format(best_val_accuracy, best_test_accuracy))

                    # update per epoch stat
                    all_stat.add_data(epoch_stat)

                    # write file during training
                    if not stat_keys: # empty
                        stat_keys = epoch_stat.keys()

                        writer.write('\t'.join(stat_keys))
                        writer.write('\n')

                    values = [epoch_stat[k] for k in stat_keys]
                    writer.write('\t'.join(['{0}'.format(val) for val in values]))
                    writer.write('\n')
                    writer.flush()

                # -------------------- all epoch end, check whether we need to write a last model --------------------
                if args.save_model_dir is not None and args.need_last_model is True:
                    next_dir = dir_helper.prepare_final_dir(args.final_model_suffix)

                    print('\n----- write final model...', end='')
                    # builder = tf.saved_model.builder.SavedModelBuilder(next_dir)
                    # builder.add_meta_graph_and_variables(sess, ['med_qa_model'])
                    # builder.save()

                    saver = tf.train.Saver(model_variables)
                    saver.save(sess, next_dir, write_meta_graph=False, write_state=False)

                    print(' finished -----')

                print('\nin total used {0} sec\n'.format(pc_all_epoch.time_used()))

        return all_stat

    def get_accuracy(self, fact, pred_dict):
        total = len(fact)
        correct = 0
        k_total = len([k for k, v in fact.items() if v[1] > 0.5])
        k_correct = 0

        for k, v in pred_dict.items():
            # k = k.decode('utf-8')
            if fact[k][0] == v[0]:
                correct += 1
                if fact[k][1] > 0.5:
                    k_correct += 1

        acc = correct / total
        k_acc = k_correct / k_total

        return acc, k_acc, correct, k_correct, total, k_total

    def output_debug(self, fact, pred_dict, pred_debug_file):
        if pred_debug_file is not None:
            with open(pred_debug_file, 'w', newline='', encoding='utf-8') as debug_f:
                pred_arr = sorted([[k, v[0], v[1]] for k,v in pred_dict.items()], key = lambda x: x[0])
                for item in pred_arr:
                    debug_f.write('{0}\t{1}\t{2}\t{3}'.format(item[0], fact[item[0]][0], item[1], ', '.join([str(v) for v in item[2]])))
                    debug_f.write('\n')

    def load_and_test_model_es(self, test_es_file, test_phase_seed, args, test_small=-1, pred_debug_file=None):

        print('\n--------------------Load ans Test ES--------------------')
        print('Test ES file: {0}'.format(test_es_file))
        print('test_small:', test_small)
        print(args)
        print()

        # result should not change, because testing does not depend on random numbers
        with perf_checker() as pc:
            test_data = self.data.load_data_from_es_new(test_es_file, self.truncate_length, self.topn)

            self.build_test_model_es(test_data, test_phase_seed, args, test_small)

            print('build testing model used {0:.7f} sec'.format(pc.time_used()))

        tf_stat, pred_dict, *temp = self.test_model(args)

        fact = {id: [np.argmax(l), ins_f[0], ins_f[1]] for id, l, ins_f in zip(test_data[3], test_data[2], test_data[1])}

        acc, k_acc, correct, k_correct, total, k_total = self.get_accuracy(fact, pred_dict)

        self.output_debug(fact, pred_dict, pred_debug_file)

        if abs(acc-tf_stat['accuracy'])>1e-5:
            print('Warning: TF accuracy inconsistent')

        print('TF: ce loss: {0}, gate loss: {1} accuracy: {2}'.format(tf_stat['ce_loss'], tf_stat['g_loss'], tf_stat['accuracy']))
        print('Based on prediction, {0} / {1} correct = {2}'.format(correct, total, acc))
        print('Knowledge problem, {0} / {1} correct = {2}'.format(k_correct, k_total, k_acc))

    def load_and_test_model_es_ensemble(self, test_es_file, test_phase_seed, model_files, args, test_small=-1, pred_debug_file=None):

        print('\n--------------------Load ans Test ES Ensemble--------------------')
        print('Test ES file: {0}'.format(test_es_file))
        print('test_small:', test_small)
        print(args)
        print()

        # result should not change, because testing does not depend on random numbers
        with perf_checker() as pc:
            test_data = self.data.load_data_from_es_new(test_es_file, self.truncate_length, self.topn)

            self.build_test_model_es(test_data, test_phase_seed, args, test_small)

            print('build testing model used {0:.7f} sec'.format(pc.time_used()))

        fact = {id: [np.argmax(l), ins_f[0]] for id, l, ins_f in zip(test_data[3], test_data[2], test_data[1])}

        correct_num = []
        k_correct_num = []
        all_k_acc = []
        total = len(fact)
        k_total = len([k for k, v in fact.items() if v[1] > 0.5])

        score_dict = {}

        for model_f in model_files:
            args.model_prefix = model_f

            tf_stat, pred_dict = self.test_model(args)

            acc, k_acc, correct, k_correct, *temp = self.get_accuracy(fact, pred_dict)

            correct_num.append(correct)
            k_correct_num.append(k_correct)

            for k, v in pred_dict.items():
                if k in score_dict:
                    score_dict[k] = np.add(score_dict[k], v[2])
                else:
                    score_dict[k] = v[2]

        score_pred_dict = {}
        for k, v in score_dict.items():
            score_pred_dict[k] = [np.argmax(v), v]

        acc, k_acc, correct, k_correct, *temp = self.get_accuracy(fact, score_pred_dict)

        self.output_debug(fact, score_pred_dict, pred_debug_file)

        print('Ensemble result for {0} models:'.format(len(model_files)))
        print(model_files)

        max_c = np.max(correct_num)
        min_c = np.min(correct_num)
        avg_c = np.mean(correct_num)
        print('\nAccuracy max: {0} / {6} = {1:.7f}, min: {2} / {6} = {3:.7f}, avg: {4} / {6} = {5:.7f}'.format(
            max_c, max_c/total, min_c, min_c/total, avg_c, avg_c/total, total))

        max_k_c = np.max(k_correct_num)
        min_k_c = np.min(k_correct_num)
        avg_k_c = np.mean(k_correct_num)
        print('Knowledge accuracy max: {0} / {6} = {1:.7f}, min: {2} / {6} = {3:.7f}, avg: {4} / {6} = {5:.7f}'.format(
                max_k_c, max_k_c / k_total, min_k_c, min_k_c / k_total, avg_k_c, avg_k_c / k_total, k_total))

        print('\nEnsemble result:')
        print('Based on prediction, {0} / {1} correct = {2}'.format(correct, total, acc))
        print('Knowledge problem, {0} / {1} correct = {2}'.format(k_correct, k_total, k_acc))


    def load_and_test_model_tfrecord(self, test_tfrecord_file, test_phase_seed, args, test_small=-1):

        print('\n--------------------Load ans Test--------------------')
        print('Test tfrecord file: {0}'.format(test_tfrecord_file))
        print('test_small:', test_small)
        print(args)
        print()

        # result should not change, because testing does not depend on random numbers
        with perf_checker() as pc:

            self.build_test_model_tfrecord(test_tfrecord_file, test_phase_seed, args, test_small)

            print('build testing model used {0:.7f} sec'.format(pc.time_used()))

        self.test_model(args)

    # -------------------- Common module for testing, with metric returned --------------------
    def test_model(self, args):
        if args.model_prefix is None:
            print('Error! Please make the model first!')

        print('\n----- test -----')

        with tf.Session() as sess:

            #file_writer = tf.summary.FileWriter('./checkpoint/', sess.graph)

            # -------------------- prepare model --------------------
            with perf_checker() as pc:

                sess.run(tf.global_variables_initializer())

                test_handle = sess.run(self.test_iter.string_handle())

                # self.embedding_ds = sess.graph.get_tensor_by_name('embedding_ds:0')
                # #self.embedding_input = tf.variables_initializer([sess.graph.get_tensor_by_name('embedding_input:0')])
                # self.embedding_input = sess.graph.get_operation_by_name('embedding_input/Assign')

                sess.run(self.embedding_input.initializer, feed_dict={self.embedding_ds: self.data.embed_vec})

                sess.run(self.test_iter.initializer, feed_dict={self.test_batch_size: args.test_batch_size})

                # get all model variables for model load and save
                print('loading model {0}'.format(args.model_prefix), end='')

                model_variables = {val.name: val for val in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}

                # model_variables.pop('delta_embedding/Adam:0', None)
                # model_variables.pop('delta_embedding/Adam_1:0', None)
                # model_variables.pop('delta_embedding:0', None)

                # print('restoring model variables:')
                # for val_name, val in model_variables.items():
                #     print('{0} -> {1}'.format(val_name, val.shape))

                saver = tf.train.Saver(model_variables)

                saver.restore(sess, args.model_prefix)

                print('\nprepare model used {0:.7f} sec'.format(pc.time_used()))

            # -------------------- we do test --------------------
            with perf_checker() as pc:

                # self.loss = sess.graph.get_tensor_by_name('loss:0')
                # self.prediction = sess.graph.get_tensor_by_name('prediction:0')
                # self.accuracy = sess.graph.get_tensor_by_name('accuracy:0')
                # self.actual_batch_size = sess.graph.get_tensor_by_name('actual_batch_size:0')
                #
                # self.handle = sess.graph.get_tensor_by_name('data_handle:0')

                test_loop_num = int(math.ceil(self.test_size / args.test_batch_size))
                test_stat = stat_helper()
                predictions = {}

                print('\ntest ', end='')
                sys.stdout.flush()

                for test_loop_index in range(test_loop_num):
                    print('{0} '.format(test_loop_index), end='')
                    sys.stdout.flush()
                    ce_loss, g_loss, de_loss, loss, accuracy, weight, ids, pred, p_logits, probs = sess.run([self.ce_loss, self.gate_reg_loss, self.delta_e_loss, self.loss, self.accuracy, self.actual_batch_size, self.ids, self.prediction, self.logits, self.prob],
                                                      feed_dict={self.handle: test_handle, self.is_training: False})

                    test_stat.add_data({'ce_loss':ce_loss, 'g_loss':g_loss, 'de_loss': de_loss, 'accuracy': accuracy, 'weight':weight})

                    for id, p, logits, prob in zip(ids, pred, p_logits, probs):
                        predictions[id.decode('utf-8')] = [p, logits, prob]
                print()

                stat = test_stat.get_weighted_avg(weight_key='weight')

                print('test ce_loss: {0:.7f}, g_loss: {1:.7f}, de_loss: {2:.7f}, loss: {3:.7f}, accuracy: {4:.7f}, used {5:.7f} sec'.format(stat['ce_loss'], stat['g_loss'], stat['de_loss'], stat['ce_loss']+stat['g_loss']+stat['de_loss'], stat['accuracy'], pc.time_used()))

        return stat, predictions

    def write_stat_file(self, stat_file, stat):
        if stat_file is not None:
            with open(stat_file, 'w', newline='', encoding='utf-8') as outf:
                keys = stat.data.keys()
                outf.write('\t'.join([k for k in keys]))
                outf.write('\n')

                values = [stat.data[k] for k in keys]
                for vals in zip(*values):
                    outf.write('\t'.join(['{0}'.format(val) for val in vals]))
                    outf.write('\n')

# --------------------------------------------------

def do_data_generation(args):
    # embedding_f = '/Users/shenge/my/projects/history_run/2/corpus_embedding_200_win_10_min_5_iter_20_neg_20.txt'
    # embedding_f = '/Users/shenge/my/projects/history_run/1/corpus_embedding_200_win_5_min_5_iter_20_neg_5.txt'
    #embedding_f = '../data/embedding/corpus_embedding_200.txt'
    #embedding_f = '../data/embedding/corpus_297k_200.txt'
    embedding_f = '../data/embedding/corpus_0711_366k_200.txt'
    #embedding_f = '../data/embedding/corpus_embedding_200_ultra.txt'
    dict_cut_file = '../data/alphaMLE_dict.txt'

    # file_list = ['../data/es_result_0625_titleplus_all_book/kexue_2018jingbian_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/kexue_exercise_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/renwei_exam_not2017_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/renwei_exercise_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/network_exam_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/network_exercise_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/final_generate_keep_shuffle_ESresult.txt']
    # file_list = ['../data/es_result_0625_titleplus_all_book/kexue_2018jingbian_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/kexue_exercise_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/renwei_exam_not2017_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/renwei_exercise_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/network_exam_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/network_exercise_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/zhuojian_lc_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/zhuojian_notlc_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/data_zb_f_ESresult.txt']
    # file_list = ['../data/es_result_0625_titleplus_all_book/kexue_2018jingbian_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/kexue_exercise_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/renwei_exam_not2017_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/renwei_exercise_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/zhuojian_lc_ESresult.txt',
    #              '../data/es_result_0625_titleplus_all_book/zhuojian_notlc_ESresult.txt']
    # file_list = ['../data/es_result_0611_titleplus_all_book/kexue_2018jingbian_ESresult.txt',
    #              '../data/es_result_0611_titleplus_all_book/kexue_exercise_ESresult.txt',
    #              '../data/es_result_0611_titleplus_all_book/renwei_exam_not2017_ESresult.txt',
    #              '../data/es_result_0611_titleplus_all_book/renwei_exercise_ESresult.txt']
    file_list = ['../data/ES/old/new_renwei_exam_2017_ESresult.txt']

    #train_tf = '../data/tfrecords/train_0710_zb_e297k_es0625.tfrecords'
    train_tf = '../data/tfrecords/val_renwei_2017_e366k_es0625.tfrecords'
    val_tf = '../data/tfrecords/val_0710_zb_e297k_es0625.tfrecords'
    test_tf = '../data/tfrecords/test_all.tfrecords'

    args.embedding_file = embedding_f
    args.wordcut_file = dict_cut_file
    #args.cut_norm = True
    #args.cut_norm = False

    test = MedExamData(args)

    test.convert_es_to_tfrecord(file_list,
                                train_out_file=train_tf,
                                validate_out_file=val_tf,
                                test_out_file=test_tf,
                                split_stratified=True,
                                shuffle_seed=0,
                                validate_ratio=0,#0.10,
                                test_ratio=0,#0.10,
                                text_truncate_length=100,
                                topn=10)

    # test.validate_tfrecord_file(train_tf)
    # test.validate_tfrecord_file(val_tf)
    # test.validate_tfrecord_file(test_tf)


def do_debug_model(args, train_stat_file = None):

    #embedding_f = '/Users/shenge/my/projects/history_run/2/corpus_embedding_200_win_10_min_5_iter_20_neg_20.txt'
    embedding_f = '/Users/shenge/my/projects/history_run/1-2/1/corpus_embedding_200_win_5_min_5_iter_20_neg_5.txt'
    model_p = './models/model-latest'
    train_tf = '/Users/shenge/my/projects/history_run/1-2/train_200.tfrecords'
    val_tf = '/Users/shenge/my/projects/history_run/1-2/val_200.tfrecords'
    test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    test_tf = '/Users/shenge/my/projects/history_run/1-2/test_200.tfrecords'
    test_small = 50


    # modify args
    args.embedding_file = embedding_f
    args.wordcut_file = '../data/alphaMLE_dict.txt'
    args.cut_norm = False

    args.use_delta_embedding = True
    args.delta_embed_loss_scale = 1.0
    # args.cut_norm = True
    # args.global_seed = 0
    # args.truncate_length = 100
    # args.topn = 10
    args.epoch_limit = 1#10
    args.train_batch_per_epoch = 2
    args.train_batch_size = 5
    args.validate_batch_per_epoch = 2
    args.validate_batch_size = 5
    args.save_model_dir = './models/'
    args.model_keep = 1 # important
    args.model_max_metrics = 'test_acc'
    args.model_metrics_scale = 1.0
    args.need_last_model = True
    args.debug_info = False
    args.summary_dir = './checkpoint/'

    model = MedExamModel(args)

    train_stat = model.load_and_train_model_test_es(train_tfrecord_file=train_tf, train_phase_seed=1, args=args, test_es_file=test_es, test_small=test_small)

    model.write_stat_file(train_stat_file, train_stat)

    args.test_batch_size = 5
    args.model_prefix = model_p

    model.load_and_test_model_es(test_es_file=test_es, test_phase_seed=0, args=args, test_small=test_small)

def do_real_model(args, train_stat_file = None):
    #embedding_f = '../data/embedding/corpus_297k_200.txt'
    embedding_f = '../data/embedding/corpus_0711_366k_200.txt'
    #embedding_f = '../data/embedding/corpus_297k_200_u.txt'
    #embedding_f = '../data/embedding/corpus_embedding_200_ultra.txt'

    #train_tf = '../data/tfrecords/train_130k_if_e297k.tfrecords'
    #train_tf = '../data/tfrecords/train_130k_if_e148k.tfrecords'
    #train_tf = '../data/tfrecords/train_127k_if_e297k_es0625.tfrecords'
    #train_tf = '../data/tfrecords/train_151k_if_e297k_es0625.tfrecords'
    # train_tf = '../data/tfrecords/train_0704_126k_e297k_es0625.tfrecords'
    # val_tf = '../data/tfrecords/val_0704_1k_e297k_es0625.tfrecords'
    # train_tf = '../data/tfrecords/train_0705_zj_161k_e297k_es0625.tfrecords'
    #val_tf = '../data/tfrecords/val_0705_zj_2k_e297k_es0625.tfrecords'
    #train_tf = '../data/tfrecords/train_0705_162k_zj_e297k_es0625.tfrecords'
    #train_tf = '../data/tfrecords/train_0706_312k_zjall_e297k_es0625.tfrecords'
    #train_tf = '../data/tfrecords/train_0709_208k_nonet_e297k_es0625.tfrecords'
    #val_tf = None
    #val_tf = '../data/tfrecords/val_0709_2k_nonet_e297k_es0625.tfrecords'
    #train_tf = '../data/tfrecords/train_0711_323k_all_e297k_es0711.tfrecords'
    #train_tf = '../data/tfrecords/train_0712_323k_all_e366k_es0711.tfrecords'
    train_tf = '../data/tfrecords/train_0716_allwith17_e366k_es0625.tfrecords'
    val_tf = '../data/tfrecords/val_renwei_2017_e366k_es0625.tfrecords'

    args.use_delta_embedding = True
    args.delta_embed_loss_scale = 1.0
    args.use_instance_feature = True
    args.use_orthogonal_init = False

    args.use_mlp_vscale_init = True
    args.use_grad_clip = True
    args.grad_clip_norm = 5.0

    # args.lr = 0.01

    train_seed = 0

    # train_stat_file = './train_[u/n]_[l/s]_[d/n]_[1.0/0.1].txt'
    # train_stat_file = './train_{0}.txt'.format(train_seed)
    train_stat_file = './train_test.txt'
    #train_stat_file = './train_0711_366k_gclip.txt'

    model_p = './models/model-latest'

    test_es = '../data/ES/old/gold_exercise_file4_ESresult.txt'

    test_small = -1

    # modify args
    args.embedding_file = embedding_f
    args.wordcut_file = '../data/alphaMLE_dict.txt'
    args.cut_norm = False

    #args.use_delta_embedding = False
    #args.delta_embed_loss_scale = 1.0
    # args.cut_norm = True
    # args.global_seed = 0
    # args.truncate_length = 100
    # args.topn = 10
    args.epoch_limit = 1000
    args.train_batch_per_epoch = 100
    args.train_batch_size = 40

    args.validate_batch_size = 50
    args.save_model_dir = './models/'
    args.model_keep = 5
    args.model_max_metrics = 'test_acc'
    # args.model_max_metrics = 'val_acc' # remmember to modify this!
    args.model_metrics_scale = 1.0
    args.need_last_model = True
    args.debug_info = False
    args.summary_dir = './checkpoint/'

    model = MedExamModel(args)

    train_stat = model.load_and_train_model_test_es(train_tfrecord_file=train_tf, train_phase_seed=train_seed, args=args, train_stat_file=train_stat_file,
                                                    test_es_file=test_es, test_small=test_small, validate_tfrecord_file=val_tf)

    #model.write_stat_file(train_stat_file, train_stat)

    args.test_batch_size = 50
    args.model_prefix = model_p
    model.load_and_test_model_es(test_es_file=test_es, test_phase_seed=0, args=args, test_small=test_small)

def do_real_model_test(args):
    # ----- history 1
    # embedding_f = '/Users/shenge/my/projects/history_run/1/corpus_embedding_200_win_5_min_5_iter_20_neg_5.txt'
    # model_p = '/Users/shenge/my/projects/history_run/1/models/model-128'
    # #model_p = '/Users/shenge/my/projects/history_run/1/models/model-latest'
    # test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # test_tf = '/Users/shenge/my/projects/history_run/1/test_200.tfrecords'


    # ----- history 2
    # embedding_f = '/Users/shenge/my/projects/history_run/2/corpus_embedding_200_win_10_min_5_iter_20_neg_20.txt'
    # #model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    # model_p = '/Users/shenge/my/projects/history_run/2/models/model-latest'
    # #model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'

    # ----- history 3
    # embedding_f = '/Users/shenge/my/projects/history_run/3/corpus_embedding_200.txt'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    # model_p = '/Users/shenge/my/projects/history_run/3/models/model-latest'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # #test_es = '../data/ES/test.txt'
    # # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'

    # ----- history 8
    # embedding_f = '/Users/shenge/my/projects/history_run/3-8_no_norm/5_8/corpus_embedding_200_ultra.txt'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    # model_p = '/Users/shenge/my/projects/history_run/3-8_no_norm/5_8/8/models/model-1202'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # #test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # test_es = '../data/ES/renwei_2017_correct.txt'
    # # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'

    # ----- history 9
    # embedding_f = '/Users/shenge/my/projects/history_run/9-10_de/9/corpus_297k_200.txt'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    # model_p = '/Users/shenge/my/projects/history_run/9-10_de/9/models/model-726'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # #test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # test_es = '../data/ES/renwei_2017_correct.txt'
    # # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'
    # args.use_delta_embedding = True
    # args.delta_embed_loss_scale = 1.0
    # args.use_instance_feature = False

    # ----- history 10
    # embedding_f = '/Users/shenge/my/projects/history_run/9-10_de/10/corpus_297k_200_u.txt'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    # model_p = '/Users/shenge/my/projects/history_run/9-10_de/10/models/model-897'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # # test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # test_es = '../data/ES/renwei_2017_correct.txt'
    # # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'
    # args.use_delta_embedding = True
    # args.delta_embed_loss_scale = 1.0

    # ----- history 11
    # embedding_f = '/Users/shenge/my/projects/history_run/11_if_neg/corpus_297k_200.txt'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    # model_p = '/Users/shenge/my/projects/history_run/11_if_neg/models/model-505'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # # test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # test_es = '../data/ES/renwei_2017_correct_txt_ESresult.txt'
    # # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'
    # args.use_delta_embedding = True
    # args.delta_embed_loss_scale = 1.0
    # args.use_instance_feature = True

    # ----- history 12
    # embedding_f = '/Users/shenge/my/projects/history_run/12-13_if_newes/corpus_297k_200.txt'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    # model_p = '/Users/shenge/my/projects/history_run/12-13_if_newes/12/models/model-872'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # #test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # #test_es = '../data/ES/renwei_2017_correct.txt'
    # #test_es = '../data/ES/renwei_2017_correct_es0625.txt'
    # #test_es = '../data/ES/new_renwei_exam_2017_ESresult.txt'
    # test_es = '../data/es_result_0625_titleplus_all_book/zhuojian_lc_ESresult.txt'
    # #test_es = '../data/ES/gold_exercise_ESresult.txt'
    # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'
    # args.use_delta_embedding = True
    # args.delta_embed_loss_scale = 1.0
    # args.use_instance_feature = True

    # ----- history 14
    # embedding_f = '/Users/shenge/my/projects/history_run/12-13_if_newes/corpus_297k_200.txt'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    # model_p = '/Users/shenge/my/projects/history_run/14_ve/models/75/model-925'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # # test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # # test_es = '../data/ES/renwei_2017_correct.txt'
    # # test_es = '../data/ES/renwei_2017_correct_es0625.txt'
    # test_es = '../data/ES/new_renwei_exam_2017_ESresult.txt'
    # #test_es = '../data/es_result_0625_titleplus_all_book/zhuojian_lc_ESresult.txt'
    # # test_es = '../data/ES/gold_exercise_ESresult.txt'
    # #test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'
    # args.use_delta_embedding = True
    # args.delta_embed_loss_scale = 1.0
    # args.use_instance_feature = True

    # ----- history 15
    # embedding_f = '/Users/shenge/my/projects/history_run/12-13_if_newes/corpus_297k_200.txt'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    # model_p = '/Users/shenge/my/projects/history_run/15_zj/models/model-724'
    # # model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # # test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # # test_es = '../data/ES/renwei_2017_correct.txt'
    # # test_es = '../data/ES/renwei_2017_correct_es0625.txt'
    # test_es = '../data/ES/old/new_renwei_exam_2017_ESresult.txt'
    # #test_es = '../data/es_result_0625_titleplus_all_book/data_zb_f_ESresult.txt'
    # # test_es = '../data/es_result_0625_titleplus_all_book/zhuojian_lc_ESresult.txt'
    # # test_es = '../data/ES/gold_exercise_ESresult.txt'
    # # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'
    # args.use_delta_embedding = True
    # args.delta_embed_loss_scale = 1.0
    # args.use_instance_feature = True

    # ----- history 16
    # #embedding_f = '/Users/shenge/my/projects/history_run/16/e297k/corpus_297k_200.txt'
    # embedding_f = '/Users/shenge/my/projects/history_run/16/e366k/corpus_0711_366k_200.txt'
    # #model_p = '/Users/shenge/my/projects/history_run/16/e297k/model-665'
    # model_p = '/Users/shenge/my/projects/history_run/16/e366k/model-773'
    # test_es = '../data/ES/0711/renwei_exam_2017_ESresult.txt'# gold_exercise_file1_ESresult.txt
    # #test_es = '../data/ES/old/new_renwei_exam_2017_ESresult.txt'# gold_exercise_file1_ESresult.txt
    # #test_es = '../data/ES/0711/gold_exercise_file1_ESresult.txt'
    # #test_es = '../data/ES/0711/gold_exercise_file2_ESresult.txt'
    # #test_es = '../data/ES/0711/gold_exercise_file3_ESresult.txt'
    # #test_es = '../data/ES/0711/gold_exercise_file4_ESresult.txt'
    # # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'
    # args.use_delta_embedding = True
    # args.delta_embed_loss_scale = 1.0
    # args.use_instance_feature = True

    # ----- history 17, with 17 e366k
    embedding_f = '../data/embedding/corpus_0711_366k_200.txt'
    # model_p = '/Users/shenge/my/projects/history_run/2/models/model-416'
    model_p = '/Users/shenge/my/projects/history_run/17_v1_with17/model-946'
    # model_p = '/Users/shenge/my/projects/history_run/2/models/model-128'
    # test_es = '../data/ES/renwei_exam_2017_ESresult.txt'
    # test_es = '../data/ES/renwei_2017_correct.txt'
    # test_es = '../data/ES/renwei_2017_correct_es0625.txt'
    #test_es = '../data/ES/old/new_renwei_exam_2017_ESresult.txt'
    test_es = '../data/ES/old/gold_exercise_file4_ESresult.txt'
    # test_es = '../data/es_result_0625_titleplus_all_book/data_zb_f_ESresult.txt'
    # test_es = '../data/es_result_0625_titleplus_all_book/zhuojian_lc_ESresult.txt'
    # test_es = '../data/ES/gold_exercise_ESresult.txt'
    # test_tf = '/Users/shenge/my/projects/history_run/2/test_200.tfrecords'
    args.use_delta_embedding = True
    args.delta_embed_loss_scale = 1.0
    args.use_instance_feature = True

    args.use_orthogonal_init = False # tested, no performance increase
    args.use_mlp_vscale_init = True # train param, should be no influence with test
    args.use_grad_clip = True

    # modify args
    args.embedding_file = embedding_f
    args.wordcut_file = '../data/alphaMLE_dict.txt'
    #args.cut_norm = False
    #args.cut_norm = True
    # args.global_seed = 0
    # args.truncate_length = 100
    # args.topn = 10
    args.epoch_limit = 500
    args.train_batch_per_epoch = 100
    args.train_batch_size = 20
    args.validate_batch_per_epoch = 10
    args.validate_batch_size = 20
    args.save_model_dir = './models/'
    args.model_keep = 3
    args.model_max_metrics = 'test_acc'
    args.model_metrics_scale = 1.0
    args.model_skip = 1
    args.need_last_model = True
    args.debug_info = False
    args.summary_dir = './checkpoint/'

    model = MedExamModel(args)

    args.test_batch_size = 50
    args.model_prefix = model_p
    model.load_and_test_model_es(test_es_file=test_es, test_phase_seed=1, args=args, test_small=-1, pred_debug_file='./debug.txt')

    # Ensemble 5
    # model.load_and_test_model_es_ensemble(test_es_file=test_es,
    #                                       test_phase_seed=1,
    #                                       model_files=['/Users/shenge/my/projects/history_run/12-13_if_newes/12/models/model-369',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/12/models/model-567',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/12/models/model-671',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/12/models/model-748',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/12/models/model-872'],
    #                                       args=args,
    #                                       test_small=-1,
    #                                       pred_debug_file='./debug.txt')

    # Ensemble 25, 14
    # model.load_and_test_model_es_ensemble(test_es_file=test_es,
    #                                       test_phase_seed=1,
    #                                       model_files=['/Users/shenge/my/projects/history_run/14_ve/models/71/model-638',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/71/model-915',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/71/model-975',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/71/model-990',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/71/model-994',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/72/model-597',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/72/model-671',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/72/model-787',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/72/model-901',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/72/model-994',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/73/model-937',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/73/model-939',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/73/model-940',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/73/model-943',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/73/model-959',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/74/model-752',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/74/model-753',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/74/model-811',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/74/model-813',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/74/model-816',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/75/model-562',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/75/model-743',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/75/model-924',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/75/model-925',
    #                                                    '/Users/shenge/my/projects/history_run/14_ve/models/75/model-930'],
    #                                       args=args,
    #                                       test_small=-1,
    #                                       pred_debug_file='./debug.txt')

    # Ensemble 25
    # model.load_and_test_model_es_ensemble(test_es_file=test_es,
    #                                       test_phase_seed=1,
    #                                       model_files=['/Users/shenge/my/projects/history_run/12-13_if_newes/13/61/model-553', # 61
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/61/model-639',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/61/model-754',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/61/model-912',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/61/model-1197',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/62/model-413', # 62
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/62/model-846',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/62/model-874',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/62/model-1009',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/62/model-1010',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/63/model-529', # 63
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/63/model-620',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/63/model-621',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/63/model-824',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/63/model-845',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/64/model-515', # 64
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/64/model-659',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/64/model-1060',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/64/model-1398',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/64/model-1416',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/65/model-612',# 65
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/65/model-632',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/65/model-633',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/65/model-634',
    #                                                    '/Users/shenge/my/projects/history_run/12-13_if_newes/13/65/model-1191'],
    #                                       args=args,
    #                                       test_small=-1,
    #                                       pred_debug_file='./debug.txt')

    # args.test_batch_size = 5
    # args.model_prefix = './models/model-latest'
    # # test_stat = model.load_and_test_model_es(test_es_file = '../data/latest/ES_result_jieba_all/exam-2017_ESresult.txt',
    # #                                       test_phase_seed = 0,
    # #                                       model_dir = './models/model-latest',
    # #                                       test_small = 50,
    # #                                       test_batch = 50,
    # #                                       lr=1e-3)
    #
    #model.load_and_test_model_tfrecord(test_tfrecord_file = test_tf, test_phase_seed = 1, args = args, test_small =-1)
    #
    # test_stat = model.load_and_test_model_tfrecord(test_tfrecord_file='../data/tfrecords/test_200.tfrecords',
    #                                                test_phase_seed=1,
    #                                                args=args,
    #                                                test_small=50)
    #
    # model.write_stat_file(train_stat_file, train_stat, test_stat_file, test_stat)

def do_check_checkpoint():
    from tensorflow.python import pywrap_tensorflow

    fname = './models/model-latest'

    reader = pywrap_tensorflow.NewCheckpointReader(fname)

    var_to_shape_map = reader.get_variable_to_shape_map()
    for k, v in var_to_shape_map.items():
        print(k, '->', v)

if __name__ == '__main__':
    parser = MedExamModelParamParser()
    args = parser.do_parse(sys.argv)

    #test()
    #do_data_generation(args)
    #do_debug_model(args, './train.txt')
    #do_check_checkpoint()
    do_real_model_test(args)
    do_real_model(args, './train.txt')

    # do_batch_check()
    # data = MedExamData('../data/embedding/corpus_embedding_200.txt',
    #                    '../data/alphaMLE_dict.txt')
    #k
    # res = data.load_data_from_es_new(es_file='../data/ES/renwei_exam_2017_ESresult.txt',
    #                                  text_truncate_length=100,
    #                                  topn=10)
    #
    # print(len(res[0]))
    pass