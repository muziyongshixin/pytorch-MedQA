#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import pickle
import sys
import keras
from keras import backend as K
from keras.optimizers import Adagrad, Adam
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import multi_gpu_model
from keras.callbacks import Callback,LearningRateScheduler
import argparse
import random
from collections import defaultdict
from keras.models import load_model
import h5py
import json
import jieba
jieba.load_userdict("./alphaMLE_dict.txt")
import operator
from collections import Counter

def load_word_emb(word_emb_path, char2id, emb_dim, num_words):
    index_from = 3
    from gensim.models import KeyedVectors
    vector_mean = []
    OOV = 0
    word_vectors = KeyedVectors.load_word2vec_format(word_emb_path, binary=False)
    for k,v in char2id.items():
        try:
            vector_mean.append(word_vectors[k])
        except:
            OOV += 1
    print("among %d words, %d are OOV, %d are found"%(len(char2id), OOV, len(vector_mean)))
    vector_mean = np.array(vector_mean)
    mean_vector = np.mean(vector_mean,0)
    word_embeddings = np.zeros((num_words+1,emb_dim))
    for k,v in char2id.items():
        if v < num_words+1-index_from:
            try:
                insert_vec = word_vectors[k]
            except:
                insert_vec = mean_vector
            word_embeddings[v+index_from,:] = insert_vec
    return word_embeddings

def load_test(path,word2id, num_words, maxlenq=None,maxlena=None, seed=113,
              start_char=1, oov_char=2, index_from=3):
    def filter_words(list_of_words):
        return [word for word in list_of_words if len(word)>0 and word != " "]
    questions,answer_pos,answer_neg = [],[],[]
    questions_idx,answer_pos_idx,answer_neg_idx = [],[],[]
    question_category_test = []
    with open(path) as f:
        lines = [line for line in f.readlines()]
        for line in lines:
            #print(line)
            q = json.loads(line.strip())
            abcde = ["A","B","C","D","E"]
            if q.get("question") and q.get("A") and q.get("B") and q.get("C") and q.get("D") and q.get("E") and q.get("answer") in abcde:
                for key in ["A","B","C","D","E"]:
                    q[key] = filter_words(list(jieba.cut(q[key], cut_all=True)))[:maxlena]
                q["question"] = filter_words(list(jieba.cut(q["question"], cut_all=True)))[:maxlenq]

                for choice in abcde:
                    if choice != q.get("answer"):
                        questions.append(q.get("question"))
                        if q["logic"][0] == 0:
                            answer_pos.append(q.get(q.get("answer")))
                            answer_neg.append(q.get(choice))
                        else:
                            answer_neg.append(q.get(q.get("answer")))
                            answer_pos.append(q.get(choice))
                        question_category_test.append((max(q["question_category"].iteritems(), key=operator.itemgetter(1))[0], q["logic"][0]))

    FILTER_MOST_FREQ = 2
    for str in questions:
        ids = [start_char]
        for w in str:
            if word2id.get(w) is not None:
                if word2id[w] > FILTER_MOST_FREQ:
                    if word2id[w] < num_words:
                        ids.append(word2id[w]+index_from)
                    else:
                        ids.append(oov_char)
            else:
                ids.append(oov_char)
        questions_idx.append(ids)

    for str in answer_pos:
        ids = [start_char]
        for w in str:
            if word2id.get(w) is not None:
                if word2id[w] > FILTER_MOST_FREQ:
                    if word2id[w] < num_words:
                        ids.append(word2id[w]+index_from)
                    else:
                        ids.append(oov_char)
            else:
                ids.append(oov_char)
        answer_pos_idx.append(ids)

    for str in answer_neg:
        ids = [start_char]
        for w in str:
            if word2id.get(w) is not None:
                if word2id[w] > FILTER_MOST_FREQ:
                    if word2id[w] < num_words:
                        ids.append(word2id[w]+index_from)
                    else:
                        ids.append(oov_char)
            else:
                ids.append(oov_char)
        answer_neg_idx.append(ids)
    return questions_idx,answer_pos_idx,answer_neg_idx,question_category_test

def load_data_with_2017_test(path='train.txt',path2='valid.txt',path3='test.txt', num_words=None, skip_top=0,
              maxlenq=None,maxlena=None, seed=113,
              start_char=1, oov_char=2, index_from=3, test_ratio = 0.1, **kwargs):
    def filter_words(list_of_words):
        return [word for word in list_of_words if len(word)>0 and word != " "]

    questions,answer_pos,answer_neg = [],[],[]
    questions_idx,answer_pos_idx,answer_neg_idx = [],[],[]
    vocab = Counter()
    word2id = {}
    q_len = []
    a_len = []
    train_until = []
    for apath in [path, path2] :
        with open(apath) as f:
            lines = [line for line in f.readlines()]
            for line in lines:
                #print(line)
                q = json.loads(line.strip())
                abcde = ["A","B","C","D","E"]
                if q.get("question") and q.get("A") and q.get("B") and q.get("C") and q.get("D") and q.get("E") and q.get("answer") in abcde:
                    for key in ["A","B","C","D","E"]:
                        q[key] = filter_words(list(jieba.cut(q[key], cut_all=True)))
                        a_len.append(len(q[key]))
                        q[key] = q[key][:maxlena] # comment this line if using char level
                    q["question"] = filter_words(list(jieba.cut(q["question"], cut_all=True)))
                    q_len.append(len(q["question"]))
                    q["question"] = q["question"][:maxlenq]

                    for str in [q.get("question"), q.get("A"), q.get("B"), q.get("C"), q.get("D"), q.get("E")]:
                        for c in str:
                            vocab[c] += 1

                    for choice in abcde:
                        if choice != q.get("answer"):
                            questions.append(q.get("question"))
                            if q["logic"][0] == 0:
                                answer_pos.append(q.get(q.get("answer")))
                                answer_neg.append(q.get(choice))
                            else:
                                answer_neg.append(q.get(q.get("answer")))
                                answer_pos.append(q.get(choice))
        train_until.append(len(answer_neg))
    question_category_test = []
    with open(path3) as f:
        lines = [line for line in f.readlines()]
        for line in lines:
            #print(line)
            q = json.loads(line.strip())
            abcde = ["A","B","C","D","E"]
            if q.get("question") and q.get("A") and q.get("B") and q.get("C") and q.get("D") and q.get("E") and q.get("answer") in abcde:
                for key in ["A","B","C","D","E"]:
                    q[key] = filter_words(list(jieba.cut(q[key], cut_all=True)))[:maxlena] # comment this line if using char level
                q["question"] = filter_words(list(jieba.cut(q["question"], cut_all=True)))[:maxlenq]

                for str in [q.get("question"), q.get("A"), q.get("B"), q.get("C"), q.get("D"), q.get("E")]:
                    for c in str:
                        vocab[c] += 1

                for choice in abcde:
                    if choice != q.get("answer"):
                        questions.append(q.get("question"))
                        if q["logic"][0] == 0:
                            answer_pos.append(q.get(q.get("answer")))
                            answer_neg.append(q.get(choice))
                        else:
                            answer_neg.append(q.get(q.get("answer")))
                            answer_pos.append(q.get(choice))
                        question_category_test.append((max(q["question_category"].iteritems(), key=operator.itemgetter(1))[0], q["logic"][0]))

    print("vocab full size: %d"%len(vocab))
    print("qlen 90 percentile %f, 75 perc %f, 50 %f"%(np.percentile(q_len,90),np.percentile(q_len,75),np.percentile(q_len,50)))
    print("alen 90 percentile %f, 75 perc %f, 50 %f"%(np.percentile(a_len,90),np.percentile(a_len,75),np.percentile(a_len,50)))
    print("vocab >1 size: %d"%len([k for k,v in vocab.most_common() if v > 1]))
    print("vocab >2 size: %d"%len([k for k,v in vocab.most_common() if v > 2]))
    for k,v in vocab.most_common():
        word2id[k] = len(word2id)

    FILTER_MOST_FREQ = 2
    num_words = len([k for k,v in vocab.most_common() if v > 1])

    for str in questions:
        ids = [start_char]
        for w in str:
            if word2id[w] > FILTER_MOST_FREQ:
                if word2id[w] < num_words:
                    ids.append(word2id[w]+index_from)
                else:
                    ids.append(oov_char)
        questions_idx.append(ids)

    for str in answer_pos:
        ids = [start_char]
        for w in str:
            if word2id[w] > FILTER_MOST_FREQ:
                if word2id[w] < num_words:
                    ids.append(word2id[w]+index_from)
                else:
                    ids.append(oov_char)
        answer_pos_idx.append(ids)

    for str in answer_neg:
        ids = [start_char]
        for w in str:
            if word2id[w] > FILTER_MOST_FREQ:
                if word2id[w] < num_words:
                    ids.append(word2id[w]+index_from)
                else:
                    ids.append(oov_char)
        answer_neg_idx.append(ids)
    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)

    train_until = train_until[1]
    questions_train, answer_pos_train, answer_neg_train = questions_idx[:train_until], answer_pos_idx[:train_until], answer_neg_idx[:train_until]
    questions_test, answer_pos_test, answer_neg_test = questions_idx[train_until:], answer_pos_idx[train_until:], answer_neg_idx[train_until:]

    print("loading word vec")
    word_embeddings = load_word_emb("./corpus_297k_200.txt", word2id, 200, num_words)
    return (questions_train, answer_pos_train, answer_neg_train), (questions_test, answer_pos_test, answer_neg_test), question_category_test, word_embeddings,word2id,num_words


def my_softmax(target, axis, epsilon=1e-12):
    max_axis = Lambda(lambda x: K.max(x, axis, keepdims=True))(target)
    t_m = Subtract()([target, max_axis])
    t_m = Lambda(lambda x:K.exp(x))(t_m)
    normalize = Lambda(lambda x:K.sum(x, axis, keepdims=True))(t_m)
    normalize = Lambda(lambda x:1. / (x+epsilon))(normalize)
    softmax = Multiply()([t_m, normalize])
    return softmax


class prediction_history(Callback):
    def __init__(self):
        self.predhis = []
    def on_epoch_end(self, epoch, logs={}):
        predictions = model.predict(x=[questions_test,answer_pos_test,answer_neg_test])
        print(epoch)
        print(predictions.shape)
        acc = evaluate_acc(predictions,question_category_test)
        if acc > 57:
            try:
                model.save('./my_model_%04d_%03d.h5'%(arguments["save_model"],epoch))
                print('done saving ./my_model_%04d_%03d.h5'%(arguments["save_model"],epoch))
            except:
                print("error in saving model ./my_model_%04d_%03d.h5"%(arguments["save_model"],epoch))


class NBatchLogger(Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            metrics_log = ''
            for k in self.params['metrics']:
                if k in logs:
                    val = logs[k]
                    if abs(val) > 1e-3:
                        metrics_log += ' - %s: %.4f' % (k, val)
                    else:
                        metrics_log += ' - %s: %.4e' % (k, val)
            print('{}/{} ... {}'.format(self.seen,self.params['samples'],metrics_log))


def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    #loss = -K.sigmoid(pos-neg)
    loss = K.maximum(2.0 + neg - pos, 0.0) # use if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

def build_qa_lstm_model_w2v(kwargs):
    # self attention
    R = kwargs["R"]
    question_input, answer_pos_input, answer_neg_input = Input(shape=(kwargs["maxlenq"],)), Input(shape=(kwargs["maxlena"],)), Input(shape=(kwargs["maxlena"],))
    emb = Embedding(kwargs["word_embeddings"].shape[0], kwargs["emb_dim"], mask_zero=False, weights=[kwargs["word_embeddings"]])
    question_emb, answer_pos_emb, answer_neg_emb = emb(question_input), emb(answer_pos_input), emb(answer_neg_input)
    #lstm_q, lstm_a = Bidirectional(LSTM(64)), Bidirectional(LSTM(64))
    lstm_q, lstm_a = Bidirectional(eval(kwargs["RNN"])(kwargs["lstm_dim"],return_sequences=True)), Bidirectional(eval(kwargs["RNN"])(kwargs["lstm_dim"],return_sequences=True))
    question_lstm = Dropout(kwargs["dropout"])(lstm_q(question_emb)) # batch x  maxlenq x 2lstm
    answer_pos_lstm = Dropout(kwargs["dropout"])(lstm_a(answer_pos_emb)) # batch x  maxlena x 2lstm
    answer_neg_lstm = Dropout(kwargs["dropout"])(lstm_a(answer_neg_emb)) # batch x  maxlena x 2lstm

    ws1a = Dense(kwargs["ws_factor"]*kwargs["lstm_dim"], activation='tanh')
    ws1q = Dense(kwargs["ws_factor"]*kwargs["lstm_dim"], activation='tanh')
    ws2q = Dense(kwargs["R"], activation='linear')
    ws2a = Dense(kwargs["R"], activation='linear')
    attq = ws2q(ws1q(question_lstm)) # batch x  maxlenq x R
    attap = ws2a(ws1a(answer_pos_lstm)) # batch x  maxlena x R
    attan = ws2a(ws1a(answer_neg_lstm)) # batch x  maxlena x R

    Aq = my_softmax(attq, axis=1) # batch x  maxlenq x R
    Ap = my_softmax(attap, axis=1) # batch x  maxlena x R
    An = my_softmax(attan, axis=1) # batch x  maxlena x R

    Mq = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]))([question_lstm, Aq]) # batch_size x 2lstm x R
    Map = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]))([answer_pos_lstm, Ap]) # batch_size x 2lstm x R
    Man = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]))([answer_neg_lstm, An]) # batch_size x 2lstm x R

    question_pos = Multiply()([Mq, Map]) # batch x x 2lstm x R
    question_neg = Multiply()([Mq, Man]) # batch x x 2lstm x R
    question_pos = Lambda(lambda x:K.sum(x, 1, keepdims=False))(question_pos) # batch x R
      = Lambda(lambda x:K.sum(x, 1, keepdims=False))(question_neg) # batch x R
    ds = Dense(1, activation='linear')
    question_pn = Concatenate()([ds(question_pos), ds(question_neg)])
    model = Model(inputs=[question_input,answer_pos_input,answer_neg_input], outputs=question_pn)
    return model


def evaluate_acc(predictions, question_category_test,print_prob=False,test_path=None):
    r,c = predictions.shape # (4xnum questions) x two prob.
    assert r % 4 == 0
    assert c == 2
    assert r == len(question_category_test)
    total = 0
    corr = 0
    total_cat = defaultdict(int)
    corr_cat = defaultdict(int)
    if print_prob:
        correct_answers = []
        with open(test_path) as f:
            lines = [line for line in f.readlines()]
            for line in lines:
                q = json.loads(line.strip())
                correct_answers.append(q.get('answer'))
            assert r/4 == len(correct_answers)
    for i in range(0,r,4):
        total += 1
        total_cat[question_category_test[i][0]] += 1
        correct = predictions[i,0]
        wrong1 = predictions[i,1]
        wrong2 = predictions[i+1,1]
        wrong3 = predictions[i+2,1]
        wrong4 = predictions[i+3,1]
        if question_category_test[i][1] == 0:
            comparator1 = correct > max([wrong1,wrong2,wrong3,wrong4])
            comparator2 = correct == max([wrong1,wrong2,wrong3,wrong4])
            print_prob_str = ["%f\t%d"%(predictions[i,0],1), "%f\t%d"%(predictions[i,1],0), "%f\t%d"%(predictions[i+1,1],0), "%f\t%d"%(predictions[i+2,1],0), "%f\t%d"%(predictions[i+3,1],0)]

        else:
            comparator1 = correct < min([wrong1,wrong2,wrong3,wrong4])
            comparator2 = correct == min([wrong1,wrong2,wrong3,wrong4])
            print_prob_str = ["%f\t%d"%(-predictions[i,0],1), "%f\t%d"%(-predictions[i,1],0), "%f\t%d"%(-predictions[i+1,1],0), "%f\t%d"%(-predictions[i+2,1],0), "%f\t%d"%(-predictions[i+3,1],0)]

        if print_prob:
            if correct_answers[i/4] == "B":
                tmp = print_prob_str[0]
                print_prob_str[0] = print_prob_str[1]
                print_prob_str[1] = tmp
            elif correct_answers[i/4] == "C":
                tmp = print_prob_str[0]
                print_prob_str[0] = print_prob_str[2]
                print_prob_str[2] = tmp
            elif correct_answers[i/4] == "D":
                tmp = print_prob_str[0]
                print_prob_str[0] = print_prob_str[3]
                print_prob_str[3] = tmp
            elif correct_answers[i/4] == "E":
                tmp = print_prob_str[0]
                print_prob_str[0] = print_prob_str[4]
                print_prob_str[4] = tmp
            for print_str in print_prob_str:
                print(print_str)

        if comparator1:
            corr += 1
            corr_cat[question_category_test[i][0]] += 1
        elif comparator2:
            candidates = [v for v in [wrong1,wrong2,wrong3,wrong4,correct] if v == correct]
            if random.random() < 1./len(candidates):
                corr += 1
                corr_cat[question_category_test[i][0]] += 1
    print("acc.:%.2f [%d/%d]"%(100.*corr/total, corr, total))
    for key,val in total_cat.items():
        print("%s acc. %.2f [%d/%d]" % (key, 100.*corr_cat[key]/val, corr_cat[key], val ))
    return 100.*corr/total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxlenq", help="", type=int)
    parser.add_argument("--maxlena", help="", type=int)
    parser.add_argument("--batch_size", help="", type=int)
    parser.add_argument("--GPUs", help="", type=int)
    parser.add_argument("--R", help="", type=int)
    parser.add_argument("--emb_dim", help="", type=int)
    parser.add_argument("--dropout", help="", type=float)
    parser.add_argument("--RNN", help="", type=str)
    parser.add_argument("--activation", help="", type=str)
    parser.add_argument("--lstm_dim", help="", type=int)
    parser.add_argument("--ws_factor", help="", type=int)
    parser.add_argument("--num_epochs", help="", type=int)
    parser.add_argument("--model_fn", help="", type=str)
    parser.add_argument("--data_file", help="", type=str)
    parser.add_argument("--test_file", help="", type=str)
    parser.add_argument("--valid_file", help="", type=str)
    parser.add_argument("--train_file", help="", type=str)
    parser.add_argument("--load_model", help="path to a saved model file", type=str)
    parser.add_argument("--save_model", help="a unique int id to identify different model config", type=int)
    args = parser.parse_args()
    arguments ={"maxlenq": args.maxlenq,
                "maxlena": args.maxlena,
                "batch_size": args.batch_size,
                "GPUs": args.GPUs,
                "R": args.R,
                "ws_factor": args.ws_factor,
                "emb_dim": args.emb_dim,
                "lstm_dim": args.lstm_dim,
                "num_epochs": args.num_epochs,
                "model_fn": args.model_fn,
                "dropout": args.dropout,
                "load_model": args.load_model,
                "save_model": args.save_model,
                "RNN": args.RNN,
                "activation": args.activation,
                "test_file": args.test_file,
                "valid_file": args.valid_file,
                "train_file": args.train_file,
                "data_file":args.data_file }

    print('Loading data...')

    if arguments["load_model"] is not None and len(arguments["load_model"]) > 0:
        print("loading existing model...")
        model = load_model(arguments["load_model"], custom_objects={"K": K,"tf": tf,"ranking_loss":ranking_loss})
        with open(arguments["data_file"]+".dict","rb") as f:
            num_words, word2id = pickle.load(f)
        questions_test,answer_pos_test,answer_neg_test,question_category_test = load_test(arguments["test_file"],word2id,num_words,maxlenq=arguments["maxlenq"], maxlena=arguments["maxlena"])
        questions_test = sequence.pad_sequences(questions_test, maxlen=arguments["maxlenq"])
        answer_pos_test = sequence.pad_sequences(answer_pos_test, maxlen=arguments["maxlena"])
        answer_neg_test = sequence.pad_sequences(answer_neg_test, maxlen=arguments["maxlena"])

        predictions = model.predict(x=[questions_test,answer_pos_test,answer_neg_test])
        acc = evaluate_acc(predictions,question_category_test,True,arguments["test_file"])
        sys.exit(0)


    if os.path.isfile(arguments["data_file"]):
        with open(arguments["data_file"],"rb") as f:
            (questions_train, answer_pos_train, answer_neg_train), (questions_test, answer_pos_test, answer_neg_test), question_category_test, word_embeddings,word2id,num_words = pickle.load(f)
    else:
        (questions_train, answer_pos_train, answer_neg_train), (questions_test, answer_pos_test, answer_neg_test), question_category_test, word_embeddings,word2id,num_words = load_data_with_2017_test(arguments["train_file"], arguments["valid_file"], arguments["test_file"], maxlenq=arguments["maxlenq"], maxlena=arguments["maxlena"])
        with open(arguments["data_file"],"wb") as f:
            pickle.dump(((questions_train, answer_pos_train, answer_neg_train), (questions_test, answer_pos_test, answer_neg_test), question_category_test, word_embeddings,word2id,num_words),f)
        with open(arguments["data_file"]+".dict","wb") as f:
            pickle.dump((num_words,word2id),f)

    arguments["word_embeddings"] = word_embeddings

    model = eval(arguments["model_fn"])(arguments)
    print(model.summary())

    if arguments["GPUs"] > 1:
        model = multi_gpu_model(model, gpus= arguments["GPUs"], cpu_merge=True)

    model.compile(loss=ranking_loss, optimizer=Adam())#Adagrad(lr=0.1, epsilon=1e-06))

    print(len(questions_train), 'train sequences')
    print(len(questions_test), 'test sequences')

    questions_train = sequence.pad_sequences(questions_train, maxlen=arguments["maxlenq"])
    answer_pos_train = sequence.pad_sequences(answer_pos_train, maxlen=arguments["maxlena"])
    answer_neg_train = sequence.pad_sequences(answer_neg_train, maxlen=arguments["maxlena"])
    questions_test = sequence.pad_sequences(questions_test, maxlen=arguments["maxlenq"])
    answer_pos_test = sequence.pad_sequences(answer_pos_test, maxlen=arguments["maxlena"])
    answer_neg_test = sequence.pad_sequences(answer_neg_test, maxlen=arguments["maxlena"])

    print('Pad sequences (samples x time)')


    print('Train...')
    print(questions_train[0])
    print(answer_pos_train[0])
    print(answer_neg_train[0])
    out_batch = NBatchLogger(1000)

    ph = prediction_history()
    model.fit(x=[questions_train,answer_pos_train,answer_neg_train],
              y= np.ones(len(questions_train)),
              validation_data=([questions_test,answer_pos_test,answer_neg_test],
                                 np.ones(len(questions_test))),
              batch_size=arguments["batch_size"], epochs=arguments["num_epochs"], verbose=2,shuffle=True, callbacks=[out_batch,ph])
    predictions = model.predict(x=[questions_test, answer_pos_test,answer_neg_test])
    evaluate_acc(predictions,question_category_test)
