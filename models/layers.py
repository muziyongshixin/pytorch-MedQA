#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from dataset.preprocess_data import PreprocessData
from utils.functions import masked_softmax, compute_mask, masked_flip
from IPython import embed
import numpy as np


class Word2VecEmbedding(torch.nn.Module):

    def __init__(self, dataset_h5_path, trainable=False):
        super(Word2VecEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embeddings, len_embedding, weights = self.load_glove_hdf5()
        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embeddings, embedding_dim=len_embedding,
                                                  _weight=weights)
        if not trainable:
            self.embedding_layer.weight.requires_grad = False

    def load_glove_hdf5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            f_meta_data = f['meta_data']
            id2vec = np.array(f_meta_data['id2vec'])  # only need 1.11s
            word_dict_size = f.attrs['word_dict_size']
            embedding_size = f.attrs['embedding_size']
        return int(word_dict_size), int(embedding_size), torch.from_numpy(id2vec)

    def forward(self, x):
        # mask 的作用就是生成一个和x维度一样的矩阵，将x中有单词的地方置位1，padding的地方置位0x
        mask = compute_mask(x, PreprocessData.padding_idx)
        tmp_emb = self.embedding_layer.forward(x)
        # 将embed的tensor变成(sequence_len,batch_size,embedding)的样子
        out_emb = tmp_emb
        return out_emb, mask


class delta_Embedding(torch.nn.Module):
    def __init__(self, n_embeddings, len_embedding, init_uniform, trainable=True):
        super(delta_Embedding, self).__init__()
        weights = torch.randn(n_embeddings, len_embedding)
        weights.uniform_(-1 * init_uniform, init_uniform)
        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embeddings, embedding_dim=len_embedding,
                                                  _weight=weights)
        if not trainable:
            self.embedding_layer.weight.requires_grad = False

    def forward(self, x):
        return self.embedding_layer.forward(x)


class My_gate_layer(torch.nn.Module):
    """使用conv操作或者FC实现gating 的功能"""

    def __init__(self, context_dim, gate_choose,dropout_p=0.0):
        super(My_gate_layer, self).__init__()
        self.gate_choose = gate_choose
        self.dropout_p=dropout_p
        if gate_choose == "CNN":
            self.gate_layer = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=context_dim, out_channels=1, kernel_size=3, stride=1, padding=1),
                # 16*1*100
                torch.nn.Sigmoid())
        elif gate_choose == "FC":
            self.gate_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=context_dim, out_features=1),
                torch.nn.Sigmoid()
            )
            self.init_parameters()

    def init_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize FC weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)

    def forward(self, x):
        if self.dropout_p != 0:
            x = torch.nn.functional.dropout(x, self.dropout_p, self.training)
        if self.gate_choose == "CNN":
            x = x.transpose(1, 2)  # size(16,256,100)
            output = self.gate_layer.forward(x)
            return output.transpose(1, 2)  # size=(16,100,1)
        elif self.gate_choose == "FC":
            batch_size = x.size()[0]
            seq_len = x.size()[1]
            context_dim = x.size()[2]
            x = x.view(batch_size * seq_len, context_dim)  # size(1600,256)
            output = self.gate_layer.forward(x)  # size(1600,1)
            output = output.view(batch_size, seq_len, 1)  # size(16,100,1)
            return output


class Noise_gate(torch.nn.Module):
    def __init__(self, context_dim, gate_choose,dropout_p=0.0):
        super(Noise_gate, self).__init__()
        self.gate_choose = gate_choose
        self.dropout_p=dropout_p
        if gate_choose == "CNN":
            self.gate_layer = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=context_dim, out_channels=1, kernel_size=3, stride=1, padding=1),
                # 16*1*100
                torch.nn.Sigmoid())
        elif gate_choose == "FC":
            self.gate_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=context_dim, out_features=1),
                torch.nn.Sigmoid())
            self.init_parameters()

    def init_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize FC weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)

    def forward(self, x):
        if self.dropout_p != 0:
            x = torch.nn.functional.dropout(x, self.dropout_p, self.training)
        if self.gate_choose == "CNN":
            x = x.transpose(1, 2)  # size(16,256,100)
            output = self.gate_layer.forward(x)
            output = output.transpose(1, 2)
            noise = torch.from_numpy(np.random.normal(loc=0.0, scale=0.1, size=output.size())).float().to(
                torch.device("cuda"))
            if self.training:
                return output + noise
            else:
                return output  # size=(16,100,1)
        elif self.gate_choose == "FC":
            batch_size = x.size()[0]
            seq_len = x.size()[1]
            context_dim = x.size()[2]
            x = x.view(batch_size * seq_len, context_dim)  # size(1600,256)
            output = self.gate_layer.forward(x)  # size(1600,1)
            output = output.view(batch_size, seq_len, 1)  # size(16,100,1)
            noise = torch.from_numpy(np.random.normal(loc=0.0, scale=0.1, size=output.size())).float().to(
                torch.device("cuda"))
            if self.training:  # eval 模式不添加noise
                return output + noise
            else:
                return output  # size=(16,100,1)


class MyLSTM(torch.nn.Module):
    """
    RNN with packed sequence and dropout
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn

    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p, enable_layer_norm=False):
        super(MyLSTM, self).__init__()
        self.mode = mode
        self.enable_layer_norm = enable_layer_norm

        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        dropout=dropout_p,
                                        bidirectional=bidirectional,
                                        batch_first=True
                                        )
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       dropout=dropout_p,
                                       bidirectional=bidirectional,
                                       batch_first=True
                                       )
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)

        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, v, mask):
        # layer normalization
        if self.enable_layer_norm:
            seq_len, batch, input_size = v.shape
            v = v.view(-1, input_size)
            v = self.layer_norm(v)
            v = v.view(seq_len, batch, input_size)
        # 当前的sentence padding到的长度
        sentence_len = v.size()[1]
        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        #  表示mask中，每个sentence1的个数  tensor([  58,  139,   75,  174,   64,   52,   52,  104,   49,   97, 119,   57,   50,  199,   99,  178], device='cuda:0')
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        # lengths_sort 表示排序后的ssentence长度，从长到短
        # =tensor([ 199,  178,  174,  139,  119,  104,   99,   97,   75,   64,  58,   57,   52,   52,   50,   49])
        # idx_sort 表示排序后的sentence在原先lengths中的下标
        # =tensor([ 13,  15,   3,   1,  10,   7,  14,   9,   2,   4,   0,  11,  5,   6,  12,   8],
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        # idx_unsort表示原始v中每个sentence的长度排第几
        # =tensor([ 10,   3,   8,   2,   9,  12,  13,   5,  15,   7,   4,  11,   14,   0,   6,   1]
        v_sort = v.index_select(0, idx_sort)
        # v_sort表示根据长度排序后的sentence向量，最长的放在前面，最短的放在后面
        # v_sort.size()=[16,100,200],v_sort[49,15]=[0,0,0,...,0,0,0],因为最后一个sentence的单词只有49个，所以49之后的词向量都是0向量

        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort, batch_first=True)
        # v_pack是一个PackedSequence的对象，其中包含data，和batch_sizes，两个子对象
        # data中存的计算的数据，是一个torch.Size([1566, 200])的矩阵，前16个是所有sentence的第一个单词，最后一个是最长的那个sentence的最后一个单词
        # batch_sizes中存储的是每一个计算步所需要计算的sentence数
        # 例如上面的数据的batch_size数据如下： 其中总共199个计算步，前49个计算步需要计算16个sentence，最后的十几个计算步只需要计算最长的那个sentence
        # tensor([16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #         16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #         16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #         16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #         16, 15, 14, 14, 12, 12, 12, 12, 12, 11, 10, 10,
        #         10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9,
        #         9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        #         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        #         8, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5,
        #         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4,
        #         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        #         4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3,
        #         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #         3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1])
        # v_dropout = self.dropout.forward(v_pack.data)

        # v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)
        # v_pack_dropout 进行过dropout后的数据

        o_pack, _ = self.hidden.forward(v_pack)  # 经过lstm层
        # o_pack_dropout.data是torch.Size([1566, 256]) 因为是双向LSTM 维度为128*2
        # embed()
        o_pack_dropout = self.dropout(o_pack.data)
        o_pack_dropout_p = torch.nn.utils.rnn.PackedSequence(o_pack_dropout, v_pack.batch_sizes)

        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout_p, batch_first=True, padding_value=0.0,
                                                      total_length=sentence_len)
        # o是经过lstm并解包之后的context向量 torch.Size([199, 16, 256])

        # unsorted o
        o_unsort = o.index_select(0, idx_unsort)  # Note that here first dim is seq_len
        # o_unsort表示恢复原始排序之后的向量，torch.Size([199, 16, 256])，因为最长的sentence为199，所以最后所有的context的长度为199
        # o_unsort[58,0,:]=[0,0,0...,0,0],因为原始的第一个sentence长度58，所以其第59个单词过lstm之后的encode还是0向量

        return o_unsort


class qa_matching_layer(torch.nn.Module):
    def __init__(self,R):
        super(qa_matching_layer,self).__init__()
        hidden_mode='LSTM'
        word_embedding_size=200
        hidden_size=150
        encoder_word_layers=1
        encoder_bidirection=True
        dropout_p=0.3
        ws_factor=1
        self.R=R

        self.question_context_layer=MyLSTM(mode=hidden_mode,
                                       input_size=word_embedding_size,
                                       hidden_size=hidden_size,
                                       num_layers=encoder_word_layers,
                                       bidirectional=encoder_bidirection,
                                       dropout_p=dropout_p)
        self.answer_context_layer=MyLSTM(mode=hidden_mode,
                                       input_size=word_embedding_size,
                                       hidden_size=hidden_size,
                                       num_layers=encoder_word_layers,
                                       bidirectional=encoder_bidirection,
                                       dropout_p=dropout_p)
        self.question_attention_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size * 2, out_features=ws_factor * hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=ws_factor * hidden_size, out_features=self.R)
        )
        self.answer_attention_layer=torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size*2,out_features=ws_factor*hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=ws_factor*hidden_size,out_features=self.R)
       )

        self.output_layer=torch.nn.Linear(in_features=R,out_features=1)
    def forward(self, questions_vec,question_mask,answers_vec,answers_mask):
        # questions 20,60
        # answers 20,12
        batch_size=questions_vec.size()[0]

        questions_encode=self.question_context_layer(questions_vec,question_mask) #batch 60 300
        answers_encode=self.answer_context_layer(answers_vec,answers_mask) #batch 12 300

        questions_attention=self.question_attention_layer(questions_encode)  #batch 60 10
        answers_attention=self.answer_attention_layer(answers_encode) #batch 12 10

        questions_attention=torch.nn.functional.softmax(questions_attention,dim=1)#batch 60 10
        answers_attention=torch.nn.functional.softmax(answers_attention,dim=1)#batch 12 10

        questions_match=torch.bmm(questions_encode.transpose(1,2),questions_attention) #batch 300 10
        answers_match=torch.bmm(answers_encode.transpose(1,2),answers_attention) #batch 300 10

        match_feature=questions_match*answers_match #batch 300 10
        sum_match_feature=torch.sum(match_feature,dim=1) #batch  10
        return sum_match_feature

class GloveEmbedding(torch.nn.Module):
    """
    Glove Embedding Layer, also compute the mask of padding index
    Args:
        - dataset_h5_path: glove embedding file path
    Inputs:
        **input** (batch, seq_len): sequence with word index
    Outputs
        **output** (seq_len, batch, embedding_size): tensor that change word index to word embeddings
        **mask** (batch, seq_len): tensor that show which index is padding
    """

    def __init__(self, dataset_h5_path, trainable=False):
        super(GloveEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embeddings, len_embedding, weights = self.load_glove_hdf5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embeddings, embedding_dim=len_embedding,
                                                  _weight=weights)
        if not trainable:
            self.embedding_layer.weight.requires_grad = False

    def load_glove_hdf5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            f_meta_data = f['meta_data']
            id2vec = np.array(f_meta_data['id2vec'])  # only need 1.11s
            word_dict_size = f.attrs['word_dict_size']
            embedding_size = f.attrs['embedding_size']

        return int(word_dict_size), int(embedding_size), torch.from_numpy(id2vec)

    def forward(self, x):
        mask = compute_mask(x, PreprocessData.padding_idx)

        tmp_emb = self.embedding_layer.forward(x)
        out_emb = tmp_emb.transpose(0, 1)

        return out_emb, mask


class CharEmbedding(torch.nn.Module):
    """
    Char Embedding Layer, random weight
    Args:
        - dataset_h5_path: char embedding file path
    Inputs:
        **input** (batch, seq_len, word_len): word sequence with char index
    Outputs
        **output** (batch, seq_len, word_len, embedding_size): tensor contain char embeddings
        **mask** (batch, seq_len, word_len)
    """

    def __init__(self, dataset_h5_path, embedding_size, trainable=False):
        super(CharEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embedding = self.load_dataset_h5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embedding, embedding_dim=embedding_size,
                                                  padding_idx=0)  # 0 号char输出的向量是0向量

        # Note that cannot directly assign value. When in predict, it's always False.
        if not trainable:
            self.embedding_layer.weight.requires_grad = False

    def load_dataset_h5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            word_dict_size = f.attrs['char_dict_size']

        return int(word_dict_size)

    def forward(self, x):
        batch_size, seq_len, word_len = x.shape
        x = x.view(-1, word_len)

        mask = compute_mask(x, 0)  # char-level padding idx is zero
        x_emb = self.embedding_layer.forward(x)
        x_emb = x_emb.view(batch_size, seq_len, word_len, -1)
        mask = mask.view(batch_size, seq_len, word_len)

        return x_emb, mask


class CharEncoder(torch.nn.Module):
    """
    char-level encoder with MyRNNBase used
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p):
        super(CharEncoder, self).__init__()

        self.encoder = MyRNNBase(mode=mode,
                                 input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=bidirectional,
                                 dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size)
        x = x.transpose(0, 1)
        char_mask = char_mask.view(-1, word_len)

        _, x_encode = self.encoder.forward(x, char_mask)  # (batch*seq_len, hidden_size)
        x_encode = x_encode.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        x_encode = x_encode * word_mask.unsqueeze(-1)

        return x_encode.transpose(0, 1)


class CharCNN(torch.nn.Module):
    """
    Char-level CNN
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, filters_size, filters_num, dropout_p):
        super(CharCNN, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.cnns = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, fn, (fw, emb_size)) for fw, fn in zip(filters_size, filters_num)])

    def forward(self, x, char_mask, word_mask):
        x = self.dropout(x)

        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size).unsqueeze(1)  # (N, 1, word_len, embedding_size)

        x = [F.relu(cnn(x)).squeeze(-1) for cnn in self.cnns]  # (N, Cout, word_len - fw + 1) * fn
        x = [torch.max(cx, 2)[0] for cx in x]  # (N, Cout) * fn
        x = torch.cat(x, dim=1)  # (N, hidden_size)

        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        x = x * word_mask.unsqueeze(-1)

        return x.transpose(0, 1)


class Highway(torch.nn.Module):
    def __init__(self, in_size, n_layers, dropout_p):
        super(Highway, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.normal_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            normal_layer_ret = F.relu(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x
        return x


class CharCNNEncoder(torch.nn.Module):
    """
    char-level cnn encoder with highway networks
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, hidden_size, filters_size, filters_num, dropout_p, enable_highway=True):
        super(CharCNNEncoder, self).__init__()
        self.enable_highway = enable_highway
        self.hidden_size = hidden_size

        self.cnn = CharCNN(emb_size=emb_size,
                           filters_size=filters_size,
                           filters_num=filters_num,
                           dropout_p=dropout_p)

        if enable_highway:
            self.highway = Highway(in_size=hidden_size,
                                   n_layers=2,
                                   dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        o = self.cnn(x, char_mask, word_mask)

        assert o.shape[2] == self.hidden_size
        if self.enable_highway:
            o = self.highway(o)

        return o


class MatchRNNAttention(torch.nn.Module):
    r"""
    attention mechanism in match-rnn
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hpi(batch, input_size): a context word encoded
        Hq(question_len, batch, input_size): whole question encoded
        Hr_last(batch, hidden_size): last lstm hidden output

    Outputs:
        alpha(batch, question_len): attention vector
    """

    def __init__(self, hp_input_size, hq_input_size, hidden_size):
        super(MatchRNNAttention, self).__init__()

        self.linear_wq = torch.nn.Linear(hq_input_size, hidden_size)
        self.linear_wp = torch.nn.Linear(hp_input_size, hidden_size)
        self.linear_wr = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wg = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hpi, Hq, Hr_last, Hq_mask):
        wq_hq = self.linear_wq(Hq)  # (question_len, batch, hidden_size)
        wp_hp = self.linear_wp(Hpi).unsqueeze(0)  # (1, batch, hidden_size)
        wr_hr = self.linear_wr(Hr_last).unsqueeze(0)  # (1, batch, hidden_size)
        G = F.tanh(wq_hq + wp_hp + wr_hr)  # (question_len, batch, hidden_size), auto broadcast
        wg_g = self.linear_wg(G) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, question_len)
        alpha = masked_softmax(wg_g, m=Hq_mask, dim=1)  # (batch, question_len)
        return alpha


class UniMatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism, one direction, using LSTM cell
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded

    Outputs:
        Hr(context_len, batch, hidden_size): question-aware context representation
        alpha(batch, question_len, context_len): used for visual show
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, gated_attention, enable_layer_norm):
        super(UniMatchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gated_attention = gated_attention
        self.enable_layer_norm = enable_layer_norm
        rnn_in_size = hp_input_size + hq_input_size

        self.attention = MatchRNNAttention(hp_input_size, hq_input_size, hidden_size)

        if self.gated_attention:
            self.gated_linear = torch.nn.Linear(rnn_in_size, rnn_in_size)

        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(rnn_in_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size=rnn_in_size, hidden_size=hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size=rnn_in_size, hidden_size=hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hp, Hq, Hq_mask):
        batch_size = Hp.shape[1]
        context_len = Hp.shape[0]

        # init hidden with the same type of input data
        h_0 = Hq.new_zeros(batch_size, self.hidden_size)  # Hq 100,16,256， h_0 16,256
        hidden = [(h_0, h_0)] if self.mode == 'LSTM' else [h_0]
        vis_para = {}
        vis_alpha = []
        vis_gated = []

        for t in range(context_len):
            cur_hp = Hp[t, ...]  # (batch, input_size) 16,256
            attention_input = hidden[t][0] if self.mode == 'LSTM' else hidden[t]

            alpha = self.attention.forward(cur_hp, Hq, attention_input, Hq_mask)  # (batch, question_len) 16,100
            vis_alpha.append(alpha)

            question_alpha = torch.bmm(alpha.unsqueeze(1), Hq.transpose(0, 1)).squeeze(
                1)  # 16,1,100 * 16,100,256 (batch, input_size)
            cur_z = torch.cat([cur_hp, question_alpha], dim=1)  # (batch, rnn_in_size)

            # gated
            if self.gated_attention:
                gate = F.sigmoid(self.gated_linear.forward(cur_z))
                vis_gated.append(gate.squeeze(-1))
                cur_z = gate * cur_z

            # layer normalization
            if self.enable_layer_norm:
                cur_z = self.layer_norm(cur_z)  # (batch, rnn_in_size)

            cur_hidden = self.hidden_cell.forward(cur_z, hidden[t])  # (batch, hidden_size), when lstm output tuple
            hidden.append(cur_hidden)

        vis_para['gated'] = torch.stack(vis_gated, dim=-1)  # (batch, context_len)
        vis_para['alpha'] = torch.stack(vis_alpha, dim=2)  # (batch, question_len, context_len)

        hidden_state = list(map(lambda x: x[0], hidden)) if self.mode == 'LSTM' else hidden
        result = torch.stack(hidden_state[1:], dim=0)  # (context_len, batch, hidden_size)
        return result, vis_para


class MatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
        - bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        - gated_attention: If ``True``, gated attention used, see more on R-NET

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded
        Hp_mask(batch, context_len): each context valued length without padding values
        Hq_mask(batch, question_len): each question valued length without padding values

    Outputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, bidirectional, gated_attention,
                 dropout_p, enable_layer_norm):
        super(MatchRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 1 if bidirectional else 2

        self.left_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                          enable_layer_norm)
        if bidirectional:
            self.right_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                               enable_layer_norm)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hp, Hp_mask, Hq, Hq_mask):
        Hp = self.dropout(Hp)
        Hq = self.dropout(Hq)
        # Hp 200,16,256   Hq 100,16,256
        left_hidden, left_para = self.left_match_rnn.forward(Hp, Hq,
                                                             Hq_mask)  # left_hidden (context_len, batch, hidden_size)
        rtn_hidden = left_hidden
        rtn_para = {'left': left_para}

        if self.bidirectional:
            Hp_inv = masked_flip(Hp, Hp_mask, flip_dim=0)
            right_hidden_inv, right_para_inv = self.right_match_rnn.forward(Hp_inv, Hq, Hq_mask)

            # flip back to normal sequence
            right_alpha_inv = right_para_inv['alpha']
            right_alpha_inv = right_alpha_inv.transpose(0, 1)  # make sure right flip
            right_alpha = masked_flip(right_alpha_inv, Hp_mask, flip_dim=2)
            right_alpha = right_alpha.transpose(0, 1)

            right_gated_inv = right_para_inv['gated']
            right_gated_inv = right_gated_inv.transpose(0, 1)
            right_gated = masked_flip(right_gated_inv, Hp_mask, flip_dim=2)
            right_gated = right_gated.transpose(0, 1)

            right_hidden = masked_flip(right_hidden_inv, Hp_mask, flip_dim=0)

            rtn_para['right'] = {'alpha': right_alpha, 'gated': right_gated}
            rtn_hidden = torch.cat((left_hidden, right_hidden), dim=2)

        real_rtn_hidden = Hp_mask.transpose(0, 1).unsqueeze(2) * rtn_hidden
        last_hidden = rtn_hidden[-1, :]

        return real_rtn_hidden, last_hidden, rtn_para


class PointerAttention(torch.nn.Module):
    r"""
    attention mechanism in pointer network
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer

    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        Hk_last(batch, hidden_size): the last hidden output of previous time

    Outputs:
        beta(batch, context_len): question-aware context representation
    """

    def __init__(self, input_size, hidden_size):
        super(PointerAttention, self).__init__()

        self.linear_wr = torch.nn.Linear(input_size, hidden_size)
        self.linear_wa = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wf = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hr, Hr_mask, Hk_pre):
        wr_hr = self.linear_wr(Hr)  # (context_len, batch, hidden_size)
        wa_ha = self.linear_wa(Hk_pre).unsqueeze(0)  # (1, batch, hidden_size)
        f = F.tanh(wr_hr + wa_ha)  # (context_len, batch, hidden_size)

        beta_tmp = self.linear_wf(f) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, context_len)

        beta = masked_softmax(beta_tmp, m=Hr_mask, dim=1)
        return beta


class SeqPointer(torch.nn.Module):
    r"""
    Sequence Pointer Net that output every possible answer position in context
    Args:

    Inputs:
        Hr: question-aware context representation
    Outputs:
        **output** every answer index possibility position in context, no fixed length
    """

    def __init__(self):
        super(SeqPointer, self).__init__()

    def forward(self, *input):
        return NotImplementedError


class UniBoundaryPointer(torch.nn.Module):
    r"""
    Unidirectional Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - enable_layer_norm: Whether use layer normalization

    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0(batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
        **hidden** (batch, hidden_size), [(batch, hidden_size)]: rnn last state
    """
    answer_len = 2

    def __init__(self, mode, input_size, hidden_size, enable_layer_norm):
        super(UniBoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enable_layer_norm = enable_layer_norm

        self.attention = PointerAttention(input_size, hidden_size)

        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size, hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size, hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hr, Hr_mask, h_0=None):
        if h_0 is None:
            batch_size = Hr.shape[1]
            h_0 = Hr.new_zeros(batch_size, self.hidden_size)

        hidden = (h_0, h_0) if self.mode == 'LSTM' and isinstance(h_0, torch.Tensor) else h_0
        beta_out = []

        for t in range(self.answer_len):
            attention_input = hidden[0] if self.mode == 'LSTM' else hidden
            beta = self.attention.forward(Hr, Hr_mask, attention_input)  # (batch, context_len)
            beta_out.append(beta)

            context_beta = torch.bmm(beta.unsqueeze(1), Hr.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)

            if self.enable_layer_norm:
                context_beta = self.layer_norm(context_beta)  # (batch, input_size)

            hidden = self.hidden_cell.forward(context_beta, hidden)  # (batch, hidden_size), (batch, hidden_size)

        result = torch.stack(beta_out, dim=0)
        return result, hidden


class BoundaryPointer(torch.nn.Module):
    r"""
    Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - bidirectional: Bidirectional or Unidirectional
        - dropout_p: Dropout probability
        - enable_layer_norm: Whether use layer normalization

    Inputs:
        Hr (context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0 (batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm):
        super(BoundaryPointer, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.left_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)
        if bidirectional:
            self.right_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr = self.dropout.forward(Hr)

        left_beta, _ = self.left_ptr_rnn.forward(Hr, Hr_mask, h_0)
        rtn_beta = left_beta
        if self.bidirectional:
            right_beta_inv, _ = self.right_ptr_rnn.forward(Hr, Hr_mask, h_0)
            right_beta = right_beta_inv[[1, 0], :]

            rtn_beta = (left_beta + right_beta) / 2

        # todo: unexplainable
        new_mask = torch.neg((Hr_mask - 1) * 1e-6)  # mask replace zeros with 1e-6, make sure no gradient explosion
        rtn_beta = rtn_beta + new_mask.unsqueeze(0)

        return rtn_beta


class MultiHopBdPointer(torch.nn.Module):
    r"""
    Boundary Pointer Net with Multi-Hops that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - num_hops: Number of max hops
        - dropout_p: Dropout probability
        - enable_layer_norm: Whether use layer normalization

    Inputs:
        Hr (context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0 (batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
    """

    def __init__(self, mode, input_size, hidden_size, num_hops, dropout_p, enable_layer_norm):
        super(MultiHopBdPointer, self).__init__()
        self.hidden_size = hidden_size
        self.num_hops = num_hops

        self.ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr = self.dropout.forward(Hr)

        beta_last = None
        for i in range(self.num_hops):
            beta, h_0 = self.ptr_rnn.forward(Hr, Hr_mask, h_0)
            if beta_last is not None and (beta_last == beta).sum().item() == beta.shape[0]:  # beta not changed
                break

            beta_last = beta

        new_mask = torch.neg((Hr_mask - 1) * 1e-6)  # mask replace zeros with 1e-6, make sure no gradient explosion
        rtn_beta = beta + new_mask.unsqueeze(0)

        return rtn_beta


class MyRNNBase(torch.nn.Module):
    """
    RNN with packed sequence and dropout
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn

    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p, enable_layer_norm=False):
        super(MyRNNBase, self).__init__()
        self.mode = mode
        self.enable_layer_norm = enable_layer_norm

        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        dropout=dropout_p,
                                        bidirectional=bidirectional
                                        )
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       dropout=dropout_p,
                                       bidirectional=bidirectional
                                       )
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)

        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, v, mask):
        # layer normalization
        if self.enable_layer_norm:
            seq_len, batch, input_size = v.shape
            v = v.view(-1, input_size)
            v = self.layer_norm(v)
            v = v.view(seq_len, batch, input_size)

        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        #  表示mask中，每个sentence1的个数  tensor([  58,  139,   75,  174,   64,   52,   52,  104,   49,   97, 119,   57,   50,  199,   99,  178], device='cuda:0')

        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        # lengths_sort 表示排序后的ssentence长度，从长到短
        # =tensor([ 199,  178,  174,  139,  119,  104,   99,   97,   75,   64,  58,   57,   52,   52,   50,   49])
        # idx_sort 表示排序后的sentence在原先lengths中的下标
        # =tensor([ 13,  15,   3,   1,  10,   7,  14,   9,   2,   4,   0,  11,  5,   6,  12,   8],
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        # idx_unsort表示原始v中每个sentence的长度排第几
        # =tensor([ 10,   3,   8,   2,   9,  12,  13,   5,  15,   7,   4,  11,   14,   0,   6,   1]
        # embed()
        v_sort = v.index_select(1, idx_sort)
        # v_sort表示根据长度排序后的sentence向量，最长的放在前面，最短的放在后面
        # v_sort.size()=[200,16,200],v_sort[49,15]=[0,0,0,...,0,0,0],因为最后一个sentence的单词只有49个，所以49之后的词向量都是0向量

        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        # v_pack是一个PackedSequence的对象，其中包含data，和batch_sizes，两个子对象
        # data中存的计算的数据，是一个torch.Size([1566, 200])的矩阵，前16个是所有sentence的第一个单词，最后一个是最长的那个sentence的最后一个单词
        # batch_sizes中存储的是每一个计算步所需要计算的sentence数
        # 例如上面的数据的batch_size数据如下： 其中总共199个计算步，前49个计算步需要计算16个sentence，最后的十几个计算步只需要计算最长的那个sentence
        # tensor([16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #         16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #         16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #         16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        #         16, 15, 14, 14, 12, 12, 12, 12, 12, 11, 10, 10,
        #         10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9,
        #         9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        #         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        #         8, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5,
        #         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4,
        #         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        #         4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3,
        #         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #         3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1])
        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)
        # v_pack_dropout 进行过dropout后的数据

        o_pack_dropout, _ = self.hidden.forward(v_pack_dropout)  # 经过lstm层
        # o_pack_dropout.data是torch.Size([1566, 256]) 因为是双向LSTM 维度为128*2

        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)
        # o是经过lstm并解包之后的context向量 torch.Size([199, 16, 256])

        # unsorted o
        o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len
        # o_unsort表示恢复原始排序之后的向量，torch.Size([199, 16, 256])，因为最长的sentence为199，所以最后所有的context的长度为199
        # o_unsort[58,0,:]=[0,0,0...,0,0],因为原始的第一个sentence长度58，所以其第59个单词过lstm之后的encode还是0向量

        # get the last time state
        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)
        o_last = o_unsort.gather(0, len_idx)
        # o_last表示最后一个单词的之后的hidden向量 torch.Size([1, 16, 256])

        return o_unsort, o_last


class AttentionPooling(torch.nn.Module):
    """
    Attention-Pooling for pointer net init hidden state generate.
    Equal to Self-Attention + MLP
    Modified from r-net.
    Args:
        input_size: The number of expected features in the input uq
        output_size: The number of expected features in the output rq_o

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output
        - **output** (batch, output_size): tensor containing the output features
    """

    def __init__(self, input_size, output_size):
        super(AttentionPooling, self).__init__()

        self.linear_u = torch.nn.Linear(input_size, output_size)
        self.linear_t = torch.nn.Linear(output_size, 1)
        self.linear_o = torch.nn.Linear(input_size, output_size)

    def forward(self, uq, mask):
        q_tanh = F.tanh(self.linear_u(uq))
        q_s = self.linear_t(q_tanh) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, seq_len)

        alpha = masked_softmax(q_s, mask, dim=1)  # (batch, seq_len)
        rq = torch.bmm(alpha.unsqueeze(1), uq.transpose(0, 1)) \
            .squeeze(1)  # (batch, input_size)

        rq_o = F.tanh(self.linear_o(rq))  # (batch, output_size)
        return rq_o


class SelfAttentionGated(torch.nn.Module):
    """
    Self-Attention Gated layer, it`s not weighted sum in the last, but just weighted
    Args:
        input_size: The number of expected features in the input x

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output
        - **output** (seq_len, batch, input_size): gated output tensor
    """

    def __init__(self, input_size):
        super(SelfAttentionGated, self).__init__()

        self.linear_g = torch.nn.Linear(input_size, input_size)
        self.linear_t = torch.nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        g_tanh = F.tanh(self.linear_g(x))
        gt = self.linear_t.forward(g_tanh) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, seq_len)

        gt_prop = masked_softmax(gt, x_mask, dim=1)
        gt_prop = gt_prop.transpose(0, 1).unsqueeze(2)  # (seq_len, batch, 1)
        x_gt = x * gt_prop

        return x_gt


class SelfGated(torch.nn.Module):
    def __init__(self, input_size):
        super(SelfGated, self).__init__()

        self.linear_g = torch.nn.Linear(input_size, input_size)

    def forward(self, x):
        x_l = self.linear_g(x)  # (seq_len, batch, input_size)
        x_gt = F.sigmoid(x_l)

        x = x * x_gt

        return x
