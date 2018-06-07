#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liyz'

import torch
from models.layers import *
from utils.functions import answer_search, multi_scale_ptr
from IPython import  embed
from utils.functions import masked_softmax, compute_mask, masked_flip



class SimpleSeaReader(torch.nn.Module):
    """
    match-lstm+ model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        context_char: (batch, seq_len, word_len)
        question_char: (batch, seq_len, word_len)

    Outputs:
        ans_range_prop: (batch, 2, context_len)
        ans_range: (batch, 2)
        vis_alpha: to show on visdom
    """

    def __init__(self, dataset_h5_path,device):
        super(SimpleSeaReader, self).__init__()

        self.device=device
        # set config
        hidden_size = 128
        hidden_mode = 'LSTM'
        dropout_p = 0.2
        # emb_dropout_p = 0.1
        enable_layer_norm = False

        word_embedding_size = 200
        encoder_word_layers = 1

        # char_embedding_size = 64
        # encoder_char_layers = 1

        encoder_bidirection = True
        encoder_direction_num = 2 if encoder_bidirection else 1

        match_lstm_bidirection = True
        match_rnn_direction_num = 2 if match_lstm_bidirection else 1

        ptr_bidirection = False
        self.enable_search = True

        # construct model
        self.embedding = Word2VecEmbedding(dataset_h5_path=dataset_h5_path)
        # self.char_embedding = CharEmbedding(dataset_h5_path=dataset_h5_path,
        #                                     embedding_size=char_embedding_size,
        #                                     trainable=True)

        # self.char_encoder = CharEncoder(mode=hidden_mode,
        #                                 input_size=char_embedding_size,
        #                                 hidden_size=hidden_size,
        #                                 num_layers=encoder_char_layers,
        #                                 bidirectional=encoder_bidirection,
        #                                 dropout_p=emb_dropout_p)
        self.context_layer = MyRNNBase(mode=hidden_mode,
                                 input_size=word_embedding_size,
                                 hidden_size=hidden_size,
                                 num_layers=encoder_word_layers,
                                 bidirectional=encoder_bidirection,
                                 dropout_p=dropout_p)

        self.decision_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=100*200*10, out_features=1000, bias=True),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(1000),
            torch.nn.Linear(in_features=1000, out_features=500, bias=True),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(500),
            torch.nn.Linear(in_features=500,out_features=2,bias=True)
        )

    def forward(self, contents, question_ans, contents_char=None, question_ans_char=None):
        # assert contents_char is not None and question_ans_char is not None
        batch_size=question_ans.size()[0]
        max_content_len=contents.size()[2]
        max_question_len=question_ans.size()[1]
        contents_num=contents.size()[1]

        # word-level embedding: (seq_len, batch, embedding_size)
        content_vec=[]
        content_mask=[]
        question_vec, question_mask = self.embedding.forward(question_ans)
        for i in range(contents_num):
            cur_content=contents[:,i,:]
            cur_content_vec, cur_content_mask = self.embedding.forward(cur_content)
            content_vec.append(cur_content_vec)
            content_mask.append(cur_content_mask)
        # char-level embedding: (seq_len, batch, char_embedding_size)
        # context_emb_char, context_char_mask = self.char_embedding.forward(context_char)
        # question_emb_char, question_char_mask = self.char_embedding.forward(question_char)
        question_encode, _ = self.context_layer.forward(question_vec, question_mask) #size=(cur_batch_max_questionans_len, batch, 256)
        content_encode=[]  # word-level encode: (seq_len, batch, hidden_size)
        for i in range(contents_num):
            cur_content_vec=content_vec[i]
            cur_content_mask=content_mask[i]
            cur_content_encode, _ = self.context_layer.forward(cur_content_vec, cur_content_mask) #size=(cur_batch_max_content_len, batch, 256)
            content_encode.append(cur_content_encode)

        # 将所有的content编码后统一到相同的长度 200，所有的question编码后统一到相同的长度100
        same_sized_content_encode=[]
        for i in range(contents_num):
            cur_content_encode=content_encode[i]
            cur_content_encode=self.full_matrix_to_specify_size(cur_content_encode,[max_content_len,batch_size,cur_content_encode.size()[2]]) # size=(200,16,256)
            same_sized_content_encode.append(cur_content_encode)
        same_sized_question_encode=self.full_matrix_to_specify_size(question_encode,[max_question_len,batch_size,question_encode.size()[-1]]) # size=(100,16,256)

        # Matching Matrix computing, question 和每一个content都要计算matching matrix
        Matching_matrix=[]
        for i in range(contents_num):
            cur_content_encode=same_sized_content_encode[i]
            cur_Matching_matrix=self.compute_matching_matrix(same_sized_question_encode,cur_content_encode)  # (batch, question_len , content_len) eg(16,100,200)
            Matching_matrix.append(cur_Matching_matrix)


        Matching_matrix_feature=torch.stack(Matching_matrix).transpose(0,1).contiguous().view(batch_size,-1)# (batch, 100*200*10)

        output=self.decision_layer.forward(Matching_matrix_feature) #size=(16,2)
        return output

        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        # # # char-level encode: (seq_len, batch, hidden_size)
        # # context_vec_char = self.char_encoder.forward(context_emb_char, context_char_mask, context_mask)
        # # question_vec_char = self.char_encoder.forward(question_emb_char, question_char_mask, question_mask)
        #
        # # context_encode = torch.cat((context_encode, context_vec_char), dim=-1)
        # # question_encode = torch.cat((question_encode, question_vec_char), dim=-1)
        #
        # # match lstm: (seq_len, batch, hidden_size)
        # qt_aware_ct, qt_aware_last_hidden, match_para = self.match_rnn.forward(content_encode, content_mask,
        #                                                                        question_encode, question_mask)
        # vis_param = {'match': match_para}
        #
        # # birnn after self match: (seq_len, batch, hidden_size)
        # qt_aware_ct_ag, _ = self.birnn_after_self.forward(qt_aware_ct, content_mask)
        #
        # # pointer net init hidden: (batch, hidden_size)
        # ptr_net_hidden = F.tanh(self.init_ptr_hidden.forward(qt_aware_last_hidden))
        #
        # # pointer net: (answer_len, batch, context_len)
        # ans_range_prop = self.pointer_net.forward(qt_aware_ct_ag, context_mask, ptr_net_hidden)
        # ans_range_prop = ans_range_prop.transpose(0, 1)
        #
        # # answer range
        # if not self.training and self.enable_search:
        #     ans_range = answer_search(ans_range_prop, context_mask)
        # else:
        #     _, ans_range = torch.max(ans_range_prop, dim=2)
        #
        # return ans_range_prop, ans_range, vis_param

    def load_glove_hdf5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            f_meta_data = f['meta_data']
            id2vec = np.array(f_meta_data['id2vec'])  # only need 1.11s
            word_dict_size = f.attrs['word_dict_size']
            embedding_size = f.attrs['embedding_size']

        return int(word_dict_size), int(embedding_size), torch.from_numpy(id2vec)

    def compute_matching_matrix(self,question_encode,content_encode):
        question_encode_trans = question_encode.transpose(0, 1)  # (batch, seq_len, embedding_size)
        content_encode_trans = content_encode.transpose(0, 1)  # (batch, seq_len, embedding_size)
        content_encode_trans = content_encode_trans.transpose(1, 2)  # (batch, embedding_size, seq_len)
        Matching_matrix = torch.bmm(question_encode_trans, content_encode_trans)  # (batch, question_len , content_len)
        return Matching_matrix

    def compute_cross_document_attention(self,content_m,content_n):
        content_n_trans = content_n.transpose(1, 2)  # (batch, 512, 200)
        Matching_matrix = torch.bmm(content_m, content_n_trans)  # (batch, question_len , content_len)
        return Matching_matrix

    def compute_RnQ(self,an_matrix,content_encode):
        content_encode_trans=content_encode.transpose(0,1) ## (batch, content_len, embedding_size)
        RnQ=torch.bmm(an_matrix,content_encode_trans)
        return RnQ

    def compute_RnD(self,bn_matrix,question_encode):
        bn_matrix_trans=bn_matrix.transpose(1,2)             #size=(batch,content_len,question_len)
        question_encode_trans=question_encode.transpose(0,1) # (batch, question_len, embedding_size)
        RnD=torch.bmm(bn_matrix_trans,question_encode_trans)
        return RnD

    # 将矩阵填充到指定的大小，填充的部分为0
    def full_matrix_to_specify_size(self,input_matrix,output_size):
        new_matrix=torch.zeros(output_size).to(self.device)
        new_matrix[0:input_matrix.size()[0],0:input_matrix.size()[1],0:input_matrix.size()[2]]=input_matrix[:,:,:]
        return new_matrix
