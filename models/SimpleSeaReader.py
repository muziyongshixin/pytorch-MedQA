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
        self.embedding = Word2VecEmbedding(dataset_h5_path=dataset_h5_path,trainable= True)
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


        self.content_reasoning_layer=MyRNNBase(mode=hidden_mode,
                                               input_size=hidden_size*4,
                                               hidden_size=hidden_size,
                                               num_layers=1,
                                               bidirectional=True,
                                               dropout_p =0.2)

        self.question_reasoning_layer = MyRNNBase(mode=hidden_mode,
                                                 input_size=hidden_size * 2,
                                                 hidden_size=hidden_size,
                                                 num_layers=1,
                                                 bidirectional=True,
                                                 dropout_p=0.2)

        self.matching_feature_reasoning_layer = MyRNNBase(mode=hidden_mode,
                                                 input_size=200,
                                                 hidden_size=hidden_size,
                                                 num_layers=1,
                                                 bidirectional=True,
                                                 dropout_p=0.2)


        self.decision_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=6552, out_features=1000, bias=True),
            torch.nn.Dropout(0.2),  # drop 50% of the neuron
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(1000),
            torch.nn.Linear(in_features=1000, out_features=500, bias=True),
            torch.nn.Dropout(0.2),  # drop 50% of the neuron
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(500),
            torch.nn.Linear(in_features=500, out_features=2, bias=True)
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

        # compute an & bn
        an_matrix=[]
        bn_matrix=[]
        for i in range(contents_num):
            cur_Matching_matrix=Matching_matrix[i]
            cur_an_matrix = torch.nn.functional.softmax(cur_Matching_matrix,dim=2)  # column wise softmax，对matching matrix每一行归一化和为1 size=(batch, question_len , content_len)
            cur_bn_matrix = torch.nn.functional.softmax(cur_Matching_matrix,dim=1)  # row_wise attention,对matching matrix每一列归一化和为1 size=(batch, question_len , content_len)
            an_matrix.append(cur_an_matrix)
            bn_matrix.append(cur_bn_matrix)

        #compute RnQ & RnD
        RnQ=[] # list[tensor[16,100,256]]
        RnD=[]
        for i in range(contents_num):
            cur_an_matrix=an_matrix[i]
            cur_content_encode=same_sized_content_encode[i]
            cur_bn_matrix=bn_matrix[i]
            cur_RnQ=self.compute_RnQ(cur_an_matrix,cur_content_encode) #size=(batch, curbatch_max_question_len , 256)     eg[16,100,256]
            cur_RnD=self.compute_RnD(cur_bn_matrix,same_sized_question_encode)  #size=(batch, curbatch_max_content_len , 256)    eg[16,200,256]
            RnQ.append(cur_RnQ)
            RnD.append(cur_RnD)

        ########### compute Mmn' ##############
        D_RnD=[] # 先获得D和RnD的concatenation
        for i in range(contents_num):
            cur_content_encode=same_sized_content_encode[i].transpose(0,1) # size=(16,200,256)
            cur_RnD=RnD[i] #size=(16,200,256)
            # embed()
            cur_D_RnD=torch.cat([cur_content_encode,cur_RnD],dim=2)  # size=(16,200,512)
            D_RnD.append(cur_D_RnD)

        RmD=[] # list[tensor（16,200,512）]
        for i in range(contents_num):
            D_RnD_m=D_RnD[i] # size=(16,200,512)
            RmD_i=[]
            for j in range(contents_num):
                D_RnD_n=D_RnD[j]  # size=(16,200,512)
                Mmn_i_j=self.compute_cross_document_attention(D_RnD_m,D_RnD_n) # 计算任意两个文档之间的attention Mmn_i_j size=（16,200,200）
                beta_mn_i_j=torch.nn.functional.softmax(Mmn_i_j,dim=2) #每一行归一化为1 size=（16,200,200）
                cur_RmD=torch.bmm(beta_mn_i_j,D_RnD_n) # size=（16,200,512）
                RmD_i.append(cur_RmD)

            RmD_i=torch.stack(RmD_i) #size=（10,16,200,512）
            RmD_i=RmD_i.transpose(0,1) #size=（16,10,200,512）
            RmD_i=torch.sum(RmD_i,dim=1) #size=（16,200,512）
            RmD.append(RmD_i)

        # RmD=torch.stack(RmD).transpose(0,1) #size=(16,10,200,512)

        matching_feature=[]  # list[tensor(16,2,200)]
        for i in range(contents_num):
            cur_Matching_matrix=Matching_matrix[i] # size=(16,100,200)
            cur_max_pooling_feature,_=torch.max(cur_Matching_matrix,dim=1) #size=(16,200)
            cur_mean_pooling_feature=torch.mean(cur_Matching_matrix,dim=1) #size=(16,200)
            cur_matching_feature=torch.stack([cur_max_pooling_feature,cur_mean_pooling_feature]).transpose(0,1) #size=(16,2,200)
            matching_feature.append(cur_matching_feature)

        reasoning_feature=[]
        for i in range(contents_num):
            cur_RnQ=RnQ[i].transpose(0,1) #size=(100,16,256)
            cur_RmD=RmD[i].transpose(0,1) #size=(200,16,512)
            cur_matching_feature=matching_feature[i].transpose(0,1)#size=(2,16,200)
            cur_RnQ_mask = compute_mask(RnQ[i].mean(dim=2) , PreprocessData.padding_idx)
            cur_RmD_mask=compute_mask(RmD[i].mean(dim=2) , PreprocessData.padding_idx)
            cur_matching_feature_mask=compute_mask(matching_feature[i].mean(dim=2) , PreprocessData.padding_idx)

            # 经过reasoning层
            cur_RnQ_reasoning_out,_=self.question_reasoning_layer.forward(cur_RnQ,cur_RnQ_mask) #size=(max_sequence_len,16,256)
            cur_RmD_reasoning_out,_=self.content_reasoning_layer.forward(cur_RmD,cur_RmD_mask) #size=(max_sequence_len,16,256)
            cur_matching_feature_reasoning_out,_=self.matching_feature_reasoning_layer.forward(cur_matching_feature,cur_matching_feature_mask) #size=(max_sequence_len,16,256)
            # 所有的矩阵变成相同的大小
            cur_RnQ_reasoning_out=self.full_matrix_to_specify_size(cur_RnQ_reasoning_out,[max_question_len,batch_size,cur_RnQ_reasoning_out.size()[2]]) # size=(200,16,256)
            cur_RmD_reasoning_out = self.full_matrix_to_specify_size(cur_RmD_reasoning_out, [max_content_len, batch_size,cur_RmD_reasoning_out.size()[2]])  # size=(100,16,256)
            cur_matching_feature_reasoning_out = self.full_matrix_to_specify_size(cur_matching_feature_reasoning_out, [2, batch_size,cur_matching_feature_reasoning_out.size()[2]])  # size=(2,16,256)
            # 将三种feature cat到一起得到302*256的表示
            cur_reasoning_feature=torch.cat([cur_RnQ_reasoning_out,cur_RmD_reasoning_out,cur_matching_feature_reasoning_out],dim=0).transpose(0,1) #size(16,302,256)
            reasoning_feature.append(cur_reasoning_feature)
        # 10个文档的cat到一起
        reasoning_feature =torch.cat(reasoning_feature,dim=1) #size=(16,3020,256)

        maxpooling_reasoning_feature_column,_=torch.max(reasoning_feature,dim=1) #size(16,256)
        meanpooling_reasoning_feature_column=torch.mean(reasoning_feature,dim=1) #size(16,256)

        maxpooling_reasoning_feature_row,_=torch.max(reasoning_feature,dim=2) #size=(16,3020)
        meanpooling_reasoning_feature_row=torch.mean(reasoning_feature,dim=2) #size=(16,3020)
        # print(228, "============================")
        # embed()
        pooling_reasoning_feature=torch.cat([maxpooling_reasoning_feature_row,meanpooling_reasoning_feature_row,maxpooling_reasoning_feature_column,meanpooling_reasoning_feature_column],dim=1).view(batch_size,-1) #size=(16,6552)

        output=self.decision_layer.forward(pooling_reasoning_feature) #size=(16,2)
        return output


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