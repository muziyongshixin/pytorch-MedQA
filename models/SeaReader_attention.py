#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liyz'

import torch
from models.layers import *
from utils.functions import answer_search, multi_scale_ptr
from IPython import embed
from utils.functions import masked_softmax, compute_mask, masked_flip


class SeaReader_attention(torch.nn.Module):
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

    def __init__(self, dataset_h5_path, device):
        super(SeaReader_attention, self).__init__()

        self.device = device
        # set config

        hidden_size = 128
        hidden_mode = 'LSTM'
        dropout_p = 0.2
        # word embedding config
        vocabulary_size = 365553  # 297392
        word_embedding_size = 200

        enable_layer_norm = False
        encoder_word_layers = 1
        encoder_bidirection = True

        reasoning_layer_bidirection = True
        reasoning_layer_nums = 1

        gate_choose = "FC"
        R = 100
        self.R=R

        self.use_content_nums = 5
        self.hidden_size = hidden_size

        # construct model
        self.fixed_embedding = Word2VecEmbedding(dataset_h5_path=dataset_h5_path, trainable=False)

        self.delta_embedding = delta_Embedding(n_embeddings=vocabulary_size, len_embedding=word_embedding_size,
                                               init_uniform=0.1, trainable=True)

        self.context_layer = MyLSTM(mode=hidden_mode,
                                    input_size=word_embedding_size,
                                    hidden_size=hidden_size,
                                    num_layers=encoder_word_layers,
                                    bidirectional=encoder_bidirection,
                                    dropout_p=dropout_p)

        self.reasoning_gating_layer = My_gate_layer(hidden_size * 2, gate_choose)

        self.decision_gating_layer = Noise_gate(hidden_size * 4, gate_choose)

        self.content_reasoning_layer = MyLSTM(mode=hidden_mode,
                                              input_size=hidden_size * 4 + 2,
                                              hidden_size=hidden_size,
                                              num_layers=reasoning_layer_nums,
                                              bidirectional=reasoning_layer_bidirection,
                                              dropout_p=dropout_p)

        self.question_reasoning_layer = MyLSTM(mode=hidden_mode,
                                               input_size=hidden_size * 2 + 2,
                                               hidden_size=hidden_size,
                                               num_layers=reasoning_layer_nums,
                                               bidirectional=reasoning_layer_bidirection,
                                               dropout_p=dropout_p)

        self.attention_layer_qa=torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size * 2, out_features=150, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=150,out_features=R,bias=True)
        )
        self.attention_layer_cont=torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size * 2, out_features=150, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=150,out_features=R,bias=True)
        )

        self.decision_layer = torch.nn.Linear(in_features=R * 2, out_features=1, bias=True)
        torch.nn.init.xavier_uniform_(self.decision_layer.weight)

    def forward(self, contents, question_ans, logics, contents_char=None, question_ans_char=None):
        # contents size(16,10,200)
        # question_ans size(16,100)
        batch_size = question_ans.size()[0]
        max_content_len = contents.size()[2]
        max_question_len = question_ans.size()[1]
        contents_num = contents.size()[1]

        # word-level embedding: (seq_len, batch, embedding_size)
        question_vec, question_mask = self.fixed_embedding.forward(question_ans)  # size=(batch, 100, 200)
        delta_emb = self.delta_embedding.forward(question_ans)  # size=(batch, 100, 200)
        question_vec += delta_emb  # size=(batch, 100, 200)
        question_encode = self.context_layer.forward(question_vec, question_mask)  # size=(batch, 100, 256)

        contents = contents[:, 0:self.use_content_nums, :]  # use top n contents # size=(batch, n, 200)
        viewed_contents = contents.contiguous().view(batch_size * self.use_content_nums, max_content_len)  # size=(batch*n, 200)
        viewed_content_vec, viewed_content_mask = self.fixed_embedding.forward(
            viewed_contents)  # size=(batch*n, 200,256)
        viewed_delta_emb = self.delta_embedding.forward(viewed_contents)  # size=(batch*n, 200,256)
        viewed_content_vec += viewed_delta_emb
        viewed_content_encode = self.context_layer.forward(viewed_content_vec,viewed_content_mask)  # size=(batch*n, 200, 256)
        content_encode = viewed_content_encode.view(batch_size, self.use_content_nums, max_content_len,self.hidden_size * 2).transpose(0, 1)  # size=(n,batch, 200, 256)

        viewed_reasoning_content_gating_val = self.reasoning_gating_layer(viewed_content_encode)  # size=(16*n,200,1)
        reasoning_question_gating_val = self.reasoning_gating_layer(question_encode)  # size=(16,100,1)

        # matching features
        matching_feature_row = []  # list[tensor(16,200,2)]
        matching_feature_col = []  # list[tensor(16,100,2)]
        # compute RnQ & RnD
        RnQ = []  # list[tensor[16,100,256]]
        RnD = []  # list[tensor[16,200,256]]
        D_RnD = []  # 获得D和RnD的concatenation
        for i in range(self.use_content_nums):
            cur_content_encode = content_encode[i]  # size=(16,200,256) 当前的第i个content
            cur_Matching_matrix = self.compute_matching_matrix(question_encode,
                                                               cur_content_encode)  # (batch, question_len , content_len) eg(16,100,200)

            cur_an_matrix = torch.nn.functional.softmax(cur_Matching_matrix,
                                                        dim=2)  # column wise softmax，对matching matrix每一行归一化和为1 size=(batch, question_len , content_len)
            cur_bn_matrix = torch.nn.functional.softmax(cur_Matching_matrix,
                                                        dim=1)  # row_wise attention,对matching matrix每一列归一化和为1 size=(batch, question_len , content_len)

            cur_RnQ = self.compute_RnQ(cur_an_matrix, cur_content_encode)  # size=(batch, 100 , 256)     eg[16,100,256]
            cur_RnD = self.compute_RnD(cur_bn_matrix, question_encode)  # size=[16,200,256]
            cur_D_RnD = torch.cat([cur_content_encode, cur_RnD], dim=2)  # size=(16,200,512)
            D_RnD.append(cur_D_RnD)
            RnQ.append(cur_RnQ)
            RnD.append(cur_RnD)

            # 计算matching feature
            cur_max_pooling_feature_row, _ = torch.max(cur_Matching_matrix, dim=1)  # size=(16,200)
            cur_mean_pooling_feature_row = torch.mean(cur_Matching_matrix, dim=1)  # size=(16,200)
            cur_matching_feature_row = torch.stack([cur_max_pooling_feature_row, cur_mean_pooling_feature_row],
                                                   dim=-1)  # size=(16,200,2)
            matching_feature_row.append(cur_matching_feature_row)

            cur_max_pooling_feature_col, _ = torch.max(cur_Matching_matrix, dim=2)  # size=(16,100)
            cur_mean_pooling_feature_col = torch.mean(cur_Matching_matrix, dim=2)  # size=(16,100)
            cur_matching_feature_col = torch.stack([cur_max_pooling_feature_col, cur_mean_pooling_feature_col],
                                                   dim=-1)  # size=(16,100,2)
            matching_feature_col.append(cur_matching_feature_col)

        RmD = []  # list[tensor（16,200,512）]
        for i in range(self.use_content_nums):
            D_RnD_m = D_RnD[i]  # size=(16,200,512)
            Mmn_i = []
            RmD_i = []
            for j in range(self.use_content_nums):
                D_RnD_n = D_RnD[j]  # size=(16,200,512)
                # 计算任意两个文档之间的attention Mmn_i_j size=（16,200,200）
                Mmn_i_j = self.compute_cross_document_attention(D_RnD_m, D_RnD_n)
                Mmn_i.append(Mmn_i_j)

            Mmn_i = torch.stack(Mmn_i, dim=-1)  # size=(16,200,200,10)
            softmax_Mmn_i = self.reduce_softmax(Mmn_i)  # size=(16,200,200,10)

            for j in range(self.use_content_nums):
                D_RnD_n = D_RnD[j]  # size=(16,200,512)
                beta_mn_i_j = softmax_Mmn_i[:, :, :, j]  # size=(16,200,200)
                cur_RmD = torch.bmm(beta_mn_i_j, D_RnD_n)  # size=（16,200,512）
                RmD_i.append(cur_RmD)

            RmD_i = torch.stack(RmD_i, dim=1)  # size=（16,10,200,512）
            RmD_i = torch.sum(RmD_i, dim=1)  # size=（16,200,512）
            RmD.append(RmD_i)

        RnQ = torch.stack(RnQ, dim=1)  # size 16,n,100,256
        viewed_RnQ = RnQ.view(batch_size * self.use_content_nums, max_question_len,
                              self.hidden_size * 2)  # size 16*n,100,256

        RmD = torch.stack(RmD, dim=1)  # size 16,n,200,256
        viewed_RmD = RmD.view(batch_size * self.use_content_nums, max_content_len,
                              self.hidden_size * 4)  # size 16*n,200,512

        viewed_matching_feature_col = torch.stack(matching_feature_col, dim=1).view(batch_size * self.use_content_nums,
                                                                                    max_question_len,
                                                                                    2)  # size 16*n,100,2
        viewed_matching_feature_row = torch.stack(matching_feature_row, dim=1).view(batch_size * self.use_content_nums,
                                                                                    max_content_len,
                                                                                    2)  # size 16*n,200,2

        viewed_RnQ = torch.cat([viewed_RnQ, viewed_matching_feature_col], dim=-1)  # size 16*n,100,258
        viewed_RmD = torch.cat([viewed_RmD, viewed_matching_feature_row], dim=-1)  # size 16*n,200,514

        viewed_RnQ_mask = compute_mask(viewed_RnQ.mean(dim=2), PreprocessData.padding_idx)  # size 16*n,100,258
        viewed_RmD_mask = compute_mask(viewed_RmD.mean(dim=2), PreprocessData.padding_idx)  # size 16*n,200,514
        viewed_reasoning_question_gating_val = reasoning_question_gating_val.repeat(self.use_content_nums, 1, 1)

        gated_cur_RnQ = self.compute_gated_value(viewed_RnQ, viewed_reasoning_question_gating_val)
        gated_cur_RmD = self.compute_gated_value(viewed_RmD, viewed_reasoning_content_gating_val)

        # 经过reasoning层
        cur_RnQ_reasoning_out = self.question_reasoning_layer.forward(gated_cur_RnQ, viewed_RnQ_mask)  # size=(16*n,100,256)
        cur_RmD_reasoning_out = self.content_reasoning_layer.forward(gated_cur_RmD, viewed_RmD_mask)  # size=(16*n,200,256)

        att_qa=self.attention_layer_qa(cur_RnQ_reasoning_out) # size=(16*n,100,100)
        att_cont=self.attention_layer_cont(cur_RmD_reasoning_out)# size=(16*n,200,100)

        att_qa=torch.nn.functional.softmax(att_qa,dim=1) # size=(16*n,100,100)
        att_cont=torch.nn.functional.softmax(att_cont,dim=1) # size=(16*n,200,100)

        match_qa=torch.bmm(cur_RnQ_reasoning_out.transpose(1,2),att_qa)  #size(16*n,256,100)
        match_cont=torch.bmm(cur_RmD_reasoning_out.transpose(1,2),att_cont) #size(16*n,256,100)

        match_score=match_qa*match_cont #size(16*n,256,100)
        match_score=torch.sum(match_score,dim=1)  #size(16*n,100)

        match_score=match_score.view(batch_size,self.use_content_nums,self.R)  #size(16,n,100)
        mean_pooling=torch.mean(match_score,dim=1) #size(16,100)
        max_pooling=torch.max(match_score,dim=1)[0]#size(16,100)

        decision_feature=torch.cat([max_pooling,mean_pooling],dim=1)  #size(16,200)
        decision_output=self.decision_layer(decision_feature)

        logics = logics.resize_(logics.size()[0], 1)
        output = decision_output * logics

        output = output.view(int(output.size()[0] / 5), 5)
        return output  # logics 是反向的话乘以-1，正向的话是乘以1

    def compute_matching_matrix(self, question_encode, content_encode):
        '''question_encode is batch*question_len*context_size
        content_encode is batch*content_len*context_size'''
        content_encode_trans = content_encode.transpose(1, 2)  # (batch, context_size, seq_len)
        Matching_matrix = torch.bmm(question_encode, content_encode_trans)  # (batch, question_len , content_len)
        return Matching_matrix

    def compute_cross_document_attention(self, content_m, content_n):
        content_n_trans = content_n.transpose(1, 2)  # (batch, 512, 200)
        Matching_matrix = torch.bmm(content_m, content_n_trans)  # (batch, question_len , content_len)
        return Matching_matrix

    def compute_RnQ(self, an_matrix, content_encode):
        ## an_matrix=(batch,100,200)
        # content_encode= (batch, content_len, embedding_size)
        return torch.bmm(an_matrix, content_encode)

    def compute_RnD(self, bn_matrix, question_encode):
        bn_matrix_trans = bn_matrix.transpose(1, 2)  # size=(batch,200,100)
        RnD = torch.bmm(bn_matrix_trans, question_encode)
        return RnD  # (batch, 200, 256)

    def reduce_softmax(self, input_matrix):
        size = input_matrix.size()
        viewd_input = input_matrix.contiguous().view(size[0], size[1], size[2] * size[3])
        softmax_input = torch.nn.functional.softmax(viewd_input, dim=2)
        resized_softmax = softmax_input.contiguous().view(size[0], size[1], size[2], size[3])
        return resized_softmax

    # 通过gate值来过滤
    def compute_gated_value(self, input_matrix, gate_val):
        '''
        input_matrix size=(16,100,258)
        gate_val size=(16,100,1)
        return gated_matrix size=(16,100,258)
        '''
        # batch_size = gate_val.size()[0]
        # words_num = gate_val.size()[-1]
        # eye_matrix = torch.zeros(batch_size, words_num, words_num).to(self.device)  # eg:size=(16,100,100)
        # eye_matrix[:, range(words_num), range(words_num)] = gate_val[:, 0, range(words_num)]
        # gated_matrix = torch.bmm(eye_matrix, input_matrix)
        gated_matrix = input_matrix * gate_val
        return gated_matrix
