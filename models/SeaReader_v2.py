#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liyz'

import torch
from models.layers import *
from utils.functions import answer_search, multi_scale_ptr
from IPython import embed
from utils.functions import masked_softmax, compute_mask, masked_flip


class SeaReader_v2(torch.nn.Module):
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
        super(SeaReader_v2, self).__init__()

        self.device = device
        # set config
        hidden_size = 128
        hidden_mode = 'LSTM'
        dropout_p = 0.2
        # word embedding config
        vocabulary_size=365553    #297392
        word_embedding_size = 200

        enable_layer_norm = False
        encoder_word_layers = 1
        encoder_bidirection = True

        reasoning_layer_bidirection = True
        reasoning_layer_nums=1

        gate_choose="FC"

        # construct model
        self.fixed_embedding = Word2VecEmbedding(dataset_h5_path=dataset_h5_path, trainable=False)

        self.delta_embedding=delta_Embedding(n_embeddings=vocabulary_size,len_embedding=word_embedding_size,init_uniform=0.1,trainable=True)

        self.context_layer = MyLSTM(mode=hidden_mode,
                                       input_size=word_embedding_size,
                                       hidden_size=hidden_size,
                                       num_layers=encoder_word_layers,
                                       bidirectional=encoder_bidirection,
                                       dropout_p=dropout_p)

        self.reasoning_gating_layer = My_gate_layer(256,gate_choose)

        self.decision_gating_layer = Noise_gate(512,gate_choose)

        self.content_reasoning_layer = MyLSTM(mode=hidden_mode,
                                                 input_size=hidden_size * 4+2,
                                                 hidden_size=hidden_size,
                                                 num_layers=reasoning_layer_nums,
                                                 bidirectional=reasoning_layer_bidirection,
                                                 dropout_p=dropout_p)

        self.question_reasoning_layer = MyLSTM(mode=hidden_mode,
                                                  input_size=hidden_size * 2+2,
                                                  hidden_size=hidden_size,
                                                  num_layers=reasoning_layer_nums,
                                                  bidirectional=reasoning_layer_bidirection,
                                                  dropout_p=dropout_p)

        self.decision_layer = torch.nn.Linear(in_features=1024, out_features=1, bias=True)
        torch.nn.init.xavier_uniform_(self.decision_layer.weight)

    def forward(self, contents, question_ans, logics, contents_char=None, question_ans_char=None):
        # assert contents_char is not None and question_ans_char is not None
        batch_size = question_ans.size()[0]
        max_content_len = contents.size()[2]
        max_question_len = question_ans.size()[1]
        contents_num = contents.size()[1]
        # word-level embedding: (seq_len, batch, embedding_size)
        content_vec = []
        content_mask = []
        question_vec, question_mask = self.fixed_embedding.forward(question_ans)
        delta_emb = self.delta_embedding.forward(question_ans)
        question_vec+=delta_emb
        question_encode = self.context_layer.forward(question_vec, question_mask)  # size=(batch, 100, 256)
        content_encode = []  # word-level encode: (seq_len, batch, hidden_size)
        for i in range(contents_num):
            cur_content = contents[:, i, :]
            cur_content_vec, cur_content_mask = self.fixed_embedding.forward(cur_content)
            delta_emb = self.delta_embedding.forward(cur_content)
            cur_content_vec+=delta_emb
            cur_content_encode = self.context_layer.forward(cur_content_vec, cur_content_mask)  # size=(batch, 200, 256)
            content_encode.append(cur_content_encode)

        # 计算gating layer的值
        reasoning_content_gating_val = []
        reasoning_question_gating_val = None
        for i in range(contents_num):
            cur_content_encode = content_encode[i]  # size=(16,200,256)
            cur_reasoning_content_gating_val = self.reasoning_gating_layer(cur_content_encode)  # size=(16,200,1)
            cur_reasoning_content_gating_val =cur_reasoning_content_gating_val+(1e-8) # 防止出现gate为0的情况,导致后面padsequence的时候出错
            reasoning_content_gating_val.append(cur_reasoning_content_gating_val)
        reasoning_question_gating_val = self.reasoning_gating_layer(question_encode)  # size=(16,100,1)
        reasoning_question_gating_val=reasoning_question_gating_val+(1e-8) # 防止出现gate为0的情况,导致后面padsequence的时候出错

        # 计算gate loss todo: 貌似无法返回多个变量，暂时无用
        # question_gate_val = torch.cat([reasoning_question_gating_val.view(-1), decision_question_gating_val.view(-1)])
        # reasoning_gate_val = torch.cat([ele.view(-1) for ele in reasoning_content_gating_val])
        # decision_gate_val = torch.cat([ele.view(-1) for ele in decision_content_gating_val])
        # all_gate_val = torch.cat([question_gate_val, reasoning_gate_val, decision_gate_val])
        # mean_gate_val = torch.mean(all_gate_val)


        # Matching Matrix computing, question 和每一个content都要计算matching matrix

        #matching features
        matching_feature_row = []  # list[tensor(16,200,2)]
        matching_feature_col = []  # list[tensor(16,100,2)]
        # compute RnQ & RnD
        RnQ = []  # list[tensor[16,100,256]]
        RnD = []  # list[tensor[16,200,256]]
        D_RnD = []  # 获得D和RnD的concatenation
        for i in range(contents_num):
            cur_content_encode = content_encode[i] #size=(16,200,256) 当前的第i个content
            cur_Matching_matrix = self.compute_matching_matrix(question_encode,cur_content_encode)  # (batch, question_len , content_len) eg(16,100,200)

            cur_an_matrix = torch.nn.functional.softmax(cur_Matching_matrix,dim=2)  # column wise softmax，对matching matrix每一行归一化和为1 size=(batch, question_len , content_len)
            cur_bn_matrix = torch.nn.functional.softmax(cur_Matching_matrix,dim=1)  # row_wise attention,对matching matrix每一列归一化和为1 size=(batch, question_len , content_len)

            cur_RnQ = self.compute_RnQ(cur_an_matrix, cur_content_encode)  # size=(batch, 100 , 256)     eg[16,100,256]
            cur_RnD = self.compute_RnD(cur_bn_matrix, question_encode)  # size=[16,200,256]
            cur_D_RnD = torch.cat([cur_content_encode, cur_RnD], dim=2)  # size=(16,200,512)
            D_RnD.append(cur_D_RnD)
            RnQ.append(cur_RnQ)
            RnD.append(cur_RnD)

            # 计算matching feature
            cur_max_pooling_feature_row, _ = torch.max(cur_Matching_matrix, dim=1)  # size=(16,200)
            cur_mean_pooling_feature_row = torch.mean(cur_Matching_matrix, dim=1)  # size=(16,200)
            cur_matching_feature_row = torch.stack([cur_max_pooling_feature_row, cur_mean_pooling_feature_row],dim=-1)  # size=(16,200,2)
            matching_feature_row.append(cur_matching_feature_row)

            cur_max_pooling_feature_col, _ = torch.max(cur_Matching_matrix, dim=2)  # size=(16,100)
            cur_mean_pooling_feature_col = torch.mean(cur_Matching_matrix, dim=2)  # size=(16,100)
            cur_matching_feature_col = torch.stack([cur_max_pooling_feature_col, cur_mean_pooling_feature_col],dim=-1)  # size=(16,100,2)
            matching_feature_col.append(cur_matching_feature_col)

        RmD = []  # list[tensor（16,200,512）]
        for i in range(contents_num):
            D_RnD_m = D_RnD[i]  # size=(16,200,512)
            Mmn_i = []
            RmD_i = []
            for j in range(contents_num):
                D_RnD_n = D_RnD[j]  # size=(16,200,512)
                # 计算任意两个文档之间的attention Mmn_i_j size=（16,200,200）
                Mmn_i_j = self.compute_cross_document_attention(D_RnD_m,D_RnD_n)
                Mmn_i.append(Mmn_i_j)

            Mmn_i = torch.stack(Mmn_i,dim=-1)  # size=(16,200,200,10)
            softmax_Mmn_i = self.reduce_softmax(Mmn_i)  # size=(16,200,200,10)

            for j in range(contents_num):
                D_RnD_n = D_RnD[j]  # size=(16,200,512)
                beta_mn_i_j = softmax_Mmn_i[:, :, :, j] # size=(16,200,200)
                cur_RmD = torch.bmm(beta_mn_i_j, D_RnD_n)  # size=（16,200,512）
                RmD_i.append(cur_RmD)

            RmD_i = torch.stack(RmD_i,dim=1)  # size=（16,10,200,512）
            RmD_i = torch.sum(RmD_i, dim=1)  # size=（16,200,512）
            RmD.append(RmD_i)

        reasoning_feature = []
        for i in range(contents_num):
            cur_RnQ = RnQ[i]  # size=(16,100,256)
            cur_RmD = RmD[i]  # size=(16,200,512)
            cur_matching_feature_col = matching_feature_col[i]  # size=(16,100,2)
            cur_matching_feature_row = matching_feature_row[i]  # size=(16,200,2)

            cur_RnQ = torch.cat([cur_RnQ, cur_matching_feature_col], dim=2)  # size=(16,100,258)
            cur_RmD = torch.cat([cur_RmD, cur_matching_feature_row], dim=2)  # size=(16,200,514)

            cur_RnQ_mask = compute_mask(cur_RnQ.mean(dim=2), PreprocessData.padding_idx)
            cur_RmD_mask = compute_mask(cur_RmD.mean(dim=2), PreprocessData.padding_idx)

            gated_cur_RnQ=self.compute_gated_value(cur_RnQ,reasoning_question_gating_val)# size=(16,100,258)
            gated_cur_RmD=self.compute_gated_value(cur_RmD,reasoning_content_gating_val[i])# size=(16,200,514)

            # 经过reasoning层
            cur_RnQ_reasoning_out = self.question_reasoning_layer.forward(gated_cur_RnQ,cur_RnQ_mask)  # size=(16,100,256)
            cur_RmD_reasoning_out= self.content_reasoning_layer.forward(gated_cur_RmD,cur_RmD_mask)  # size=(16,200,256)

            RnQ_reasoning_out_max_pooling,_=torch.max(cur_RnQ_reasoning_out,dim=1)# size=(16,256)
            RmD_reasoning_out_max_pooling,_=torch.max(cur_RmD_reasoning_out,dim=1) #size(16,256)

            cur_reasoning_feature=torch.cat([RnQ_reasoning_out_max_pooling,RmD_reasoning_out_max_pooling],dim=1)# size(16,512)
            reasoning_feature.append(cur_reasoning_feature)

        reasoning_feature=torch.stack( reasoning_feature,dim=1) #size=(16,10,512)
        noise_gate_val=self.decision_gating_layer(reasoning_feature) #size=(16,10,1)
        gated_reasoning_feature=self.compute_gated_value(reasoning_feature,noise_gate_val) #size=(16,10,512)

        reasoning_out_max_feature,_=torch.max(gated_reasoning_feature,dim=1) #size=(16,512)
        reasoning_out_mean_feature=torch.mean(gated_reasoning_feature,dim=1)#size=(16,512)

        decision_input=torch.cat([reasoning_out_max_feature,reasoning_out_mean_feature],dim=1)#size(16,1024)

        decision_output = self.decision_layer.forward(decision_input)  # size=(16,1)
        logics=logics.resize_(logics.size()[0],1)
        output=decision_output*logics

        output=output.view(int(output.size()[0]/5), 5)
        return output # logics 是反向的话乘以-1，正向的话是乘以1

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
        #content_encode= (batch, content_len, embedding_size)
        return torch.bmm(an_matrix, content_encode)

    def compute_RnD(self, bn_matrix, question_encode):
        bn_matrix_trans = bn_matrix.transpose(1, 2)  # size=(batch,200,100)
        RnD = torch.bmm(bn_matrix_trans, question_encode)
        return RnD # (batch, 200, 256)

    def reduce_softmax(self,input_matrix):
        size=input_matrix.size()
        viewd_input=input_matrix.contiguous().view(size[0],size[1],size[2]*size[3])
        softmax_input=torch.nn.functional.softmax(viewd_input,dim=2)
        resized_softmax=softmax_input.contiguous().view(size[0],size[1],size[2],size[3])
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
        gated_matrix=input_matrix*gate_val
        return gated_matrix
