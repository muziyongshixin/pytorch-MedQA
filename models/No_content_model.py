import torch
from models.layers import *

class No_content_model(torch.nn.Module):
    def __init__(self,dataset_h5_path):
        super(No_content_model,self).__init__()
        hidden_mode='LSTM'
        word_embedding_size=200
        hidden_size=150
        encoder_word_layers=1
        encoder_bidirection=True
        dropout_p=0.3
        ws_factor=1
        R=10
        embedding_trainable=False

        vocabulary_size=365553

        self.fixed_embedding=Word2VecEmbedding(dataset_h5_path=dataset_h5_path, trainable=embedding_trainable)
        self.delta_embedding = delta_Embedding(n_embeddings=vocabulary_size, len_embedding=word_embedding_size,
                                               init_uniform=0.1, trainable=True)
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
            torch.nn.Linear(in_features=ws_factor * hidden_size, out_features=R)
        )
        self.answer_attention_layer=torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size*2,out_features=ws_factor*hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=ws_factor*hidden_size,out_features=R)
       )

        self.output_layer=torch.nn.Linear(in_features=R,out_features=1)
    def forward(self, questions,answers,logics):
        # questions 20,60
        # answers 20,12
        batch_size=answers.size()[0]
        questions_vec,question_mask=self.fixed_embedding(questions)       #batch 60 200
        answers_vec,answers_mask=self.fixed_embedding(answers) #batch 12 200
        questions_vec +=self.delta_embedding(questions)
        answers_vec +=self.delta_embedding(answers)

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

        output=self.output_layer(sum_match_feature) #batch 1
        output=output.view(int(batch_size/5),5)
        return output