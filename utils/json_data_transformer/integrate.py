# -*- coding: utf-8 -*-
# 本文件用于数据预处理的相关操作
#
#
# integrate_files(folder_path ):
# 将多个文件的数据合并到一个文件里，同时将text和question里的句子进行分词
# 
# def sentence2word(folder_path = "C:\\Users\\32706\\Desktop\\新建文件夹\\train"):
# 将一个句子变成单词集合
# 
# def get_word2vec_data(json_filepath,output_filepath):
# 此函数用于将json数据里的sentence单独拿出来，构成数据集用于word2vec训练
#  


import os
import json
import jieba
import random


JIEBA_DIC_PATH="/m/liyz/MedQA-workspace/rawdata/alphaMLE_dict.txt"
jieba.load_userdict(JIEBA_DIC_PATH)

# 将多个文件的数据合并到一个文件里，同时将text和question里的句子进行分词
file_list=["exam-2017_result"]
exe_subname=["A","B","C","D","E"]
def integrate_files(folder_path ):
    raw_files=os.listdir(folder_path)
    sample_count=0
    with open(folder_path + "/dev.json", "w",encoding='utf-8') as new_file:
        for file in file_list:
            print(file+"==================================================================")
            if not os.path.isdir(file):
                cur_file=open(folder_path+"/"+file,"r")
                for line in cur_file.readlines():
                    cur_data=json.loads(line)
                    for candidate in exe_subname:
                        candidate_text=cur_data[candidate]["text"]
                        word_list=jieba.cut(candidate_text)
                        new_candidate_text = " ".join(word_list)
                        cur_data[candidate]["text"]=new_candidate_text
                    question_text=cur_data["question"]
                    word_list=jieba.cut(question_text)
                    new_question_text=" ".join(word_list)
                    cur_data["question"]=new_question_text
                    new_json_str=json.dumps(cur_data,ensure_ascii=False)
                    new_file.writelines(new_json_str+"\n")
                    sample_count+=1
                    if sample_count%100==0:
                        print(sample_count)

    print(sample_count)



# 将一个句子变成单词集合
def sentence2word(folder_path = "C:\\Users\\32706\\Desktop\\新建文件夹\\train"):
    output_file = open(folder_path + "/split_word.txt", "w",encoding='utf-8')
    word_select = ["question", "A", "B", "C", "D", "E"]

    word_set = set()
    json_data = json.load(open(folder_path + "/example_data.json", "rb"))
    data = json_data["data"]
    for ele in data:
        for sub_name in word_select:
            try:
                cur_setence = ele[sub_name].strip()
                word_list = jieba.cut(cur_setence)
                temp_line = " ".join(word_list)
                output_file.writelines(temp_line + "\n")
                word_list=temp_line.split(" ")
                for word in word_list:
                    word_set.add(word)

            except:
                print("出错句子=====================", ele[sub_name].strip())

    output_file.flush()
    output_file.close()
    # print(ele["question"])
    # word_list=jieba.lcut(ele["question"])
    # print(" ".join(word_list))
    # set.add(word for word in word_list)

    print("total chinese word is %d" % len(word_set))



#此函数用于将json数据里的sentence单独拿出来，构成数据集用于word2vec训练
def get_word2vec_data(json_filepath,output_filepath):
    sample_count=0
    with open(json_filepath,"r") as json_file:
        with open(output_filepath,"w",encoding='utf-8') as output_file:
            for line in json_file.readlines():
                json_data=json.loads(line)
                question=json_data["question"]
                output_file.writelines(question+"\n")
                
                for candidate in exe_subname:
                    candidate_data=json_data[candidate]
                    output_file.writelines(candidate_data["text"]+"\n")
                    for fact in candidate_data["facts"]:
                        content_data=fact["content"]
                        output_file.writelines(content_data+"\n")
                sample_count+=1
                
                if sample_count%1000==0:
                    print(sample_count)
                        
# 本函数作用是将所有数据分为训练集和验证集
def split_train_and_dev_dataset(file_path,train_path,dev_path,train_num,dev_num):
    all_samples=[]
    with open(file_path,"r") as source_file:
        for line in source_file.readlines():
            all_samples.append(line)

    print("read file completed========")
    random.shuffle(all_samples)
    dev_sample=all_samples[0:dev_num]
    train_sample=all_samples[dev_num:]
    temp_count=0
    with open(dev_path, "w", encoding="utf-8") as dev_file:
        for sample in dev_sample:
            dev_file.writelines(sample)
            temp_count+=1

    print("dev data completed",temp_count)

    temp_count = 0
    with open(train_path,"w",encoding="utf-8") as train_file:
        for sample in train_sample:
            train_file.writelines(sample)
            temp_count += 1
    print("train data completed",temp_count)



if __name__ == '__main__':
    split_train_and_dev_dataset(file_path="/m/liyz/MedQA-workspace/rawdata/jieba_data/train_and_dev_65136.json",
                                train_path="/m/liyz/MedQA-workspace/rawdata/jieba_data/train_60000.json",
                                dev_path="/m/liyz/MedQA-workspace/rawdata/jieba_data/dev_5136.json",
                                train_num=60000,
                                dev_num=5136)