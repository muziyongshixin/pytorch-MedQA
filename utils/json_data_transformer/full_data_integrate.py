# -*- coding: utf-8 -*-
# 本文件用于数据预处理的相关操作  处理的数据集是full data 那个文件夹
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

JIEBA_DIC_PATH = "/m/liyz/MedQA-workspace/rawdata/alphaMLE_dict.txt"
jieba.load_userdict(JIEBA_DIC_PATH)

# 将多个文件的数据合并到一个文件里，同时将text和question里的句子进行分词
file_list = ["kexue_2018jingbian_ESresult.txt", "kexue_exercise_ESresult.txt", "renwei_exam_not2017_ESresult.txt",
             "renwei_exercise_ESresult.txt","network_exam_ESresult.txt","network_exercise_ESresult.txt"]
path = "/m/liyz/MedQA-workspace/rawdata/full_data/"
exe_subname = ["A", "B", "C", "D", "E"]


def integrate_files():
    with open('sample.txt', 'w', encoding='utf-8') as out_file:
        sample_count = 0
        fake_count = 0
        flag = 0
        for file in file_list:
            file = path + file
            print(file + "==================================================================")
            f = open(file, "r")
            # print(len(f))
            for line in f:
                flag = 0
                cur_data = json.loads(line)
                out_data = {}

                question_str = cur_data['question']
                word_list = jieba.cut(question_str)
                out_data['question'] = ' '.join(word_list)

                out_data['id'] = cur_data['id']

                out_data['source'] = cur_data['source']

                out_data['correct'] = cur_data['answer']

                candidate_list = cur_data['es_research_facts']
                for candidate in exe_subname:
                    candidate_now = candidate_list['Q+' + candidate]

                    word_list = jieba.cut(candidate_now['text'])
                    candidate_now['text'] = ' '.join(word_list)

                    facts_list = candidate_now['facts']
                    facts_list_out = []
                    if len(facts_list) is not 10:
                        fake_count += 1
                        print(fake_count)
                        flag = 1
                        break
                    else:
                        for fact in facts_list:
                            word_list = jieba.cut(fact['content'])
                            fact['content'] = ' '.join(word_list)
                            facts_list_out.append(fact)

                    candidate_now['facts'] = facts_list_out

                    out_data[candidate] = candidate_now

                out_data_str = json.dumps(out_data, ensure_ascii=False)
                if flag is not 1:
                    out_file.writelines(out_data_str + "\n")
                    sample_count += 1
                if sample_count % 100 == 0:
                    print(sample_count)
            print('fake_count' + str(fake_count))
            print('sample_count' + str(sample_count))


# 将一个句子变成单词集合
def sentence2word(folder_path="C:\\Users\\32706\\Desktop\\新建文件夹\\train"):
    output_file = open(folder_path + "/split_word.txt", "w", encoding='utf-8')
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
                word_list = temp_line.split(" ")
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


# 此函数用于将json数据里的sentence单独拿出来，构成数据集用于word2vec训练
def get_word2vec_data(json_filepath, output_filepath):
    sample_count = 0
    with open(json_filepath, "r") as json_file:
        with open(output_filepath, "w", encoding='utf-8') as output_file:
            for line in json_file.readlines():
                json_data = json.loads(line)
                question = json_data["question"]
                output_file.writelines(question + "\n")

                for candidate in exe_subname:
                    candidate_data = json_data[candidate]
                    output_file.writelines(candidate_data["text"] + "\n")
                    for fact in candidate_data["facts"]:
                        content_data = fact["content"]
                        output_file.writelines(content_data + "\n")
                sample_count += 1

                if sample_count % 1000 == 0:
                    print(sample_count)


# 本函数作用是将所有数据分为训练集和验证集
def split_train_and_dev_dataset(file_path, train_path, dev_path, train_num, dev_num):
    all_samples = []
    with open(file_path, "r") as source_file:
        for line in source_file.readlines():
            all_samples.append(line)

    print("read file completed========")
    random.shuffle(all_samples)
    dev_sample = all_samples[0:dev_num]
    train_sample = all_samples[dev_num:]
    temp_count = 0
    with open(dev_path, "w", encoding="utf-8") as dev_file:
        for sample in dev_sample:
            dev_file.writelines(sample)
            temp_count += 1

    print("dev data completed", temp_count)

    temp_count = 0
    with open(train_path, "w", encoding="utf-8") as train_file:
        for sample in train_sample:
            train_file.writelines(sample)
            temp_count += 1
    print("train data completed", temp_count)


if __name__ == '__main__':
    split_train_and_dev_dataset(file_path="sample.txt",
                                train_path="train_120000.json",
                                dev_path="dev_10031.json",
                                train_num=120000,
                                dev_num=10031)
    # integrate_files()
