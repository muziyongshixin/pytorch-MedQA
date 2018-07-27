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

JIEBA_DIC_PATH = "/m/liyz/MedQA-workspace/rawdata/alphaMLE_dict.txt"
jieba.load_userdict(JIEBA_DIC_PATH)

path = "/m/liyz/MedQA-workspace/rawdata/es_result_0713_production_idx/"

# 将多个文件的数据合并到一个文件里，同时将text和question里的句子进行分词
# train_dev_file_list = ["data_zb_f_ESresult.txt", "kexue_2018jingbian_ESresult.txt",
#                        "kexue_exercise_ESresult.txt", "network_exam_ESresult.txt",
#                        "network_exercise_ESresult.txt","renwei_exercise_ESresult.txt",
#                        "renwei_exam_not2017_ESresult.txt", "zhuojian_notlc_ESresult.txt",
#                        "yishijie_ESresult.txt","zhuojian_lc_ESresult.txt"]
train_dev_file_list = ["zhuojian_notlc_ESresult.txt"]
test_file_list = ['renwei_exam_2017_ESresult.txt']

out_train_file = path + "v6_for_try/v6_train.json"
out_dev_file = path + "v6_for_try/v6_dev.json"
out_test_file = path + 'v6_for_try/v6_test.json'

exe_subname = ["A", "B", "C", "D", "E"]


def integrate_files(file_list, type):
    all_sample = []
    sample_count = 0
    fake_count = 0
    flag = 0
    for file in file_list:
        file = path + file
        print(file + "=========================================================")
        f = open(file, "r")
        # print(len(f))
        for line in f:
            flag = 0
            cur_data = json.loads(line)
            out_data = {}

            question_str = cur_data['question']
            word_list = jieba.cut(question_str)
            out_data['question'] = ' '.join(word_list)
            if out_data['question'] == '':
                print("question长度出现问题，单词数为0 问题id为%s" % cur_data['id'])
                fake_count += 1
                print("fake sample 句子长度出现问题", fake_count)
                continue

            out_data['id'] = cur_data['id']

            out_data['source'] = cur_data['source']

            out_data['correct'] = cur_data['answer']

            out_data['logic'] = cur_data['logic'][0]

            out_data['question_category'] = cur_data['question_category']

            candidate_list = cur_data['es_research_facts']
            for candidate in exe_subname:
                candidate_now = candidate_list['Q+' + candidate]

                word_list = jieba.cut(candidate_now['text'])
                candidate_now['text'] = ' '.join(word_list)
                if candidate_now['text']=='':
                    print("content长度出现问题，单词数为0 问题id为%s"%cur_data['id'])
                    flag=1
                    fake_count+=1
                    print("fake sample 句子长度出现问题", fake_count)
                    break
                facts_list = candidate_now['facts']
                facts_list_out = []
                if len(facts_list) is not 10:
                    fake_count += 1
                    print("fake sample",fake_count)
                    flag = 1
                    break
                else:
                    for fact in facts_list:
                        word_list = jieba.cut(fact['content'])
                        fact['content'] = ' '.join(word_list)
                        facts_list_out.append(fact)

                candidate_now['facts'] = facts_list_out

                out_data[candidate] = candidate_now

            # out_data_str = json.dumps(out_data, ensure_ascii=False)
            if flag is not 1:
                all_sample.append(out_data)
                sample_count += 1
            if sample_count % 1000 == 0:
                print(sample_count)
    print('fake_count' + str(fake_count))
    print('sample_count' + str(sample_count))
    return all_sample


# 本函数作用是将所有数据分为训练集和验证集
def split_train_and_dev_dataset(all_sample_data, out_train_path, out_dev_path, train_num, dev_num):
    sample_count = len(all_sample_data)
    train_count = train_num
    dev_count = sample_count - train_num
    if dev_count != dev_num:
        print("set dev_num is incorrect !!!!!!!!!!!!!!!!")
    all_samples = all_sample_data
    print("all sample =%d \t set train size=%d  set dev size=%d  ...." % (sample_count, train_count, dev_count))
    random.shuffle(all_samples)
    train_sample = all_samples[0:train_num]

    dev_sample = all_samples[train_num:]

    write_sample_to_files(dev_sample, out_dev_path)
    print("dev data completed", len(dev_sample))

    write_sample_to_files(train_sample, out_train_path)
    print("train data completed", len(train_sample))


def write_sample_to_files(samples, output_path):
    sample_count = len(samples)
    temp_count = 0
    file_path = output_path + ".%d" % sample_count
    print("save samples to path: %s" % file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        for sample in samples:
            out_data_str = json.dumps(sample, ensure_ascii=False)
            file.writelines(out_data_str + "\n")
            temp_count += 1
            if temp_count % 1000 == 0:
                print(temp_count)


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


def main():
    print("test execute begin......")
    test_samples = integrate_files(test_file_list, 'test')
    write_sample_to_files(test_samples, out_test_file)

    print("train and dev execute begin......")
    train_dev_samples = integrate_files(train_dev_file_list, 'train')
    split_train_and_dev_dataset(train_dev_samples, out_train_file, out_dev_file, train_num=140000, dev_num=9009)

    print("completed ......")


if __name__ == '__main__':
    # integrate_files()
    main()
# integrate_files()
