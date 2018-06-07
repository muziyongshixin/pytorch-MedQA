# -*- coding: utf-8 -*-

import os
import json
import jieba

def integrate_files(folder_path = "C:\\Users\\32706\\Desktop\\新建文件夹\\train"):

    files=os.listdir(folder_path)

    example_dataset={"description":"example_dataset",
                     "sample_count":0,
                     "data":[]}

    sample_count=0
    for file in files:
        if not os.path.isdir(file):
            cur_file=open(folder_path+"/"+file,"rb")
            for line in cur_file:
                cur_data=json.loads(line)
                # print(cur_data)
                if "question" in cur_data and  "answer" in cur_data:
                    example_dataset["data"].append(cur_data)
                    sample_count+=1
                else:
                    print(cur_file,line)
            if sample_count%100==0:
                print(sample_count)

    example_dataset["sample_count"]=sample_count

    json_str=json.dumps(example_dataset,ensure_ascii=False)

    with open(folder_path + "/example_data.json", "w",encoding='utf-8') as f:
        f.write(json_str)
    print(sample_count)




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


if __name__ == '__main__':
    sentence2word()