import json

file_path="E:/pyCharm WorkSpace/pytorch-MedQA/data/MedQA/ES抽取结果_陈玉莲/临床执业助理医生-模拟题-parsed-all/f_result"
new_file_path="E:/pyCharm WorkSpace/pytorch-MedQA/data/MedQA/ES抽取结果_陈玉莲/临床执业助理医生-模拟题-parsed-all/new_f_result"

data=[]
file=open(file_path, "r")
for line in file.readlines():
    json_data = json.loads(line)
    print(line)
    data.append(json_data)



print("json loaded")
json_str = json.dumps(data, ensure_ascii=False)
print("json dumped")
with open(new_file_path, "w", encoding='utf-8') as f:
    f.write(json_str)


print("done")