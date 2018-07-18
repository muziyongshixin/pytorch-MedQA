import json
import csv

path = 'C:/Users/32706/Desktop/avg_problem_acc=0.5033.csv'

csv_file = csv.reader(open(path, "r"))
row_count = 1

result = {"stat_correct": 0, "stat_sum": 0, "knw_correct": 0, "knw_sum": 0, "infr_correct": 0, "infr_sum": 0}
logic_result={"p_logic_correct":0,"p_logic_sum":0,"n_logic_correct":0,"n_logic_sum":0}
for row in csv_file:
    if (row_count - 1) != 0 and (row_count - 1) % 6 == 0:
        logic_str=row[7]
        if logic_str=="1":
            logic_str="p_logic"
        else:
            logic_str="n_logic"
        category_str=row[6].replace("\'","\"")
        categorys = json.loads(category_str)
        cur_catgory = 'stat'
        cur_cat_score = 0.0
        for subname in categorys:
            if categorys[subname] > cur_cat_score:
                cur_cat_score = categorys[subname]
                cur_catgory = subname

        is_correct = row[10]

        if is_correct=="TRUE":
            result[cur_catgory + "_correct"] += 1
            result[cur_catgory + "_sum"] += 1
            logic_result[logic_str+"_correct"]+=1
            logic_result[logic_str+"_sum"]+=1
        else:
            result[cur_catgory + "_sum"] += 1
            logic_result[logic_str+"_sum"]+=1
    row_count+=1

print(result)
print(logic_result)
