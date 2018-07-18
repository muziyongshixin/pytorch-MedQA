

candidates = ["A", "B", "C", "D", "E"]
count = 0
with open(path, 'r') as f:
    for line in f:
        cur_data = json.loads(line)
        for candidate in candidates:
            cur_text = cur_data[candidate]['text']
            words = cur_text.split(" ")
            for word in words:
                if '\u0001' == word:
                    count += 1
                    print(True)
print(count)
