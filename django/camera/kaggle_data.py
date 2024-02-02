import pandas as pd
import numpy as np
from words import Words


'''for i in range(3, 10):
    for j in range(3, 10):
        for k in range(3, 10):
            for l in range(3, 10):
                for m in range(3, 10):
                    print(i, j, k, l, m)'''


files = ['100015657', '1000035562', '1000106739', '1000210073', '1000240708']

files = ['1000035562', '1000210073', '1000240708', '1000241583', '1000255522',]
'''      '1000661926',
         '1000862366',
         '1001145816', '1001158776',
         ]''' #  '100035691', '100039661']
all_words_new = []
word_names = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
words = Words()
words.num_dimensions = 2

for i in range(len(files)):
    a = pd.read_parquet('trained_landmark_files/{0}.parquet'.format(files[i]), engine='pyarrow')
    max_index_number = -1
    for index, row in a.iterrows():
        if row["type"] == "right_hand":
            rowid = row["row_id"]
            index_number = int(rowid[rowid.find("right_hand-") + len("right_hand-"):])
            if index_number > max_index_number:
                max_index_number = index_number

    new_list = [[]] * (max_index_number + 1)
    count = 0
    check_null_x = a['x'].isnull()
    check_null_y = a['y'].isnull()
    for index, row in a.iterrows():
        if row["type"] == "right_hand":
            rowid = row["row_id"]
            index_number = int(rowid[rowid.find("right_hand-") + len("right_hand-"):])
            if not check_null_x[index] and not check_null_y[index]:
                new_list[index_number] = new_list[index_number] + [[float(row["x"]), float(row["y"])]]
                count += 1
    if word_names[i] == "first":
        words.update_word({word_names[i]: new_list}, 5)
    elif word_names[i] == "second":
        words.update_word({word_names[i]: new_list}, 9)
    elif word_names[i] == "third":
        words.update_word({word_names[i]: new_list}, 3)
    elif word_names[i] == "fourth":
        words.update_word({word_names[i]: new_list}, 3)
    else:
        words.update_word({word_names[i]: new_list}, 3)
    print(len(new_list))
    #  print(words.check_word(new_list[0]))
    all_words_new.append(new_list)
    print(word_names[i])

total = 0
correct = 0
for i in range(len(files)):
    for j in range(len(all_words_new[i])):
        ans = words.check_word(all_words_new[i][j], states=False)
        print(word_names[i], ":", ans)
        if word_names[i] == ans:
            correct += 1
        total += 1
print("accuracy:", correct*100/total)


"""
first: num_states = 5 (acc 85.7 with all others = 3
second: 9
third: 3
fourth: 



blow
wait
cloud
bird
owie
duck
minemy
lips
flower
time
vacuum
apple
puzzle
"""
