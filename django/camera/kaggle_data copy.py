import pandas as pd
import numpy as np
from words import Words
from os import walk
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(2500)


train_files = [
    'bird/1006690809',
    'bird/1019494452',
    # 'bird/1025250464',
    #         'bird/1068361635',
    #         'bird/104663714',
    # 'bird/1044708688',
    # 'blow/1002092995',
    'blow/1008843441',
    #       'blow/1028804567',
    # 'blow/1031724967',
    'blow/1061460539',
    #         'blow/1052304950',
    #         'owie/1006264561',
    #'owie/1007127288',
    #'owie/1011584461'
    #         'owie/1073640705',
    #         'owie/1018622870',
    # 'owie/1070645326',
    # 'cloud/100015657',
    # 'wait/1000106739'
    'duck/1000241583',
    'minemy/1000255522'
]
test_files = [
    'blow/1028804567',
    'bird/1068361635',
    'bird/104663714',
    'bird/1025250464',
    'minemy/1000255522'
]
word_names = [
    "bird","bird",
    "blow", "blow",
    # "owie",
    # "cloud",
    # "wait",
    "duck",
    "minemy",
]
test_word_names = [
    'blow',
    'bird', 'bird', 'bird', "minemy",
]

test_words = {}
train_words = {}

words = Words()
words.num_dimensions = 8

word_names = ['flower', 'mitten', 'bird', 'time', 'vacuum',]
# 'flower', 'mitten', 'bird']
# word_names = ['all', 'animal', 'another', 'any', 'apple']
word_names = ['flower', 'mitten', 'vacuum', 'cloud', 'duck']   # total accuracy: 100.0 - [3, 3, 3, 3, 3] : 100.0
word_names = ['puzzle', 'minemy', 'vacuum', 'cloud', 'duck']  # total accuracy: 93.33333333333333 - [3, 3, 3, 3, 3]

for word_name in word_names:
    filenames = next(walk('trained_landmark_files/{0}'.format(word_name)), (None, None, []))[2]
    # filenames = next(walk('/Users/rohan/Downloads/processed_250_signs_hands/train/{0}'.format(word_name)), (None, None, []))[2]  # [] if no file
    print(filenames)
    for filename in filenames:
        a = pd.read_parquet('trained_landmark_files/{0}/{1}'.format(word_name, filename), engine='pyarrow')
        # a = pd.read_parquet('/Users/rohan/Downloads/processed_250_signs_hands/train/{0}/{1}'.format(word_name, filename), engine='pyarrow')
        df = a.loc[a['type'] == 'right_hand']
        df_left = a.loc[a['type'] == 'left_hand']
        # nose_tip_x = (a.loc[a['type'] == 'face'])['x'][1]
        # nose_tip_y = (a.loc[a['type'] == 'face'])['y'][1]
        df = df.dropna()
        df_left = df_left.dropna()
        if df.empty:
            continue
        print("new:")
        print((1-df_left.empty) * ("LEFT: "+word_name))
        print((1-df.empty) * ("RIGHT: "+word_name))
        frames = pd.unique(df['frame'])
        new_list = []
        for j in range(len(frames)):
            frame_df = df.loc[df['frame'] == frames[j]]
            x_values = list(frame_df['x'][:21])
            y_values = list(frame_df['y'][:21])
            new_x_values = [
                # nose_tip_x - x_values[0],
                x_values[0],
                x_values[4] - x_values[0],
                x_values[12] - x_values[0],
                x_values[20] - x_values[0],
            ]
            new_y_values = [
                # nose_tip_y - y_values[0],
                y_values[0],
                y_values[4] - y_values[0],
                y_values[12] - y_values[0],
                y_values[20] - y_values[0],
            ]
            x_values = new_x_values
            y_values = new_y_values
            dimensions = x_values + y_values
            new_list.append(dimensions)
        if word_name not in train_words:
            train_words[word_name] = []
        if new_list != []:
            train_words[word_name].append(new_list)
        if len(train_words[word_name]) > 150:
            break

'''for i in range(len(train_files)):
    a = pd.read_parquet('trained_landmark_files/{0}.parquet'.format(train_files[i]), engine='pyarrow')
    df = a.loc[a['type'] == 'right_hand']
    df = df.dropna()
    frames = pd.unique(df['frame'])
    new_list = []
    for j in range(len(frames)):
        frame_df = df.loc[df['frame'] == frames[j]]
        x_values = list(frame_df['x'][:10])
        y_values = list(frame_df['y'][:10])
        dimensions = x_values + y_values
        new_list.append(dimensions)
    if word_names[i] not in train_words:
        train_words[word_names[i]] = []
    train_words[word_names[i]].append(new_list)'''

num_states1 = [3,3,2,2,3]
num_states1 = [3,3,3,3,3]

count = 0
for word in train_words:
    print("training", word)
    print("training words[wordname] length:", len(train_words[word_name]))
    try:
        words.update_word({word: train_words[word]}, num_states1[count])
    except RecursionError:
        words.update_word({word: train_words[word][:100]}, num_states1[count])
    ans = words.check_word(train_words[word][0], states=False)
    print("ANS:", ans)
    count += 1

'''
for word_name in word_names:
    filenames = next(walk('/Users/rohan/Downloads/processed_250_signs_hands/train/{0}'.format(word_name)), (None, None, []))[2]  # [] if no file
    for filename in filenames:
        a = pd.read_parquet('/Users/rohan/Downloads/processed_250_signs_hands/train/{0}/{1}'.format(word_name, filename), engine='pyarrow')
        df = a.loc[a['type'] == 'right_hand']
        df = df.dropna()
        if df.empty:
            continue
        frames = pd.unique(df['frame'])
        new_list = []
        for j in range(len(frames)):
            frame_df = df.loc[df['frame'] == frames[j]]
            x_values = list(frame_df['x'][:10])
            y_values = list(frame_df['y'][:10])
            dimensions = x_values + y_values
            new_list.append(dimensions)
        if word_name not in test_words:
            test_words[word_name] = []
        if new_list != []:
            test_words[word_name].append(new_list)
        if len(test_words[word_name]) > 20:
            break

print("TESTWORDS:")
print(test_words)'''
test_words = train_words

'''for i in range(len(test_files)):
    a = pd.read_parquet('trained_landmark_files/{0}.parquet'.format(test_files[i]), engine='pyarrow')
    df = a.loc[a['type'] == 'right_hand']
    df = df.dropna()
    frames = pd.unique(df['frame'])
    new_list = []
    for j in range(len(frames)):
        frame_df = df.loc[df['frame'] == frames[j]]
        x_values = list(frame_df['x'][:10])
        y_values = list(frame_df['y'][:10])
        dimensions = x_values + y_values
        new_list.append(dimensions)
    if test_word_names[i] not in test_words:
        test_words[test_word_names[i]] = []
    test_words[test_word_names[i]].append(new_list)'''

total_total = 0
correct_total = 0
for word in test_words:
    total = 0
    correct = 0
    print("word:", word)
    print(len(test_words[word]))
    for j in range(len(test_words[word])):
        ans = words.check_word(test_words[word][j], states=False)
        # print("Guessed:", ans)
        if word == ans:
            correct += 1
            correct_total += 1
        else:
            pass
            # print(test_words[word][j])
        total += 1
        total_total += 1
    print("accuracy:", correct*100/total)
    print(num_states1, ":", correct*100/total)
    print(correct)
    print(total)
try:
    print("total accuracy:", correct_total*100/total_total)
    print(num_states1, ":", correct_total*100/total_total)
except:
    pass
print(correct_total)
print(total_total)


'''


- i’m now using relative positions of the palm from the nose and fingers from the palm and it works better
- the program doesn’t let me train more than 100 samples for a word because of a recursive limit. So I’m trying to convert the recursive function to a loop using my own stack

- the clean data wasn’t working as well, but i think that’s because i still need to figure out which one is the dominant hand
- how do i check which hand is dominant? Both hands have x,y positions in the clean data, so I’m not sure which one I should use


3,3,3,3: 50%
[4, 3, 3, 3] : 25.0
[3, 5, 3, 3] : 100.0
[]
apple: 
flower: 
lips: 3
mitten: 3
there: 

time, vacuum, flower, mitten, 

word: flower
accuracy: 77.77777777777777
word: mitten
accuracy: 100.0
word: bird
accuracy: 100.0
word: time
accuracy: 87.5
word: vacuum
accuracy: 85.71428571428571
total accuracy: 88.57142857142857

'''
