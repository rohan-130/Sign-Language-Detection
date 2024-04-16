import pandas as pd
import numpy as np
from words import Words
from os import walk
import sys
import pickle
import os

train_dir = os.listdir('/Users/rohan/Downloads/processed_250_signs_hands/train')
NUM_WORDS = len(train_dir)

sys.setrecursionlimit(4000)

combinations = []

for i in range(0, NUM_WORDS - 4, 5):
    combinations.append([train_dir[j] for j in range(i, i + 5)])


# combinations = [['jacket', 'brown', 'toy', 'morning', 'tree']]
# combinations = [['red', 'elephant', 'owl', 'dad', 'where']]

# for word_names in combinations:
def check_acc(word_names, num_states1=[3, 3, 3, 3, 2]):
    test_words = {}
    train_words = {}
    words = Words()
    words.num_dimensions = 8
    num_right = {}
    num_left = {}

    for word_name in word_names:
        print(word_name)
        filenames = \
        next(walk('/Users/rohan/Downloads/processed_250_signs_hands/train/{0}'.format(word_name)), (None, None, []))[
            2]
        for filename in filenames:
            a = pd.read_parquet(
                '/Users/rohan/Downloads/processed_250_signs_hands/train/{0}/{1}'.format(word_name, filename),
                engine='pyarrow')
            df_right = a.loc[a['type'] == 'right_hand']
            df_left = a.loc[a['type'] == 'left_hand']
            count_right = count_left = 0
            for val in df_left['x']:
                if not np.isnan(val):
                    count_left += 1
            for val in df_right['x']:
                if not np.isnan(val):
                    count_right += 1
            if count_right > count_left:
                if word_name not in num_right:
                    num_right[word_name] = 0
                num_right[word_name] += 1
            else:
                if word_name not in num_left:
                    num_left[word_name] = 0
                num_left[word_name] += 1
    print("num_left", num_left)
    print("num_right", num_right)

    for word_name in word_names:
        print(word_name)
        filenames = \
        next(walk('/Users/rohan/Downloads/processed_250_signs_hands/train/{0}'.format(word_name)), (None, None, []))[
            2]
        for filename in filenames:
            a = pd.read_parquet(
                '/Users/rohan/Downloads/processed_250_signs_hands/train/{0}/{1}'.format(word_name, filename),
                engine='pyarrow')
            df_right = a.loc[a['type'] == 'right_hand']
            df_left = a.loc[a['type'] == 'left_hand']
            count_right = count_left = 0
            for val in df_left['x']:
                if not np.isnan(val):
                    count_left += 1
            for val in df_right['x']:
                if not np.isnan(val):
                    count_right += 1
            df_right = df_right.dropna()
            df_left = df_left.dropna()
            if word_name in ['tree', 'jacket'] and count_left > count_right:
                continue
            if word_name not in ['tree', 'jacket'] and count_right > count_left:
                continue
            if (num_left[word_name] > num_right[word_name] and count_right > count_left) \
                    or (num_right[word_name] > num_left[word_name] and count_left > count_right):
                continue
            if count_right > count_left:
                frames = pd.unique(df_right['frame'])
            else:
                frames = pd.unique(df_left['frame'])
            new_list = []
            for j in range(len(frames)):
                if count_right > count_left:
                    frame_df = df_right.loc[df_right['frame'] == frames[j]]
                else:
                    frame_df = df_left.loc[df_left['frame'] == frames[j]]
                x_values = list(frame_df['x'][:21])
                y_values = list(frame_df['y'][:21])
                new_x_values = [
                    # nose_tip_x - x_values[0],
                    x_values[0],
                    (x_values[4] - x_values[0]),  # * 100,#
                    x_values[12] - x_values[0],
                    x_values[20] - x_values[0]
                ]
                new_y_values = [
                    # nose_tip_y - y_values[0],
                    y_values[0],
                    y_values[4] - y_values[0],
                    y_values[12] - y_values[0],
                    y_values[20] - y_values[0]
                ]
                x_values = new_x_values
                y_values = new_y_values
                dimensions = x_values + y_values
                new_list.append(dimensions)
            if word_name not in train_words:
                train_words[word_name] = []
            if new_list:
                new_new_list = []
                for i in range(len(new_list) - 1):
                    dims = new_list[i]
                    new_new_list.append([])
                    for j in range(len(dims)):
                        new_new_list[i].append(new_list[i + 1][j] - new_list[i][j])
                new_new_list = new_list
                train_words[word_name].append(new_new_list)
            if len(train_words[word_name]) > 100:
                break

    count = 0
    for word in train_words:
        words.update_word({word: train_words[word]}, num_states1[count])
        try:
            words.update_word({word: train_words[word]}, num_states1[count])
        except RecursionError:
            print("recursion error")
            # sys.exit()
            print("RecursionError")
            words.update_word({word: train_words[word][:50]}, num_states1[count])
        count += 1

    with open('newmodel.pkl', 'wb') as file:
        pickle.dump(words, file)
    test_words = {}
    for word_name in word_names:
        filenames = \
        next(walk('/Users/rohan/Downloads/processed_250_signs_hands/test/{0}'.format(word_name)), (None, None, []))[
            2]
        for filename in filenames:
            a = pd.read_parquet(
                '/Users/rohan/Downloads/processed_250_signs_hands/test/{0}/{1}'.format(word_name, filename),
                engine='pyarrow')
            df_right = a.loc[a['type'] == 'right_hand']
            df_left = a.loc[a['type'] == 'left_hand']
            # nose_tip_x = (a.loc[a['type'] == 'face'])['x'][1]
            # nose_tip_y = (a.loc[a['type'] == 'face'])['y'][1]
            count_right = count_left = 0
            for val in df_left['x']:
                if not np.isnan(val):
                    count_left += 1
            for val in df_right['x']:
                if not np.isnan(val):
                    count_right += 1
            df_right = df_right.dropna()
            df_left = df_left.dropna()
            if word_name in ['tree', 'jacket'] and count_left > count_right:
                continue
            if word_name not in ['tree', 'jacket'] and count_right > count_left:
                continue
            '''if (num_left[word_name] > num_right[word_name] and count_right > count_left)\
                    or (num_right[word_name] > num_left[word_name] and count_left > count_right):
                continue'''
            if count_right > count_left:
                frames = pd.unique(df_right['frame'])
            else:
                frames = pd.unique(df_left['frame'])
            new_list = []
            for j in range(len(frames)):
                if count_right > count_left:
                    frame_df = df_right.loc[df_right['frame'] == frames[j]]
                else:
                    frame_df = df_left.loc[df_left['frame'] == frames[j]]
                x_values = list(frame_df['x'][:21])
                y_values = list(frame_df['y'][:21])
                new_x_values = [
                    # nose_tip_x - x_values[0],
                    x_values[0],
                    (x_values[4] - x_values[0]),
                    x_values[12] - x_values[0],
                    x_values[20] - x_values[0],
                ]
                new_y_values = [
                    # nose_tip_y - y_values[0],
                    y_values[0],
                    (y_values[4] - y_values[0]),
                    y_values[12] - x_values[0],
                    y_values[20] - y_values[0],
                ]
                x_values = new_x_values
                y_values = new_y_values
                dimensions = x_values + y_values
                new_list.append(dimensions)
            if word_name not in test_words:
                test_words[word_name] = []
            if new_list:
                new_new_list = []
                for i in range(len(new_list) - 1):
                    dims = new_list[i]
                    new_new_list.append([])
                    for j in range(len(dims)):
                        new_new_list[i].append(new_list[i + 1][j] - new_list[i][j])
                new_new_list = new_list
                test_words[word_name].append(new_new_list)
            if len(test_words[word_name]) > 1000:
                break
    print("TEST WORDS:")
    # test_words = train_words
    for w in test_words:
        print(w, test_words[w])
    total_total = 0
    correct_total = 0
    for word in test_words:
        total = 0
        correct = 0
        for j in range(len(test_words[word])):
            ans = words.check_word(test_words[word][j], states=False)
            if word == ans:
                correct += 1
                correct_total += 1
            total += 1
            total_total += 1
        print("accuracy:", correct * 100 / total)
    try:
        print("total accuracy:", correct_total * 100 / total_total)
    except:
        pass
    print(correct_total)
    print(total_total)

    # print(num_right)
    # print(num_left)
    return correct_total * 100 / total_total


all_best_states = {}
all_best_accs = {}
comb = combinations[0]
# comb = ['jacket', 'brown', 'toy', 'morning', 'tree']
# comb = ['red', 'elephant', 'owl', 'dad', 'where']
# for comb in combinations[20:21]:
for comb in [comb]:
    best_num_states = [2, 2, 2, 2, 2]
    # best_num_states = [3, 3, 3, 8, 4]
    highest_acc = 0
    i = 0
    count = 0
    while i < len(comb):
        num = 2
        num_states = best_num_states.copy()
        num_states[i] = num
        while num < 5:
            print(num_states)
            acc = check_acc(comb, num_states)
            if acc > highest_acc:
                highest_acc = acc
                best_num_states[i] = num
            num += 1
            num_states[i] = num
        if i == (len(comb) - 1):
            count += 1
            if count <= 1:
                i = 0
            else:
                i += 1
        else:
            i += 1
    all_best_states[",".join(comb)] = best_num_states
    all_best_accs[",".join(comb)] = highest_acc

for comb in all_best_states:
    print(comb, all_best_states[comb])

for comb in all_best_accs:
    print(comb, all_best_accs[comb])

# red,elephant,owl,dad,where [3, 3, 3, 3, 4]
# red,elephant,owl,dad,where 70.14925373134328

