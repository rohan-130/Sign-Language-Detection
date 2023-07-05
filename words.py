import pickle
from convert import multidimensional_viterbi, divide, convert


class Words:
    def __init__(self):
        self.all_words = {}
        self.states = []
        self.prior_probs = {}
        self.emission_paras = {}
        self.transition_probs = {}
        self.state_nums = ["1", "2", "3", "end"]
        self.num_dimensions = 2

    def update_word(self, input_words):
        for word in input_words:
            if word not in self.all_words:
                self.all_words[word] = []
                for state_num in self.state_nums:
                    self.states.append(word + state_num)
                    self.emission_paras[word + state_num] = [(None, None), (None, None)]
                    self.transition_probs[word + state_num] = {}
                    for state_num2 in self.state_nums:
                        self.transition_probs[word + state_num][word + state_num2] = (0, 0)
            for vector in input_words[word]:
                self.all_words[word].append(vector)

        # Change all prior probs in self.prior_probs and add new ones
        for word in self.all_words:
            self.prior_probs[word + self.state_nums[0]] = 1.0 / (
                    float(len(self.states)) / len(self.state_nums))  # 1 / number of words
            for state_num in self.state_nums[1:]:
                self.prior_probs[word + state_num] = 0

        for word in input_words:
            x_vectors = []
            y_vectors = []
            for i in range(len(self.all_words[word])):
                vector = self.all_words[word][i]
                x_vectors.append([])
                y_vectors.append([])
                for x, y in vector:
                    x_vectors[i].append(x)
                    y_vectors[i].append(y)
            result_x = divide(x_vectors)
            result_y = divide(y_vectors)
            result_x = convert(result_x)
            result_y = convert(result_y)
            for i in range(len(self.state_nums) - 1):
                self.emission_paras[word + self.state_nums[i]] = [(result_x[1][i], result_x[2][i]),
                                                                  (result_y[1][i], result_y[2][i])]
            result_x = result_x[0]
            result_y = result_y[0]
            for i in range(len(self.state_nums) - 1):
                avg_x = 0
                avg_y = 0
                for j in range(len(result_x[i])):
                    vector_state_x = result_x[i][j]
                    vector_state_y = result_y[i][j]
                    avg_x += len(vector_state_x)
                    avg_y += len(vector_state_y)
                avg_x = avg_x / len(input_words[word])
                avg_y = avg_y / len(input_words[word])
                self.transition_probs[word + str(i + 1)][word + str(i + 1)] = (1 - round(1.0 / avg_x, 3),
                                                                               1 - round(1.0 / avg_y, 3))
                if i < 2:
                    self.transition_probs[word + str(i + 1)][word + str(i + 2)] = (round(1.0 / avg_x, 3),
                                                                                   round(1.0 / avg_y, 3))
                else:
                    self.transition_probs[word + str(i + 1)][word + "end"] = (round(1.0 / avg_x, 3),
                                                                              round(1.0 / avg_y, 3))

    def update_word2(self, input_words):
        for word in input_words:
            if word not in self.all_words:
                self.all_words[word] = []
                for state_num in self.state_nums:
                    self.states.append(word + state_num)
                    self.emission_paras[word + state_num] = [(None, None)] * self.num_dimensions
                    self.transition_probs[word + state_num] = {}
                    for state_num2 in self.state_nums:
                        self.transition_probs[word + state_num][word + state_num2] = (0,) * self.num_dimensions
            for vector in input_words[word]:
                self.all_words[word].append(vector)

        # Change all prior probs in self.prior_probs and add new ones
        for word in self.all_words:
            self.prior_probs[word + self.state_nums[0]] = 1.0 / (
                    float(len(self.states)) / len(self.state_nums))  # 1 / number of words
            for state_num in self.state_nums[1:]:
                self.prior_probs[word + state_num] = 0

        for word in input_words:
            all_vectors = []
            for z in range(self.num_dimensions):
                all_vectors.append([])
            for i in range(len(self.all_words[word])):
                vector = self.all_words[word][i]
                for z in range(self.num_dimensions):
                    (all_vectors[z]).append([])
                for tup in vector:
                    for z in range(self.num_dimensions):
                        (all_vectors[z])[i].append(tup[z])
            all_results = [[]] * self.num_dimensions
            for z in range(self.num_dimensions):
                all_results[z] = divide(all_vectors[z])
                all_results[z] = convert(all_results[z])
            for i in range(len(self.state_nums) - 1):
                self.emission_paras[word + self.state_nums[i]] = []
                for z in range(self.num_dimensions):
                    self.emission_paras[word + self.state_nums[i]].append((all_results[z][1][i], all_results[z][2][i]))
            for z in range(self.num_dimensions):
                all_results[z] = all_results[z][0]
            for i in range(len(self.state_nums) - 1):
                all_avg = [0] * self.num_dimensions
                for j in range(len(all_results[0][i])):
                    all_vector_states = []
                    for z in range(self.num_dimensions):
                        all_vector_states.append(all_results[z][i][j])
                    for z in range(self.num_dimensions):
                        all_avg[z] += len(all_vector_states[z])
                for z in range(self.num_dimensions):
                    all_avg[z] = all_avg[z] / len(input_words[word])
                self.transition_probs[word + str(i + 1)][word + str(i + 1)] = []
                for z in range(self.num_dimensions):
                    self.transition_probs[word + str(i + 1)][word + str(i + 1)].append(1 - round(1.0 / all_avg[z], 3))
                self.transition_probs[word + str(i + 1)][word + str(i + 1)] = tuple(self.transition_probs[word + str(i + 1)][word + str(i + 1)])
                if i < 2:
                    self.transition_probs[word + str(i + 1)][word + str(i + 2)] = []
                    for z in range(self.num_dimensions):
                        self.transition_probs[word + str(i + 1)][word + str(i + 2)].append(round(1.0 / all_avg[z], 3))
                    self.transition_probs[word + str(i + 1)][word + str(i + 2)] = tuple(self.transition_probs[word + str(i + 1)][word + str(i + 2)])
                else:
                    self.transition_probs[word + str(i + 1)][word + "end"] = []
                    for z in range(self.num_dimensions):
                        self.transition_probs[word + str(i + 1)][word + "end"].append(round(1.0 / all_avg[z], 3))
                    self.transition_probs[word + str(i + 1)][word + "end"] = tuple(self.transition_probs[word + str(i + 1)][word + "end"])

    def check_word(self, evidence_vector, states=True):
        s, p = multidimensional_viterbi(evidence_vector, self.states, self.prior_probs, self.transition_probs, self.emission_paras)
        if s is None:
            return s, p
        if states:
            return s
        return s[0][:-1]


words1 = Words()
words1.update_word({"new":   [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], [(2, 2), (3, 3), (1, 1), (4, 4), (5, 5)], [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]})
words1.update_word({"newa":   [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], [(2, 2), (3, 3), (1, 1), (4, 4), (5, 5)], [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]})

words2 = Words()
words2.num_dimensions = 2
words2.update_word({"new":   [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], [(2, 2), (3, 3), (1, 1), (4, 4), (5, 5)], [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]})
words2.update_word({"newa":   [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], [(2, 2), (3, 3), (1, 1), (4, 4), (5, 5)], [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]})

print(words1.transition_probs == words2.transition_probs)
print(words1.emission_paras == words2.emission_paras)
print(words1.all_words == words2.all_words)
print(words1.prior_probs == words2.prior_probs)
print(words1.states == words2.states)
