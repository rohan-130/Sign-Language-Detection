import pickle
from .convert import multidimensional_viterbi, divide, convert, divide_into_num_states


class Words:
    def __init__(self):
        self.all_words = {}
        self.states = []
        self.prior_probs = {}
        self.emission_paras = {}
        self.transition_probs = {}
        self.state_nums = ["1", "2", "3", "4", "end"]
        self.num_dimensions = 2
        self.state_nums_per_word = {}

    def update_word(self, input_words, numStates=None):
        for word in input_words:
            if word not in self.all_words:
                self.state_nums_per_word[word] = []
                for num in range(int(numStates)):
                    self.state_nums_per_word[word].append(str(num+1))
                self.state_nums_per_word[word].append("end")
                self.all_words[word] = []
                for state_num in self.state_nums_per_word[word]:
                    self.states.append(word + state_num)
                    '''1 self.emission_paras[word + state_num] = [(None, None), (None, None)]'''
                    self.emission_paras[word + state_num] = [(None, None)] * self.num_dimensions
                    self.transition_probs[word + state_num] = {}
                    for state_num2 in self.state_nums_per_word[word]:
                        '''2 self.transition_probs[word + state_num][word + state_num2] = (0, 0)'''
                        self.transition_probs[word + state_num][word + state_num2] = (0,) * self.num_dimensions
            for vector in input_words[word]:
                self.all_words[word].append(vector)

        # Change all prior probs in self.prior_probs and add new ones
        for word in self.all_words:
            '''self.prior_probs[word + self.state_nums_per_word[word][0]] = 1.0 / (
                    float(len(self.states)) / len(self.state_nums_per_word[word]))  # 1 / number of words'''
            self.prior_probs[word + self.state_nums_per_word[word][0]] = 1.0 / (
                    len(self.all_words))  # 1 / number of words
            for state_num in self.state_nums_per_word[word][1:]:
                self.prior_probs[word + state_num] = 0
        for word in input_words:
            all_dim_vectors = [[]] * self.num_dimensions
            '''3 x_vectors = []
            y_vectors = []'''
            for i in range(len(self.all_words[word])):
                vector = self.all_words[word][i]
                '''4 x_vectors.append([])
                y_vectors.append([])'''
                for i_dimension in range(self.num_dimensions):
                    all_dim_vectors[i_dimension] = all_dim_vectors[i_dimension] + [[]]
                '''5 for x, y in vector:
                    x_vectors[i].append(x)
                    y_vectors[i].append(y)'''
                for all_dims in vector:
                    for i_dimension in range(self.num_dimensions):
                        all_dim_vectors[i_dimension][i] = all_dim_vectors[i_dimension][i] + [all_dims[i_dimension]]
            # result_x = divide(x_vectors)
            # result_y = divide(y_vectors)
            '''6 result_x = divide_into_num_states(x_vectors, num_states=len(self.state_nums_per_word[word])-1)
            result_y = divide_into_num_states(y_vectors, num_states=len(self.state_nums_per_word[word])-1)
            result_x = convert(result_x)
            result_y = convert(result_y)'''
            all_dim_results = []
            for i_dimension in range(self.num_dimensions):
                all_dim_results = all_dim_results + \
                                  [divide_into_num_states(all_dim_vectors[i_dimension], num_states=len(self.state_nums_per_word[word])-1)]
            new_all_dim_results = []
            for i_dimension in range(self.num_dimensions):
                new_all_dim_results = new_all_dim_results + [convert(all_dim_results[i_dimension])]
            for i in range(len(self.state_nums_per_word[word]) - 1):
                '''7 self.emission_paras[word + self.state_nums_per_word[word][i]] = [(result_x[1][i], result_x[2][i]),
                                                                  (result_y[1][i], result_y[2][i])]'''
                self.emission_paras[word + self.state_nums_per_word[word][i]] = []
                for i_dimension in range(len(all_dim_vectors)):
                    self.emission_paras[word + self.state_nums_per_word[word][i]].append(
                        (new_all_dim_results[i_dimension][1][i], new_all_dim_results[i_dimension][2][i])
                    )
            '''8 result_x = result_x[0]
            result_y = result_y[0]'''
            for i_dimension in range(self.num_dimensions):
                new_all_dim_results[i_dimension] = new_all_dim_results[i_dimension][0]
            for i in range(len(self.state_nums_per_word[word]) - 1):
                '''9 avg_x = 0
                avg_y = 0'''
                all_avg = [0] * self.num_dimensions
                for j in range(len(new_all_dim_results[0][i])):
                    '''10 vector_state_x = result_x[i][j]
                    vector_state_y = result_y[i][j]'''
                    vector_state_i = []
                    for i_dimension in range(self.num_dimensions):
                        vector_state_i.append(new_all_dim_results[i_dimension][i][j])
                    '''11 avg_x += len(vector_state_x)
                    avg_y += len(vector_state_y)'''
                    for i_dimension in range(len(all_dim_vectors)):
                        all_avg[i_dimension] += len(vector_state_i[i_dimension])
                '''12 avg_x = avg_x / len(input_words[word])
                avg_y = avg_y / len(input_words[word])'''
                for i_dimension in range(self.num_dimensions):
                    all_avg[i_dimension] = all_avg[i_dimension] / len(input_words[word])
                '''13 self.transition_probs[word + str(i + 1)][word + str(i + 1)] = (1 - round(1.0 / avg_x, 3),
                                                                               1 - round(1.0 / avg_y, 3))'''
                self.transition_probs[word + str(i + 1)][word + str(i + 1)] = []
                for i_dimension in range(self.num_dimensions):
                    self.transition_probs[word + str(i + 1)][word + str(i + 1)].append(
                        1 - round(1.0 / all_avg[i_dimension], 3)
                    )
                if i < len(self.state_nums_per_word[word]) - 2:  # if i < 2:
                    '''14 self.transition_probs[word + str(i + 1)][word + str(i + 2)] = (round(1.0 / avg_x, 3),
                                                                                   round(1.0 / avg_y, 3))'''
                    self.transition_probs[word + str(i + 1)][word + str(i + 2)] = []
                    for i_dimension in range(self.num_dimensions):
                        self.transition_probs[word + str(i + 1)][word + str(i + 2)].append(
                            round(1.0 / all_avg[i_dimension], 3)
                        )
                else:
                    ''' 15self.transition_probs[word + str(i + 1)][word + "end"] = (round(1.0 / avg_x, 3),
                                                                              round(1.0 / avg_y, 3))'''
                    self.transition_probs[word + str(i + 1)][word + "end"] = []
                    for i_dimension in range(self.num_dimensions):
                        self.transition_probs[word + str(i + 1)][word + "end"].append(
                            round(1.0 / all_avg[i_dimension], 3)
                        )


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
        s, p = multidimensional_viterbi(evidence_vector, self.states, self.prior_probs, self.transition_probs, self.emission_paras, num_dimensions=self.num_dimensions)
        if s is None:
            return s, p
        if states:
            return s
        return s[0][:-1]



"""
from .convert import multidimensional_viterbi, divide2, convert
import pickle


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
        state_nums = self.state_nums
        for word in input_words:
            if word not in self.all_words:
                self.all_words[word] = []
                for state_num in state_nums:
                    self.states.append(word + state_num)
                    self.emission_paras[word + state_num] = [(None, None)] * self.num_dimensions
                    self.transition_probs[word + state_num] = {}
                    for state_num2 in state_nums:
                        self.transition_probs[word + state_num][word + state_num2] = (0,) * self.num_dimensions
            for vector in input_words[word]:
                self.all_words[word].append(vector)

        # Change all prior probs in self.prior_probs and add new ones
        for word in self.all_words:
            self.prior_probs[word + state_nums[0]] = 1.0 / (
                    float(len(self.states)) / len(state_nums))  # 1 / number of words
            for state_num in state_nums[1:]:
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
                all_results[z] = divide2(all_vectors[z], len(state_nums)-1)
                all_results[z] = convert(all_results[z])
            for i in range(len(state_nums) - 1):
                self.emission_paras[word + state_nums[i]] = []
                for z in range(self.num_dimensions):
                    self.emission_paras[word + state_nums[i]].append((all_results[z][1][i], all_results[z][2][i]))
            for z in range(self.num_dimensions):
                all_results[z] = all_results[z][0]
            for i in range(len(state_nums) - 1):
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

    def check_word(self, evidence_vector, states=True, prob=False):
        s, p = multidimensional_viterbi(evidence_vector, self.states, self.prior_probs, self.transition_probs, self.emission_paras, num_dimensions=self.num_dimensions)
        if s is None:
            return s, p
        if prob:
            return s[0][:-1], p
        if s is None or states:
            return s
        return s[0][:-1]




words = Words()
words.update_word(
{'newword': [[(0.1579970121383667, 0.06726086139678955), (-0.31495988368988037, -0.08040368556976318), (-0.017449259757995605, -0.018408894538879395), (1.0261744260787964, 0.3371506929397583), (4.214540123939514, 1.046016812324524), (3.962603211402893, 0.5798548460006714), (6.631410121917725, -0.2239525318145752), (7.5177401304244995, -0.29274821281433105), (4.747426509857178, -0.056049227714538574), (7.039737701416016, -0.7765293121337891), (4.425829648971558, -0.4848212003707886), (7.0679426193237305, -0.7608354091644287), (5.142533779144287, 0.48505067825317383), (5.15788197517395, 0.39650797843933105), (7.784014940261841, 1.3573944568634033), (6.9550275802612305, 1.9887030124664307), (5.531060695648193, 3.9568960666656494), (2.325916290283203, 1.1115610599517822), (1.680433750152588, 0.5227982997894287), (-3.4291625022888184, -3.673577308654785), (-3.038966655731201, -3.7627696990966797), (-6.323021650314331, -3.693905472755432), (-7.963508367538452, -3.0900269746780396), (-7.795137166976929, -1.679489016532898), (-4.31513786315918, -0.0721365213394165), (-4.643237590789795, 0.2211928367614746), (-7.2032153606414795, -0.09873509407043457), (-5.266517400741577, -0.1528918743133545), (-7.944843173027039, 0.17755329608917236)]]}, 3
)
# print(words.all_words)
print("Done!")

"""