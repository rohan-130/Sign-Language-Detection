import pickle
from convert import multidimensional_viterbi, convert, divide_into_num_states

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
                    self.emission_paras[word + state_num] = [(None, None)] * self.num_dimensions
                    self.transition_probs[word + state_num] = {}
                    for state_num2 in self.state_nums_per_word[word]:
                        self.transition_probs[word + state_num][word + state_num2] = (0,) * self.num_dimensions
            for vector in input_words[word]:
                self.all_words[word].append(vector)

        for word in self.all_words:
            self.prior_probs[word + self.state_nums_per_word[word][0]] = 1.0 / (
                    len(self.all_words))
            for state_num in self.state_nums_per_word[word][1:]:
                self.prior_probs[word + state_num] = 0
        for word in input_words:
            all_dim_vectors = [[]] * self.num_dimensions
            for i in range(len(self.all_words[word])):
                vector = self.all_words[word][i]
                for i_dimension in range(self.num_dimensions):
                    all_dim_vectors[i_dimension] = all_dim_vectors[i_dimension] + [[]]
                for all_dims in vector:
                    for i_dimension in range(self.num_dimensions):
                        all_dim_vectors[i_dimension][i] = all_dim_vectors[i_dimension][i] + [all_dims[i_dimension]]
            all_dim_results = []
            for i_dimension in range(self.num_dimensions):
                all_dim_results = all_dim_results + \
                                  [divide_into_num_states(all_dim_vectors[i_dimension], num_states=len(self.state_nums_per_word[word])-1)]
            new_all_dim_results = []
            for i_dimension in range(self.num_dimensions):
                new_all_dim_results = new_all_dim_results + [convert(all_dim_results[i_dimension])]
            for i in range(len(self.state_nums_per_word[word]) - 1):
                self.emission_paras[word + self.state_nums_per_word[word][i]] = []
                for i_dimension in range(len(all_dim_vectors)):
                    self.emission_paras[word + self.state_nums_per_word[word][i]].append(
                        (new_all_dim_results[i_dimension][1][i], new_all_dim_results[i_dimension][2][i])
                    )
            for i_dimension in range(self.num_dimensions):
                new_all_dim_results[i_dimension] = new_all_dim_results[i_dimension][0]
            for i in range(len(self.state_nums_per_word[word]) - 1):
                all_avg = [0] * self.num_dimensions
                for j in range(len(new_all_dim_results[0][i])):
                    vector_state_i = []
                    for i_dimension in range(self.num_dimensions):
                        vector_state_i.append(new_all_dim_results[i_dimension][i][j])
                    for i_dimension in range(len(all_dim_vectors)):
                        all_avg[i_dimension] += len(vector_state_i[i_dimension])
                for i_dimension in range(self.num_dimensions):
                    all_avg[i_dimension] = all_avg[i_dimension] / len(input_words[word])
                self.transition_probs[word + str(i + 1)][word + str(i + 1)] = []
                for i_dimension in range(self.num_dimensions):
                    self.transition_probs[word + str(i + 1)][word + str(i + 1)].append(
                        1 - round(1.0 / all_avg[i_dimension], 3)
                    )
                if i < len(self.state_nums_per_word[word]) - 2:  # if i < 2:
                    self.transition_probs[word + str(i + 1)][word + str(i + 2)] = []
                    for i_dimension in range(self.num_dimensions):
                        self.transition_probs[word + str(i + 1)][word + str(i + 2)].append(
                            round(1.0 / all_avg[i_dimension], 3)
                        )
                else:
                    self.transition_probs[word + str(i + 1)][word + "end"] = []
                    for i_dimension in range(self.num_dimensions):
                        self.transition_probs[word + str(i + 1)][word + "end"].append(
                            round(1.0 / all_avg[i_dimension], 3)
                        )

    def check_word(self, evidence_vector, states=True):
        s, p = multidimensional_viterbi(evidence_vector, self.states, self.prior_probs, self.transition_probs, self.emission_paras, num_dimensions=self.num_dimensions)
        if s is None or s == []:
            return s, p
        if states:
            return s
        return s[0][:-1]
