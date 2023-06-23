from convert import viterbi, divide, convert


class Words:
    def __init__(self):
        self.all_words = {}
        self.states = []
        self.prior_probs = {}
        self.emission_paras = {}
        self.transition_probs = {}
        self.state_nums = ["1", "2", "3", "end"]

    def update_word(self, input_words):
        for word in input_words:
            if word not in self.all_words:
                self.all_words[word] = []
                for state_num in self.state_nums:
                    self.states.append(word + state_num)
                    self.emission_paras[word + state_num] = (None, None)
                    self.transition_probs[word + state_num] = {}
                    for state_num2 in self.state_nums:
                        self.transition_probs[word + state_num][word + state_num2] = 0
            for vector in input_words[word]:
                self.all_words[word].append(vector)

        # Change all prior probs in self.prior_probs and add new ones
        for word in self.all_words:
            self.prior_probs[word + self.state_nums[0]] = 1.0 / (
                    float(len(self.states)) / len(self.state_nums))  # 1 / number of words
            for state_num in self.state_nums[1:]:
                self.prior_probs[word + state_num] = 0

        for word in input_words:
            result = divide(self.all_words[word])
            result = convert(result)
            for i in range(len(self.state_nums) - 1):
                self.emission_paras[word + self.state_nums[i]] = (result[1][i], result[2][i])
            result = result[0]
            for i in range(len(self.state_nums) - 1):
                avg = 0
                for vector_state in result[i]:
                    avg += len(vector_state)
                avg = avg / len(input_words[word])
                self.transition_probs[word + str(i + 1)][word + str(i + 1)] = 1 - round(1.0 / avg, 3)
                if i < 2:
                    self.transition_probs[word + str(i + 1)][word + str(i + 2)] = round(1.0 / avg, 3)
                else:
                    self.transition_probs[word + str(i + 1)][word + "end"] = round(1.0 / avg, 3)

    def check_word(self, evidence_vector, states=False):
        s, p = viterbi(evidence_vector, self.states, self.prior_probs, self.transition_probs, self.emission_paras)
        if s is None:
            return s, p
        if states:
            return s
        return s[0][:-1]
