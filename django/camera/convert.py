import statistics
import numpy as np
import math


def gaussian_prob(x, para_tuple):
    if list(para_tuple) == [None, None]:
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std ** 2) ** -0.5 * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    return gaussian_percentile


def viterbi(evidence_vector, states, prior_probs, transition_probs, emission_paras):
    sequence = []
    probability = 0.0

    if len(evidence_vector) == 0:
        return sequence, probability

    nl = []

    for i in range(len(states)):
        nl.append([prior_probs[states[i]] * gaussian_prob(evidence_vector[0], emission_paras[states[i]])] + [0] * (
                len(evidence_vector) - 1))

    for i in range(1, len(evidence_vector)):
        for j in range(len(states)):
            state = states[j]
            if j >= 1 and i >= 1:
                max_val = -1
                best_prev_prob = None
                k_new = None
                for k in range(len(states)):
                    if states[j] in transition_probs[states[k]] and nl[k][i - 1] * \
                            transition_probs[states[k]][states[j]] > max_val:
                        max_val = nl[k][i - 1] * transition_probs[states[k]][states[j]]
                        best_prev_prob = nl[k][i - 1]
                        k_new = k
                prev_prob = best_prev_prob
                prev_state = states[k_new]
            elif i >= 1:
                prev_prob = nl[j][i - 1]
                prev_state = states[j]
            a = gaussian_prob(evidence_vector[i], emission_paras[state])
            nl[j][i] = prev_prob * a * transition_probs[prev_state][state]

    highest_prob = -1
    for j in range(len(states)):
        if highest_prob < nl[j][-1]:
            highest_prob = nl[j][-1]
            highest_prob_index = j

    sequence.append(states[highest_prob_index])
    probability = highest_prob

    for i in range(len(evidence_vector) - 2, -1, -1):
        highest_prob = -1

        for j in range(len(states)):
            if sequence[0] not in transition_probs[states[j]]:
                continue
            if nl[j][i] * transition_probs[states[j]][sequence[0]] > highest_prob:
                highest_prob = nl[j][i] * transition_probs[states[j]][sequence[0]]
                best_state = states[j]
        if best_state is not None:
            sequence = [best_state] + sequence

    if probability == 0:
        return None, 0

    return sequence, probability


def multidimensional_viterbi(evidence_vector, states, prior_probs, transition_probs, emission_paras, num_dimensions=2):
    sequence = []
    probability = 0.0

    if len(evidence_vector) == 0:
        return sequence, probability

    nl = []

    for i in range(len(states)):
        prior_prob = prior_probs[states[i]]
        prior_prob = math.log(prior_prob) if prior_prob != 0 else -math.inf
        gaussian_prob_all = []
        for z in range(num_dimensions):
            gaussian_prob_all.append(gaussian_prob(evidence_vector[0][z], emission_paras[states[i]][z]))
            gaussian_prob_all[z] = math.log(gaussian_prob_all[z]) if gaussian_prob_all[z] != 0 else -math.inf
        gaussian_prob_sum = 0
        for z in range(num_dimensions):
            gaussian_prob_sum += gaussian_prob_all[z]
        nl.append([prior_prob + gaussian_prob_sum] + [0] * (len(evidence_vector) - 1))

    for i in range(1, len(evidence_vector)):
        for j in range(len(states)):
            state = states[j]
            if j >= 1 and i >= 1:
                max_val = -math.inf
                best_prev_prob = None
                k_new = None
                for k in range(len(states)):
                    transition_prob_sum = 0
                    try:
                        transition_prob_all = []
                        for z in range(num_dimensions):
                            transition_prob_all.append(transition_probs[states[k]][states[j]][z])
                            transition_prob_all[z] = math.log(transition_prob_all[z]) if transition_prob_all[z] != 0 else -math.inf
                            transition_prob_sum += transition_prob_all[z]
                    except:
                        pass
                    if states[j] in transition_probs[states[k]] and (
                            nl[k][i - 1] + transition_prob_sum) >= max_val:
                        max_val = nl[k][i - 1] + transition_prob_sum
                        best_prev_prob = nl[k][i - 1]
                        k_new = k
                prev_prob = best_prev_prob
                prev_state = states[k_new]
            elif i >= 1:
                prev_prob = nl[j][i - 1]
                prev_state = states[j]
            gaussian_all = []
            gaussian_sum = 0
            for z in range(num_dimensions):
                gaussian_all.append(gaussian_prob(evidence_vector[i][z], emission_paras[state][z]))
                gaussian_all[z] = math.log(gaussian_all[z]) if gaussian_all[z] != 0 else -math.inf
                gaussian_sum += gaussian_all[z]
            transition_all = []
            transition_sum = 0
            for z in range(num_dimensions):
                transition_all.append(transition_probs[prev_state][state][z])
                transition_all[z] = math.log(transition_all[z]) if transition_all[z] != 0 else -math.inf
                transition_sum += transition_all[z]
            nl[j][i] = prev_prob + gaussian_sum + transition_sum
    new_s = []
    seq = []

    highest_prob = -math.inf
    highest_prob_index = None
    for j in range(len(states)):
        if highest_prob < nl[j][-1]:
            highest_prob = nl[j][-1]
            highest_prob_index = j
    new_s.append(highest_prob)
    sequence.append(states[highest_prob_index])
    probability = highest_prob

    for i in range(len(evidence_vector) - 2, -1, -1):
        highest_prob = -math.inf
        new_highest_prob = -math.inf
        best_state = None
        nj = None
        ni = None

        for j in range(len(states)):
            if sequence[0] not in transition_probs[states[j]]:
                continue
            transition_all = []
            transition_sum = 0
            for z in range(num_dimensions):
                transition_all.append(transition_probs[states[j]][sequence[0]][1])
                transition_all[z] = math.log(transition_all[z]) if transition_all[z] != 0 else -math.inf
                transition_sum += transition_all[z]
            if (nl[j][i] + transition_sum) > highest_prob:
                highest_prob = nl[j][i] + transition_sum
                new_highest_prob = nl[j][i]
                best_state = states[j]
                nj = j
                ni = i
        if best_state:
            sequence = [best_state] + sequence
            new_s = [new_highest_prob] + new_s
            seq = seq + [(nj, ni)]

    if probability == 0:
        return None, 0

    return sequence, math.e**probability


def average(a):
    avg = 0
    count = 0
    for i in a:
        for j in i:
            avg += j
            count += 1
    avg /= count
    return round(avg, 3)


def sd1(a):
    new_list = []
    for i in a:
        for j in i:
            new_list.append(j)
    return round(statistics.pstdev(new_list), 3)


def convert(a, prev_avg=None, prev_sd=None, iterations=0):
    avg = [average(a_i) for a_i in a]
    sd = [sd1(a_i) for a_i in a]

    if prev_avg == avg and prev_sd == sd:
        return a, avg, sd, iterations

    for index in range(len(a) - 1):
        for i in range(len(a[index])):
            boundary = 0
            for n in a[index][i][::-1]:
                if sd[index] == 0 and sd[index + 1] != 0:
                    boundary += 1
                elif sd[index + 1] == 0 and sd[index] != 0:
                    break
                elif sd[index + 1] == 0 and sd[index] == 0:
                    break
                elif (abs(n - avg[index]) / sd[index]) > (abs(n - avg[index + 1]) / sd[index + 1]):
                    boundary += 1
                else:
                    break
            if boundary > 0:
                for z in range(boundary):
                    if len(a[index][i]) > 1:
                        a[index + 1][i] = a[index][i][(len(a[index][i]) - 1):] + a[index + 1][i]
                        a[index][i] = a[index][i][:(len(a[index][i]) - 1)]
            if boundary == 0:
                for n in a[index + 1][i][:]:
                    if sd[index] == 0 and sd[index + 1] != 0:
                        boundary += 1
                    elif sd[index + 1] == 0 and sd[index] != 0:
                        break
                    elif sd[index + 1] == 0 and sd[index] == 0:
                        break
                    elif (abs(n - avg[index + 1]) / sd[index + 1]) > (abs(n - avg[index]) / sd[index]):
                        boundary += 1
                    else:
                        break
                if boundary > 0:
                    for z in range(boundary):
                        if len(a[index + 1][i]) > 1:
                            a[index][i] = a[index][i] + a[index + 1][i][:1]
                            a[index + 1][i] = a[index + 1][i][1:]

    return convert(a, prev_avg=avg, prev_sd=sd, iterations=iterations + 1)


def divide(evidence_vectors):
    result = [[], [], []]

    for vector in evidence_vectors:
        index_1 = int(len(vector) / 3)
        index_2 = index_1 + int(len(vector) / 3)
        result[0] += [vector[:index_1]]
        result[1] += [vector[index_1:index_2]]
        result[2] += [vector[index_2:]]
    return result
