from django.shortcuts import render
import pickle
from rest_framework import status
from rest_framework.response import Response
from .input_words import WordsSerializer, InputSerializer, Input
from rest_framework.decorators import api_view
from django.conf import settings
from .words import Words


def test_cam(request):
    return render(request, 'new_test.html')


def train_cam(request):
    return render(request, 'new_train.html')


@api_view(('GET',))
def get_all_words(request):
    with open(str(settings.BASE_DIR) + '/words.pkl', 'rb') as file:
        words = pickle.load(file)
    return Response(WordsSerializer(words).data, status=status.HTTP_200_OK)


@api_view(('GET',))
def check_word(request):
    with open(str(settings.BASE_DIR)+'/words.pkl', 'rb') as file:
        words = pickle.load(file)
    count = int(request.GET.get('count'))
    vector = []
    for i in range(count):
        coordinates = request.GET.getlist("vector["+str(i)+"][]")
        vector.append((float(coordinates[0]), float(coordinates[1])))
    vector = vector[:-1]
    result = words.check_word(evidence_vector=vector, states=False)
    return Response({'word': result}, status=status.HTTP_200_OK)


@api_view(('POST',))
def train_word(request):
    with open(str(settings.BASE_DIR) + '/words.pkl', 'rb') as file:
        words = pickle.load(file)
    word_name = request.data.get('wordName')
    count = int(request.data.get('count'))
    vector = []
    for i in range(count):
        coordinates = request.data.getlist("vector[" + str(i) + "][]")
        vector.append((float(coordinates[0]), float(coordinates[1])))
    vector = vector[:-1]
    best_state_nums = 0
    try:
        if word_name not in words.all_words:
            highest_prob = -1
            best_state_nums = None
            for num_states in range(3, 7):
                state_nums = []
                for i in range(num_states):
                    state_nums.append(str(i + 1))
                state_nums.append("end")
                prob = check_new_probability(word_name, vector, num_states)
                if prob > highest_prob:
                    highest_prob = prob
                    best_state_nums = state_nums
            if highest_prob == -1:
                raise Exception("too similar to another word")
            words.update_word({word_name: [vector]}, best_state_nums)
        else:
            words.update_word({word_name: [vector]})
        success = word_name + " trained " + str(len(words.all_words[word_name])) + " times" + " " + str(best_state_nums)
        with open(str(settings.BASE_DIR) + '/words.pkl', 'wb') as file:
            pickle.dump(words, file)
    except Exception as e:
        print(e)
        success = 'error'
    return Response({'success': success}, status=status.HTTP_200_OK)


def check_new_probability(word_name, vector, num_states):
    with open(str(settings.BASE_DIR) + '/words.pkl', 'rb') as file:
        words = pickle.load(file)
    state_nums = []
    avg_prob = 0
    for i in range(num_states):
        state_nums.append(str(i+1))
    state_nums.append("end")
    words.update_word({word_name: [vector]}, state_nums)
    for word in words.all_words:
        w, p = words.check_word(words.all_words[word][0], prob=True)
        if w == word:
            avg_prob += p
        else:
            avg_prob = -1
            break
    return avg_prob
