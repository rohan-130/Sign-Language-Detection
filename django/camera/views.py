from django.shortcuts import render
import pickle
from rest_framework import status
from rest_framework.response import Response
from .input_words import WordsSerializer, InputSerializer
from rest_framework.decorators import api_view
from django.conf import settings
from .words import Words


'''
words = Words()
with open('/Users/rohan/Dropbox (GaTech)/sign-language-recognition/asl_recognition/words.pkl', 'wb') as file:
    pickle.dump(words, file)'''


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
    print(vector)
    print("WORD:", result)
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
    success = 'success'
    try:
        words.update_word({word_name: [vector]})
        success = word_name + " trained " + str(len(words.all_words[word_name])) + " times"
        with open(str(settings.BASE_DIR) + '/words.pkl', 'wb') as file:
            pickle.dump(words, file)
    except Exception as e:
        print(e)
        success = 'error'
    return Response({'success': success}, status=status.HTTP_200_OK)

