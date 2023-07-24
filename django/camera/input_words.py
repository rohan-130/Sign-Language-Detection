from rest_framework import serializers


def convert_vector_to_int_list(vector, count):
    for i in range(count):
        vector[i] = int(vector[i])
    return vector


class Input:
    def __init__(self, wordName, count, vector):
        self.wordName = wordName
        self.count = count
        self.vector = convert_vector_to_int_list(vector)


class WordsSerializer(serializers.Serializer):
    # all_words = serializers.DictField()
    states = serializers.ListField()
    prior_probs = serializers.DictField()
    emission_paras = serializers.DictField()
    transition_probs = serializers.DictField()
    state_nums = serializers.ListField()
    num_dimensions = serializers.IntegerField()


class InputSerializer(serializers.Serializer):
    wordName = serializers.StringRelatedField()
    count = serializers.IntegerField()
    vector = serializers.ListField()



