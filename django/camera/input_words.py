from rest_framework import serializers

class Input:
    def __init__(self, vector, count):
        self.vector = vector
        self.count = count


class WordsSerializer(serializers.Serializer):
    all_words = serializers.DictField()
    states = serializers.ListField()
    prior_probs = serializers.DictField()
    emission_paras = serializers.DictField()
    transition_probs = serializers.DictField()
    state_nums = serializers.ListField()
    num_dimensions = serializers.IntegerField()

class InputSerializer(serializers.Serializer):
    vector = serializers.ListField()
    count = serializers.IntegerField()
