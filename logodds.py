from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import cleaner
import math
from operator import itemgetter


def create_set(list_of_sentences):
    word_set = set()
    for sentence in tqdm(list_of_sentences, ascii=True, desc='Making set', leave=True):
        word_set.update([w for w in sentence])
    return sorted(word_set)


def word_in_corpus(list_of_sentences, word):
    sum = 0
    for sentence in list_of_sentences:
        sum += sentence.count(word)
    return sum


def source_frequency(source_dictionary):
    frequency_dictionary = {}
    for key, value in source_dictionary.items():
        frequency_dictionary_source = defaultdict(int)
        sentences = [sentence for document in value for sentence in document]
        list_of_sentences = [sentence.split() for sentence in sentences]
        list_of_words = create_set(list_of_sentences)
        if os.path.isfile('incidents_{}'.format(key)):
            frequency_dictionary_source = pickle.load(open('incidents_{}'.format(key), 'rb'))
        else:
            for word in tqdm(list_of_words, ascii=True, desc='Creating dictionary', leave=True):
                count = word_in_corpus(list_of_sentences, word)
                if count > 10:
                    frequency_dictionary_source[word] = count
            with open('incidents_{}'.format(key), 'wb') as fp:
                pickle.dump(frequency_dictionary_source, fp)
        frequency_dictionary[key] = frequency_dictionary_source
    return frequency_dictionary


def source_probability(source_dictionary):
    frequency_dictionary = {}
    for key, value in source_dictionary.items():
        frequency_dictionary_source = defaultdict(int)
        sentences = [sentence for document in value for sentence in document]
        list_of_sentences = [sentence.split() for sentence in sentences]
        list_of_words = create_set(list_of_sentences)
        if os.path.isfile('probability_{}'.format(key)):
            frequency_dictionary_source = pickle.load(open('probability_{}'.format(key), 'rb'))
        else:
            total = 0
            for word in tqdm(list_of_words, ascii=True, desc='Creating dictionary', leave=True):
                total += word_in_corpus(list_of_sentences, word)
            for word in tqdm(list_of_words, ascii=True, desc='Creating dictionary', leave=True):
                count = word_in_corpus(list_of_sentences, word)
                if count > 10:
                    frequency_dictionary_source[word] = count / total
            with open('probability_{}'.format(key), 'wb') as fp:
                pickle.dump(frequency_dictionary_source, fp)
                print(frequency_dictionary_source)
        frequency_dictionary[key] = frequency_dictionary_source
    return frequency_dictionary


def fix_dict(base_dict, partner_dict):
    list_of_keys = set([*partner_dict]).intersection([*base_dict])
    base = {k: v for k, v in base_dict.items() if k in list_of_keys}
    partner = {k: v for k, v in partner_dict.items() if k in list_of_keys}
    return base, partner


def include_keys(dict, keys):
    return {k: v for k, v in dict.items() if k in keys}


def dict_to_list(dict):
    list = []
    for key, value in dict.items():
        temp_pair = [key, value]
        list.append(temp_pair)
    list = sorted(list, key=itemgetter(1))
    list = list[::-1]
    return list

def prob(fq1, fq2, k):
    return fq1[k] / (fq1[k] + fq2[k])

def logodds_corpus(source_dictionary):
    frequency_dictionary = source_probability(source_dictionary)
    list_of_sources = [key for key in frequency_dictionary]
    candidate_keys_1 = set([key for key, value in frequency_dictionary[list_of_sources[0]].items() if value > 100])
    candidate_keys_2 = set([key for key, value in frequency_dictionary[list_of_sources[1]].items() if value > 100])
    list_of_keys = [*candidate_keys_1.intersection(candidate_keys_2)]
    fc0 = {k: v for k, v in frequency_dictionary[list_of_sources[0]].items() if k in list_of_keys}
    fc1 = {k: v for k, v in frequency_dictionary[list_of_sources[1]].items() if k in list_of_keys}

    # return {k: math.log(((freq_count_0[k] / freq_count_0[k] + freq_count_1[k])
    #                    /(1-(freq_count_0[k] / freq_count_0[k] + freq_count_1[k])))
    #                    /((freq_count_1[k] / freq_count_0[k] + freq_count_1[k])
    #                    /(1-(freq_count_1[k] / freq_count_0[k] + freq_count_1[k]))))
    #        for k in list_of_keys}
    test = {k: math.log(prob(fc0, fc1, k) / (1-prob(fc0, fc1, k)))
            + math.log(prob(fc1, fc0, k) / (1-prob(fc1, fc0, k))) for k in list_of_keys}
    assert False
    return {k: math.log((freq_count_0[k] / (freq_count_0[k] + freq_count_1[k]))
                        / (1 - (freq_count_0[k] / (freq_count_0[k] + freq_count_1[k])))) +
               math.log((freq_count_1[k] / (freq_count_0[k] + freq_count_1[k]))
                        / (1 - (freq_count_1[k] / (freq_count_0[k] + freq_count_1[k])))) for k in list_of_keys}
