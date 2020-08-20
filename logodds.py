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
    probability_dictionary = {}
    for key, value in source_dictionary.items():
        probability_dictionary_source = defaultdict(float)
        sentences = [sentence for document in value for sentence in document]
        list_of_sentences = [sentence.split() for sentence in sentences]
        list_of_words = create_set(list_of_sentences)
        # sentences = [sentence for document in documents for sentence in document]
        total = len([w for sentence in list_of_sentences for w in sentence])
        if os.path.isfile('probability_{}'.format(key)):
            probability_dictionary_source = pickle.load(open('probability_{}'.format(key), 'rb'))
        else:
            if os.path.isfile('incidents_{}'.format(key)):
                frequency_dictionary_source = pickle.load(open('incidents_{}'.format(key), 'rb'))
                for word in tqdm(list_of_words, ascii=True, desc='Creating dictionary', leave=True):
                    if frequency_dictionary_source[word] > 50:
                        probability_dictionary_source[word] = frequency_dictionary_source[word] / total
            else:
                for word in tqdm(list_of_words, ascii=True, desc='Creating dictionary', leave=True):
                    count = word_in_corpus(list_of_sentences, word)
                    if count > 50:
                        probability_dictionary_source[word] = count / total
            with open('probability_{}'.format(key), 'wb') as fp:
                pickle.dump(probability_dictionary_source, fp)
        probability_dictionary[key] = probability_dictionary_source
    return probability_dictionary


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


def prob(fq0, fq1, k):
    return fq0[k] / (fq0[k] + fq1[k])


def logodds_corpus(source_dictionary, co_occurance_matrix_dictionary):
    frequency_dictionary = source_probability(source_dictionary)
    list_of_sources = [key for key in frequency_dictionary]
    candidate_keys_1 = set([key for key, value in frequency_dictionary[list_of_sources[0]].items()])
    matrix_keys_1 = set([k for k in co_occurance_matrix_dictionary[list_of_sources[0]]])
    candidate_keys_2 = set([key for key, value in frequency_dictionary[list_of_sources[1]].items()])
    matrix_keys_2 = set([k for k in co_occurance_matrix_dictionary[list_of_sources[1]]])
    list_of_keys = [*candidate_keys_1.intersection(candidate_keys_2)]
    list_of_matrix_terms = [*matrix_keys_1.intersection(matrix_keys_2)]
    fq0 = {k: v for k, v in frequency_dictionary[list_of_sources[0]].items() if k in list_of_keys
           and list_of_matrix_terms}
    fq1 = {k: v for k, v in frequency_dictionary[list_of_sources[1]].items() if k in list_of_keys
           and list_of_matrix_terms}

    return {k: math.log(prob(fq0, fq1, k) / (1 - prob(fq0, fq1, k)))
               + math.log(prob(fq1, fq0, k) / (1 - prob(fq1, fq0, k))) for k in list_of_keys}
