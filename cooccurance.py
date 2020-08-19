import pickle
import os
from collections import OrderedDict
from operator import itemgetter
from tqdm import tqdm
from nltk.corpus import stopwords

def dict_to_list(dict):
    list = []
    #tqdm here?
    for key, value in dict.items():
        temp_pair = [key, value]
        list.append(temp_pair)
    list = sorted(list, key=itemgetter(1))
    list = list[::-1]
    return list


def find_matrices(source_dictionary):
    sw = stopwords.words("english")
    matrix_dictionary = {}
    #tqdm
    for key, value in tqdm(source_dictionary.items(), ascii=True, desc='Loading matrices'):
        if os.path.isfile('matrix_{}'.format(key)):
            temp = pickle.load(open('matrix_{}'.format(key), 'rb'))
            matrix_dictionary[key] = temp
        else:
            # use value to make a co-occurance matrix
            sentences = [sentence for document in value for sentence in document]
            corpus_arr = [sentence.split() for sentence in sentences]

            significance_dictionary = pickle.load(open('tfidf_{}'.format(key), 'rb'))
            significance_list = dict_to_list(significance_dictionary)
            list_not_found = []

            corpus_set = [term[0] for term in significance_list[:len(significance_list) // 10]]

            occurrences = OrderedDict((term, OrderedDict((term, 0) for term in corpus_set)) for term in corpus_set)
            j = 0
            for sentence in tqdm(corpus_arr, ascii=True, desc='Creating matrices'):
                j = j + 1
                for i in range(len(sentence)):
                    for item in sentence[:i] + sentence[i + 1:]:
                        # if item in terms and sentence[i] in terms:
                        try:
                            ### good place to kill words like 'and'
                            if item not in sw:
                                occurrences[sentence[i]][item] += 1
                        except KeyError:
                            list_not_found.append(item)
            matrix_dictionary[key] = occurrences
            with open('matrix_{}'.format(key), 'wb') as fp:
                pickle.dump(occurrences, fp)
    return matrix_dictionary
