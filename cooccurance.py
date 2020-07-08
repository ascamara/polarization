import json
from collections import OrderedDict


def create_co_occurence(directory_stem, dictionary_of_models, list_of_terms):
    dictionary_of_co_occurence = {}
    terms = [terms[0] for terms in list_of_terms[:len(list_of_terms) // 25]]
    for key, value in dictionary_of_models.items():
        stem = directory_stem + '\corpora_{}.csv'
        corpus = key
        print("Working on matrix for: " + key)
        df = pd.read_csv(stem.format(corpus), sep=',')
        df = df.dropna().reset_index(drop=True)
        corpus_arr = [sentences.split() for sentences in df['lines']]
        corpus_set = set()
        for sentence in corpus_arr:
            corpus_set.update(sentence)
        occurrences = OrderedDict((term, OrderedDict((term, 0) for term in corpus_set)) for term in corpus_set)
        j = 0
        for sentence in corpus_arr:
            j = j + 1
            print("progress: {} / {}".format(j, len(corpus_arr)))
            for i in range(len(sentence)):
                for item in sentence[:i] + sentence[i + 1:]:
                    # if item in terms and sentence[i] in terms:
                    occurrences[sentence[i]][item] += 1
        dictionary_of_co_occurence[key] = occurrences
        with open('matrix_{}.json'.format(key), 'w') as fp:
            print("writing: {}".format(key))
            fp.write(json.dumps(dictionary_of_co_occurence[key]))
    return dictionary_of_co_occurence


def read_co_occurance(dictionary_of_models):
    dictionary_of_co_occurence = {}
    for key, value in dictionary_of_models.items():
        with open('matrix_{}.json'.format(key), 'r') as fp:
            dictionary_of_co_occurence[key] = json.loads(fp.read())
    return dictionary_of_co_occurence
