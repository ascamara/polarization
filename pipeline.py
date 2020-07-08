import cleaner
import tfidf
import embed
import align
import cooccurance
import plot
import nearestk
import polarization_calculator
import collections
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import os


def preprocessing(directory_stem, list_of_corpora, perform=False):
    list_of_corpora_clean = []
    if perform:
        for corpus in list_of_corpora:
            directory = directory_stem + '\\' + corpus
            list_of_corpora_clean.append(cleaner.clean_folder(directory_stem, directory, corpus))
            # to do - get the files into a folder
    # use pre-defined material
    else:
        for corpus in list_of_corpora:
            list_of_corpora_clean.append('clean_{}'.format(corpus))
    return list_of_corpora_clean


def find_terms(directory_stem, list_of_corpora_clean, perform=False, tfidf_file='tfidf_final'):
    list_of_terms = []
    if perform:
        list_of_terms = tfidf.tfidf_corpora(directory_stem, list_of_corpora_clean)
    # use pre-defined material
    else:
        list_of_terms = tfidf.read_tfidf(tfidf_file)
    list_of_terms = [term for term in list_of_terms if term[0] not in stopwords.words('english')]
    return list_of_terms


def create_models(directory_stem, list_of_corpora_clean, pretrain=True):
    dictionary_of_sentences = embed.create_dictionary_of_sentences(directory_stem, list_of_corpora_clean)
    return embed.create_models(dictionary_of_sentences, pretrain)


def create_models_random(directory_stem, list_of_corpora_clean, pretrain=True):
    dictionary_of_sentences = embed.create_dictionary_of_sentences(directory_stem, list_of_corpora_clean)
    dictionary_of_sentences_random = collections.defaultdict(list)
    list_a, list_b = [], []
    for key, value in dictionary_of_sentences.items():
        for sentence in dictionary_of_sentences[key]:
            # randomize
            val = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            if val >= .5:
                list_a.append(sentence)
            else:
                list_b.append(sentence)
    dictionary_of_sentences_random['a'] = embed.fix_arrays(list_a)
    dictionary_of_sentences_random['b'] = embed.fix_arrays(list_b)
    return embed.create_models(dictionary_of_sentences_random, pretrain)


def read_models(directory_stem, list_of_corpora_clean):
    dictionary_of_models = {}
    for item in list_of_corpora_clean:
        path = 'model_{}.txt'.format(item)
        dictionary_of_models[item] = KeyedVectors.load_word2vec_format(path, binary=False)
    path = 'model_base.txt'
    dictionary_of_models['base'] = KeyedVectors.load_word2vec_format(path, binary=False)
    return dictionary_of_models


def fix_model_dimensions(dictionary_of_models):
    base_embed = 0
    for key, value in dictionary_of_models.items():
        if base_embed == 0:
            base_embed = value
        else:
            dictionary_of_models[key] = align.smart_procrustes_align_gensim(base_embed, value)
    return dictionary_of_models


def without_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


if __name__ == '__main__':
    print("Polarization Pipeline")

    directory_stem = r'C:\Users\ascam\PycharmProjects\polarizat'
    list_of_corpora = ['fox', 'nbc']

    # PART 1: PREPROCESSING
    perform_preprocessing = False
    list_of_corpora_clean = preprocessing(directory_stem,
                                          list_of_corpora,
                                          perform_preprocessing)

    # PART 2: TF-IDF on the SUBCORPUSES
    perform_tfidf = False
    list_of_terms = find_terms(directory_stem,
                               list_of_corpora_clean,
                               perform_tfidf,
                               tfidf_file='tfidf_final')

    # PART 3: CREATE and TRAIN MODELS
    pretrain = True
    generate_new_models = False
    if generate_new_models:
        dictionary_of_models = create_models(directory_stem,
                                             list_of_corpora_clean,
                                             pretrain)
    else:
        dictionary_of_models = read_models(directory_stem,
                                           list_of_corpora_clean)

    # PART 4 (Optional): CREATE CO-OCCURENCE MATRIX FOR EACH DICTIONARY
    models_excl_base = without_keys(dictionary_of_models, 'base')
    perform_cocurrence = False
    perform_matrix_generation = False
    if perform_cocurrence and perform_matrix_generation:
        dictionary_of_co_occurence = cooccurance.create_co_occurence(directory_stem,
                                                                     models_excl_base,
                                                                     list_of_terms)
    elif perform_cocurrence and not perform_matrix_generation:
        dictionary_of_co_occurence = cooccurance.read_co_occurance(models_excl_base)

    # PART 5: FIND R SQUARED
    polarization_calculator.driver(dictionary_of_models,
                                   list_of_terms,
                                   k=50,
                                   use_pretrained=False)

    # PART 6: RUN RANDOMIZED PIPELINES
    pretrain = True
    generate_new_models = True
    if generate_new_models:
        dictionary_of_models = create_models_random(directory_stem,
                                                    list_of_corpora_clean,
                                                    pretrain)
    else:
        list_of_random_models = ['a', 'b']
        dictionary_of_models = read_models(directory_stem,
                                           list_of_random_models)
    # PART 7 (Optional): CREATE CO-OCCURENCE MATRIX FOR RANDOMIZED
    # PART 8: FIND R SQUARED
    polarization_calculator.driver(dictionary_of_models,
                                   list_of_terms,
                                   k=50,
                                   use_pretrained=False,
                                   is_random=True)

