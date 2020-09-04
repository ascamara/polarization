import pickle
import math
import cleaner
import tfidf
import embed
import cooccurance
import nearestk
import numpy as np
import pandas as pd
import logodds
import corpus_measurements
import plot
from collections import defaultdict
from operator import itemgetter
from time import time
from gensim.models import KeyedVectors

# Polarization Pipeline

# folder location of the data
directory_stem = r'C:\Users\ascam\PycharmProjects\polarization_pipeline'
# sources to be considered
sources = ['breitbart', 'foxnews', 'cnn', 'huffpo']
# size of context for controversy scoring
CONTEXT_SIZE = 20
# are your documents different files or unique lines in a/some file(s)
DOCUMENT_AT_LINE_LEVEL = False
# pretrain on Google News
PRETRAIN = True
# align models
ALIGN = True


## PLEASE SEE THE MAIN FUNCTION TO RUN THE PIPELINE ##


def clean(directory_stem, sources):
    return cleaner.bclean_corpus(directory_stem, sources)


def significance(source_dictionary):
    significance_dictionary = tfidf.tfidf_corpus(source_dictionary, DOCUMENT_AT_LINE_LEVEL)
    return significance_dictionary


def matrix_generator(source_dictionary):
    return cooccurance.find_matrices(source_dictionary)


def logodds_calc(source_dictionary, matrix_dictionary):
    significance_dictionary = logodds.logodds_corpus(source_dictionary, matrix_dictionary)
    return significance_dictionary


def controversy(significance_dictionary, matrix_dictionary, model_dictionary):
    significance_list = dict_to_list(significance_dictionary)
    return nearestk.controversy_dictionary_use_co_occurance(model_dictionary, significance_list,
                                                            matrix_dictionary, CONTEXT_SIZE)


def dict_to_list(dict):
    list = []
    for key, value in dict.items():
        temp_pair = [key, value]
        list.append(temp_pair)
    list = sorted(list, key=itemgetter(1))
    list = list[::-1]
    return list


def list_to_txt(sources, list_, txt_name="undef"):
    with open('{}_{}_{}'.format(txt_name, sources[0], sources[1]), 'w', encoding='utf-8') as fp:
        print(*list_, sep='\n', file=fp)


def create_report(sources,
                  significance_dictionary,
                  controversy_dictionary,
                  logodds_dictionary,
                  matrix_dictionary,
                  model_dictionary):
    # term, controversy, tdidf, logodds
    controversy_keys = set([k for k in controversy_dictionary])
    significance_keys = set([k for k in significance_dictionary])
    logodds_keys = set([k for k in logodds_dictionary])
    all_keys = [*controversy_keys.union(significance_keys, logodds_keys)]
    list_of_tuples = []
    for elt in all_keys:
        list_of_tuples.append((elt,
                               controversy_dictionary.get(elt, np.nan),
                               significance_dictionary.get(elt, np.nan),
                               logodds_dictionary.get(elt, np.nan)))
    df = pd.DataFrame(list_of_tuples, columns=['term', 'controversy', 'tfidf', 'logodds'])
    df.to_csv('summary_{}_{}.csv'.format(sources[0], sources[1]))

    # plot.y_v_x_scatter('logodds', logodds_dictionary,
    #                   'controversy', controversy_dictionary, sources, title='Log odds v Controversy')
    # plot.y_v_x_scatter('tfidf', significance_dictionary,
    #                   'controversy', controversy_dictionary, sources, title='Significance v Controversy')
    # plot.y_v_x_scatter('tfidf', significance_dictionary,
    #                   'logodds', logodds_dictionary, sources, title='Significance v Log Odds')

    significance_list = dict_to_list(significance_dictionary)[:1000]
    controversy_list = dict_to_list(controversy_dictionary)
    logodds_list = dict_to_list(logodds_dictionary)[:1000]

    with open('report_{}_{}'.format(sources[0], sources[1]), 'w', encoding='utf-8') as fp:
        print('Polarization pipeline summary for ' + sources[0] + ' and ' + sources[1] + ':', file=fp)
        matrices, models = [], []
        for key, value in matrix_dictionary.items():
            matrices.append(value)
            models.append(model_dictionary[key])
        print('--- Controversy List ---', file=fp)
        for term in controversy_list:
            print(term, file=fp)
            print_list = nearestk.return_contexts(models, matrices, term, CONTEXT_SIZE)
            print('Context in both: ' + ', '.join(print_list[0]), file=fp)
            print('Context in ' + sources[0] + ": " + ', '.join(print_list[1]), file=fp)
            print('Context in ' + sources[1] + ": " + ', '.join(print_list[2]), file=fp)
        print('--- Significance List ---', file=fp)
        for term in significance_list:
            print(term, file=fp)
            print_list = nearestk.return_contexts(models, matrices, term, CONTEXT_SIZE)
            print('Context in both: ' + ', '.join(print_list[0]), file=fp)
            print('Context in ' + sources[0] + ": " + ', '.join(print_list[1]), file=fp)
            print('Context in ' + sources[1] + ": " + ', '.join(print_list[2]), file=fp)
        print('--- Log Odds List ---', file=fp)
        for term in logodds_list:
            print(term, file=fp)
            print_list = nearestk.return_contexts(models, matrices, term, CONTEXT_SIZE)
            print('Context in both: ' + ', '.join(print_list[0]), file=fp)
            print('Context in ' + sources[0] + ": " + ', '.join(print_list[1]), file=fp)
            print('Context in ' + sources[1] + ": " + ', '.join(print_list[2]), file=fp)

    return -1


def fixlen(significance_dictionary, controversy_dictionary):
    n = len(controversy_dictionary)
    significance_list = dict_to_list(significance_dictionary)[:n]
    significance_list = sorted(significance_list, key=itemgetter(1))
    significance_list = significance_list[::-1]
    adj_significance_dictionary = {}
    for element in significance_list:
        adj_significance_dictionary[element[0]] = element[1]
    return adj_significance_dictionary


def pipeline(sources):
    # Cleaning
    source_dictionary = clean(directory_stem, sources)

    # Significance scoring
    significance_dictionary = significance(source_dictionary)
    print(len(significance_dictionary))

    # Co-occurance matrix generation
    matrix_dictionary = matrix_generator(source_dictionary)
    model_dictionary = embed.create_models(source_dictionary, PRETRAIN, ALIGN)

    # Log-Odds scoring
    logodds_dictionary = logodds_calc(source_dictionary, matrix_dictionary)
    print(len(logodds_dictionary))

    # Controversy scoring
    controversy_dictionary = controversy(significance_dictionary, matrix_dictionary, model_dictionary)
    print(len(controversy_dictionary))

    '''
    significance_dictionary = fixlen(significance_dictionary, controversy_dictionary)
    logodds_dictionary = fixlen(logodds_dictionary, controversy_dictionary)
    print(len(controversy_dictionary))
    print(len(significance_dictionary))
    print(len(logodds_dictionary))
    '''

    create_report(sources,
                  significance_dictionary,
                  controversy_dictionary,
                  logodds_dictionary,
                  matrix_dictionary,
                  model_dictionary)

    return -1


if __name__ == '__main__':
    # ['breitbart', 'foxnews', 'cnn', 'huffpo']
    t = time()
    c = pipeline([sources[1], sources[2]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    a = pipeline([sources[0], sources[1]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    b = pipeline([sources[2], sources[3]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))

    print(a)
    print(b)
    print(c)

