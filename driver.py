import pickle
import math
import cleaner
import tfidf
import embed
import cooccurance
import nearestk
import numpy as np
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
sources = ['breitbart', 'foxnews', 'reuters', 'cnn', 'huffpo']
# size of context for controversy scoring
CONTEXT_SIZE = 15
# size of terms considered for corpus-wide measurements
CORPUS_WIDE_MEASUREMENT_TERMS = 50
# number of words to show in the printed report
NUMBER_OF_WORDS_TO_DISPLAY = 50
# pretrain on Google News
PRETRAIN = True
# align models
ALIGN = True



## PLEASE SEE THE MAIN FUNCTION TO RUN THE PIPELINE ##


def clean(directory_stem, sources):
    return cleaner.clean_corpus(directory_stem, sources)


def significance(source_dictionary):
    significance_dictionary = tfidf.tfidf_corpus(source_dictionary)
    return significance_dictionary

def logodds_calc(source_dictionary):
    co_occurance_matrix_dictionary = cooccurance.find_matrices(source_dictionary)
    significance_dictionary = logodds.logodds_corpus(source_dictionary, co_occurance_matrix_dictionary)
    return significance_dictionary


def controversy(source_dictionary, significance_dictionary):
    significance_list = dict_to_list(significance_dictionary)
    model_dictionary = embed.create_models(source_dictionary, PRETRAIN, ALIGN)
    co_occurance_matrix_dictionary = cooccurance.find_matrices(source_dictionary)
    return nearestk.controversy_dictionary_use_co_occurance(model_dictionary, significance_list,
                                                            co_occurance_matrix_dictionary, CONTEXT_SIZE)


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
                  avg_pol,
                  significance_dictionary,
                  controversy_dictionary,
                  logodds_dictionary):
    significance_list = dict_to_list(significance_dictionary)[:NUMBER_OF_WORDS_TO_DISPLAY]
    controversy_list = dict_to_list(controversy_dictionary)[:NUMBER_OF_WORDS_TO_DISPLAY]
    logodds_list = dict_to_list(logodds_dictionary)[:NUMBER_OF_WORDS_TO_DISPLAY]

    # main score, word 1, score 1,
    with open('summary_{}_{}.csv'.format(sources[0], sources[1]), 'w', encoding='utf-8') as fp:
        print(str(avg_pol[0]), file=fp)
        print_sig_list = [str(avg_pol[1])] + [str(elt[0]) + ',' + str(elt[1]) for elt in significance_list]
        print(*print_sig_list, sep=',', file=fp)
        print_con_list = [str(avg_pol[2])] + [str(elt[0]) + ',' + str(elt[1]) for elt in controversy_list]
        print(*print_con_list, sep=',', file=fp)
        print_con_list = [str(avg_pol[3])] + [str(elt[0]) + ',' + str(elt[1]) for elt in logodds_list]
        print(*print_con_list, sep=',', file=fp)

    plot.y_v_x_scatter('logodds', logodds_dictionary,
                       'controversy', controversy_dictionary, sources, title='Log odds v Controversy')
    plot.y_v_x_scatter('tfidf', significance_dictionary,
                       'controversy', controversy_dictionary, sources, title='Significance v Controversy')
    plot.y_v_x_scatter('tfidf', significance_dictionary,
                       'logodds', logodds_dictionary, sources, title='Significance v Log Odds')

    with open('report_{}_{}'.format(sources[0], sources[1]), 'w', encoding='utf-8') as fp:
        print('Polarization pipeline summary for ' + sources[0] + ' and ' + sources[1] + ':', file=fp)
        models, matrices = [], []
        for source in sources:
            models.append(KeyedVectors.load_word2vec_format('model_{}.txt'.format(source), binary=False))
            matrices.append(pickle.load(open('matrix_{}'.format(source), 'rb')))
        logodds_list = dict_to_list(logodds_dictionary)[:25]
        controversy_list = dict_to_list(controversy_dictionary)[:25]
        significance_list = dict_to_list(significance_dictionary)[:25]
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

    # Log-Odds scoring
    logodds_dictionary = logodds_calc(source_dictionary)

    # Controversy scoring
    controversy_dictionary = controversy(source_dictionary, significance_dictionary)
    significance_dictionary = fixlen(significance_dictionary, controversy_dictionary)
    logodds_dictionary = fixlen(logodds_dictionary, controversy_dictionary)
    print(len(controversy_dictionary))
    print(len(significance_dictionary))
    print(len(logodds_dictionary))

    # r^2 of all terms
    r2_all = corpus_measurements.r_squared(directory_stem, source_dictionary,
                                           significance_dictionary, len(significance_dictionary),
                                           PRETRAIN, ALIGN)

    # r^2 of top X signif
    r2_top_signf = corpus_measurements.r_squared(directory_stem, source_dictionary,
                                                 significance_dictionary, CORPUS_WIDE_MEASUREMENT_TERMS,
                                                 PRETRAIN, ALIGN)
    # r^2 of top x logodds
    r2_top_logod = corpus_measurements.r_squared(directory_stem, source_dictionary,
                                                 logodds_dictionary, CORPUS_WIDE_MEASUREMENT_TERMS,
                                                 PRETRAIN, ALIGN)
    # r^2 of top X controv
    r2_top_contr = corpus_measurements.r_squared(directory_stem, source_dictionary,
                                                 controversy_dictionary, CORPUS_WIDE_MEASUREMENT_TERMS,
                                                 PRETRAIN, ALIGN)

    create_report(sources,
                  [r2_all, r2_top_signf, r2_top_logod, r2_top_contr],
                  significance_dictionary,
                  controversy_dictionary,
                  logodds_dictionary)

    return [r2_all, r2_top_signf, r2_top_logod, r2_top_contr]


if __name__ == '__main__':
    t = time()
    a = pipeline([sources[0], sources[1]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))
    t = time()
    b = pipeline([sources[3], sources[4]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))
    t = time()
    c = pipeline([sources[0], sources[4]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))
    t = time()
    d = pipeline([sources[1], sources[3]])
    print('Time: {} mins'.format(round((time() - t) / 60, 2)))
    print(a)
    print(b)
    print(c)
    print(d)
