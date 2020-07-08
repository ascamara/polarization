import numpy as np
import matplotlib as pyplot
from gensim.models import Word2Vec
import operator
from time import time
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import math


def find_controversy_2(models,
                       anchor_word,
                       k,
                       use_pretrained):
    # build contexts
    context_a, context_b = [], []
    model_a, model_b = models[0], models[1]
    pretrain = models[-1]
    if use_pretrained:
        similarity_a = model_a.wv.most_similar(positive=[anchor_word],
                                               topn=k,
                                               restrict_vocab=len(model_a.wv.vocab) // 4)
        temp_dict_a = [term[0] for term in similarity_a]
        context_a = [term for term in temp_dict_a if term in pretrain.wv.vocab]
        similarity_b = model_b.wv.most_similar(positive=[anchor_word],
                                               topn=k,
                                               restrict_vocab=len(model_b.wv.vocab) // 4)
        temp_dict_b = [term[0] for term in similarity_b]
        context_b = [term for term in temp_dict_b if term in pretrain.wv.vocab]
    else:
        similarity_a = model_a.wv.most_similar(positive=[anchor_word],
                                               topn=k,
                                               restrict_vocab=len(model_a.wv.vocab) // 4)
        temp_dict_a = [term[0] for term in similarity_a]
        context_a = [term for term in temp_dict_a if term in model_a.wv.vocab]
        similarity_b = model_b.wv.most_similar(positive=[anchor_word],
                                               topn=k,
                                               restrict_vocab=len(model_b.wv.vocab) // 4)
        temp_dict_b = [term[0] for term in similarity_b]
        context_b = [term for term in temp_dict_b if term in model_b.wv.vocab]

    if use_pretrained:
        context_a_vec = np.array([pretrain.wv[term] for term in context_a])
        # print(len(context_a_vec))  # should be 25
        context_b_vec = np.array([pretrain.wv[term] for term in context_b])
        # print(len(context_b_vec))  # should be 25
        all_vec = np.concatenate((context_a_vec, context_b_vec))
        # print(len(all_vec))  # should be 50?
        a_term = len(context_a) * (np.linalg.norm(np.mean(context_a_vec, axis=0) - np.mean(all_vec, axis=0)) ** 2)
        b_term = len(context_b) * (np.linalg.norm(np.mean(context_b_vec, axis=0) - np.mean(all_vec, axis=0)) ** 2)
        denom_terms = []
        for vec in all_vec:
            denom_terms.append(np.linalg.norm(vec - np.mean(all_vec, axis=0)) ** 2)
        # return value
        return (a_term + b_term) / sum(denom_terms)
    else:
        context_a_vec = np.array([model_a.wv[term] for term in context_a])
        # print(len(context_a_vec))  # should be 25
        context_b_vec = np.array([model_b.wv[term] for term in context_b])
        # print(len(context_b_vec))  # should be 25
        all_vec = np.concatenate((context_a_vec, context_b_vec))
        # print(len(all_vec))  # should be 50?
        a_term = len(context_a) * (np.linalg.norm(np.mean(context_a_vec, axis=0) - np.mean(all_vec, axis=0)) ** 2)
        b_term = len(context_b) * (np.linalg.norm(np.mean(context_b_vec, axis=0) - np.mean(all_vec, axis=0)) ** 2)
        denom_terms = []
        for vec in all_vec:
            denom_terms.append(np.linalg.norm(vec - np.mean(all_vec, axis=0)) ** 2)
        # return value
        return (a_term + b_term) / sum(denom_terms)


def driver(dictionary_of_models,
           list_of_terms,
           k=10,
           use_pretrained=False,
           is_random=False):
    list_of_models = list(dictionary_of_models.values())
    if use_pretrained:
        terms = [term[0] for term in list_of_terms if term[0] in list_of_models[-1].wv.vocab]
        tfidf = [term[1] for term in list_of_terms if term[0] in list_of_models[-1].wv.vocab]
    else:
        terms = [term[0] for term in list_of_terms if
                 term[0] in list_of_models[0].wv.vocab and term[0] in list_of_models[1].wv.vocab]
        tfidf = [term[1] for term in list_of_terms if
                 term[0] in list_of_models[0].wv.vocab and term[0] in list_of_models[1].wv.vocab]
    max_tfidf = max(tfidf)
    controversy = []
    for term in terms:
        term_controversy = find_controversy_2(list_of_models,
                                              term,
                                              k,
                                              use_pretrained)
        if term_controversy is None:
            term_controversy = 0
        controversy.append((term, term_controversy))

    # measure for each term
    polarity = []
    weight = []
    assert len(terms) == len(tfidf) == len(controversy)
    for term, term_tfidf, term_controversy in zip(terms, tfidf, controversy):
        polarity.append((term, term_tfidf / max_tfidf * math.sqrt(term_controversy[1])))
        weight.append(term_tfidf / max_tfidf)

    # measure for corpus
    assert len(controversy) == len(weight)
    controversy_ = [math.sqrt(term[1]) for term in controversy]
    weighted_avg = np.average(controversy_, weights=weight)
    with open('weighted_average_{}_{}_{}'.format(use_pretrained, k, is_random), 'w', encoding='utf-8') as fp:
        print(weighted_avg, sep='\n', file=fp)

    controversy = sorted(controversy, key=operator.itemgetter(1))
    controversy = controversy[::-1]
    with open('raw_controversy_{}_{}_{}'.format(use_pretrained, k, is_random), 'w', encoding='utf-8') as fp:
        print(*controversy, sep='\n', file=fp)

    polarity = sorted(polarity, key=operator.itemgetter(1))
    polarity = polarity[::-1]
    with open('polarity_{}_{}_{}'.format(use_pretrained, k, is_random), 'w', encoding='utf-8') as fp:
        print(*polarity, sep='\n', file=fp)
