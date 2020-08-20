from collections import defaultdict
from gensim.models import KeyedVectors
import numpy as np
import plot
from tqdm import tqdm
import math

def cosine_sim(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norma = np.linalg.norm(vec1)
    normb = np.linalg.norm(vec2)
    cos = dot / (norma * normb)
    return math.sqrt(abs(cos))


def return_contexts(models, matrices, anchor, context_size):
    try:
        list_of_contexts = []
        for i in range(len(models)):
            matrix = matrices[i]
            dict = matrix[anchor[0]]
            list = sorted(dict.items(), key=lambda x: x[1], reverse=True)
            list_of_contexts.append([item[0] for item in list if item[0] in models[i].wv.vocab and item[1] > 10][:context_size])
        new_context = []
        common_words = set(list_of_contexts[0]).intersection(set(list_of_contexts[1]))
        for context in list_of_contexts:
            new_context.append([term for term in context if term not in common_words])
        new_context.insert(0, [term for term in common_words])
        return new_context
    except KeyError:
        return [['none'], ['none'], ['none']]



def context_builder(model, matrix, anchor, context_size):
    dict = matrix[anchor]
    list = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in list if item[0] in model.wv.vocab and item[1] > 10][:context_size]


def remove_common_words(list_of_contexts):
    if len(list_of_contexts) == 1:
        return list_of_contexts
    else:
        new_context = []
        common_words = set(list_of_contexts[0]).intersection(set(list_of_contexts[1]))
        for context in list_of_contexts:
            new_context.append([term for term in context if term not in common_words])
    return new_context


def find_controversy(list_of_models, anchor, list_of_matrices, context_size):
    list_of_contexts = []
    for model, matrix in zip(list_of_models, list_of_matrices):
        list_of_contexts.append(context_builder(model, matrix, anchor, context_size))
    # formula in paper
    context_vector_arrays = []
    for context, model in zip(list_of_contexts, list_of_models):
        context_vector_array = np.array([model.wv[term] for term in context])
        # assert len(context_vector_array) == k
        context_vector_arrays.append(context_vector_array)
    all_vector_array = np.concatenate(([cva for cva in context_vector_arrays]), axis=0)
    # assert len(all_vector_array) == k * len(list_of_models)

    source_least_sq = []
    for cva in context_vector_arrays:
        source_least_sq.append(
            (len(cva) * (np.linalg.norm(np.mean(cva, axis=0) - np.mean(all_vector_array, axis=0)) ** 2)))
    total_least_sq = []
    for cva in all_vector_array:
        total_least_sq.append(np.linalg.norm(cva - np.mean(all_vector_array, axis=0)) ** 2)
    # assert len(source_least_sq) == len(list_of_models)
    # assert len(total_least_sq) == k * len(list_of_models)
    return math.sqrt(sum(source_least_sq) / sum(total_least_sq))


def controversy_dictionary_use_co_occurance(model_dictionary, significance_list,
                                            co_occurance_matrix_dictionary, context_size):
    list_of_models = list(model_dictionary.values())
    list_of_matrices = list(co_occurance_matrix_dictionary.values())
    controversy = defaultdict(float)
    # top ten percent
    for element in tqdm(significance_list[:math.floor(len(significance_list) * .1)], ascii=True, desc='Calculating polarizing'):
        term_qualifies = True
        for model, matrix in zip(list_of_models, list_of_matrices):
            if element[0] not in model.wv.vocab or element[0] not in matrix:
                term_qualifies = False
        if term_qualifies:
            term = element[0]
            try:
                term_controversy = find_controversy(list_of_models, term, list_of_matrices, context_size)
                controversy[term] = term_controversy
            except ValueError:
                continue
    return controversy
