import embed
import numpy as np
import math
from operator import itemgetter

def dict_to_list(dict):
    list = []
    for key, value in dict.items():
        temp_pair = [key, value]
        list.append(temp_pair)
    list = sorted(list, key=itemgetter(1))
    list = list[::-1]
    return list

def r_squared(directory_stem, source_dictionary, analysis_dictionary, num_terms, PRETRAIN, ALIGN):
    dictionary_of_models = embed.create_models(source_dictionary, PRETRAIN, ALIGN)
    analysis_list = dict_to_list(analysis_dictionary)

    # list of contexts is an n elt list (where n is models) with m elts (where m is the words)
    list_of_models = list(dictionary_of_models.values())
    base_list = []
    count = 0
    while count < num_terms:
        add_element = True
        for model in list_of_models:
            if analysis_list[count][0] not in model.wv.vocab:
                add_element = False
        if add_element:
            base_list.append(analysis_list[count][0])
        count += 1

    list_of_contexts = [base_list, base_list]

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

def pairwise():
    pass
