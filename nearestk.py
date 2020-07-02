import numpy as np
import matplotlib as pyplot
from gensim.models import Word2Vec
import operator
from time import time
from gensim.models import KeyedVectors
from nltk.corpus import stopwords


def cosine_sim(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norma = np.linalg.norm(vec1)
    normb = np.linalg.norm(vec2)
    cos = dot / (norma * normb)
    return cos


def find_tightness_model(model, matrix, anchor_word, k,
                         use_cocurrence, use_pretrained_comparison, pretrained_model, n=90):
    comparison_dict, values_array = [], []
    if use_cocurrence and use_pretrained_comparison:
        top_n_pct = np.percentile([matrix[anchor_word][term] for term in matrix[anchor_word]], n)
        for key in matrix[anchor_word]:
            if matrix[anchor_word][key] > top_n_pct and key in pretrained_model.wv.vocab \
                    and key not in stopwords.words('english'):
                comparison_dict.append(key)
        print(anchor_word)
        print(comparison_dict)
        for term in comparison_dict:
            vec1 = pretrained_model.wv[anchor_word]
            vec2 = pretrained_model.wv[term]
            values_array.append(cosine_sim(vec1, vec2))
        return 1 / len(values_array) * sum(values_array)

    elif use_cocurrence and not use_pretrained_comparison:
        top_n_pct = np.percentile([matrix[anchor_word][term] for term in matrix[anchor_word]], n)
        for key in matrix[anchor_word]:
            if matrix[anchor_word][key] > top_n_pct and key in model.wv.vocab \
                    and key not in stopwords.words('english'):
                comparison_dict.append(key)
        print(anchor_word)
        print(comparison_dict)
        for term in comparison_dict:
            vec1 = model.wv[anchor_word]
            vec2 = model.wv[term]
            values_array.append(cosine_sim(vec1, vec2))
        return 1 / len(values_array) * sum(values_array)

    elif not use_cocurrence and use_pretrained_comparison:
        similarity = model.wv.most_similar(positive=[anchor_word], topn=k, restrict_vocab=len(model.wv.vocab) // 4)
        temp_dict = [term[0] for term in similarity]
        comparison_dict = [term for term in temp_dict if term in pretrained_model.wv.vocab]
        print(anchor_word)
        print(comparison_dict)
        for term in comparison_dict:
            vec1 = pretrained_model.wv[anchor_word]
            vec2 = pretrained_model.wv[term]
            values_array.append(cosine_sim(vec1, vec2))
        return 1 / len(values_array) * sum(values_array)

    elif not use_cocurrence and not use_pretrained_comparison:
        similarity = model.wv.most_similar(positive=[anchor_word], topn=k, restrict_vocab=len(model.wv.vocab) // 4)
        temp_dict = [term[0] for term in similarity]
        comparison_dict = [term for term in temp_dict if term in model.wv.vocab]
        print(anchor_word)
        print(comparison_dict)
        for term in comparison_dict:
            vec1 = model.wv[anchor_word]
            vec2 = model.wv[term]
            values_array.append(cosine_sim(vec1, vec2))
        return 1 / len(values_array) * sum(values_array)
    else:
        return -1


def find_tightness(list_of_models, list_of_matrices, term, k,
                   use_cocurrence, use_pretrained_comparison):
    tightnesses = []
    for model, matrix in zip(list_of_models, list_of_matrices):
        tightnesses.append(find_tightness_model(model, matrix, term, k,
                                                use_cocurrence, use_pretrained_comparison, list_of_models[-1]))
    return 1 / len(list_of_models) * sum(tightnesses)


def find_dispersion_2(list_of_models, list_of_matrices, anchor_word, k,
                      use_cocurrence, use_pretrained_comparison, n=90):
    # model one anchor
    values_array, comparison_dict = [], []
    values_array_b, comparison_dict_b = [], []
    pretrained_model = list_of_models[-1]
    model, opposite = list_of_models[0], list_of_models[1]
    matrix, oppo_matrix = list_of_matrices[0], list_of_matrices[1]
    if use_cocurrence and use_pretrained_comparison:
        top_n_pct = np.percentile([oppo_matrix[anchor_word][term] for term in oppo_matrix[anchor_word]], n)
        for key in oppo_matrix[anchor_word]:
            if oppo_matrix[anchor_word][key] > top_n_pct \
                    and key in pretrained_model.wv.vocab \
                    and key not in stopwords.words('english'):
                comparison_dict.append(key)
        print(anchor_word)
        print(comparison_dict)
        for term in comparison_dict:
            vec1 = pretrained_model.wv[anchor_word]
            vec2 = pretrained_model.wv[term]
            values_array.append(1 - cosine_sim(vec1, vec2))
        term_one = 1 / len(values_array) * sum(values_array)

        top_n_pct_b = np.percentile([matrix[anchor_word][term] for term in matrix[anchor_word]], n)
        for key in matrix[anchor_word]:
            if matrix[anchor_word][key] > top_n_pct_b \
                    and key in pretrained_model.wv.vocab \
                    and key not in stopwords.words('english'):
                comparison_dict_b.append(key)
        print(anchor_word)
        print(comparison_dict_b)
        for term in comparison_dict_b:
            vec1 = pretrained_model.wv[anchor_word]
            vec2 = pretrained_model.wv[term]
            values_array_b.append(1 - cosine_sim(vec1, vec2))
        term_two = 1 / len(values_array_b) * sum(values_array_b)
        return 1 / 2 * (term_one + term_two)

    elif use_cocurrence and not use_pretrained_comparison:
        top_n_pct = np.percentile([oppo_matrix[anchor_word][term] for term in oppo_matrix[anchor_word]], n)
        for key in oppo_matrix[anchor_word]:
            if oppo_matrix[anchor_word][key] > top_n_pct \
                    and key in opposite.wv.vocab \
                    and key not in stopwords.words('english'):
                comparison_dict.append(key)
        print(anchor_word)
        print(comparison_dict)
        for term in comparison_dict:
            vec1 = model.wv[anchor_word]
            vec2 = opposite.wv[term]
            values_array.append(1 - cosine_sim(vec1, vec2))
        term_one = 1 / len(values_array) * sum(values_array)

        top_n_pct_b = np.percentile([matrix[anchor_word][term] for term in matrix[anchor_word]], n)
        for key in matrix[anchor_word]:
            if matrix[anchor_word][key] > top_n_pct_b \
                    and key in model.wv.vocab \
                    and key not in stopwords.words('english'):
                comparison_dict_b.append(key)
        print(anchor_word)
        print(comparison_dict_b)
        for term in comparison_dict_b:
            vec1 = opposite.wv[anchor_word]
            vec2 = model.wv[term]
            values_array_b.append(1 - cosine_sim(vec1, vec2))
        term_two = 1 / len(values_array_b) * sum(values_array_b)
        return 1 / 2 * (term_one + term_two)

    elif not use_cocurrence and use_pretrained_comparison:
        similarity = opposite.wv.most_similar(positive=[anchor_word], topn=k,
                                              restrict_vocab=len(opposite.wv.vocab) // 4)
        temp_dict = [term[0] for term in similarity]
        comparison_dict = [term for term in temp_dict if term in pretrained_model.wv.vocab]
        print(anchor_word)
        print(comparison_dict)
        for term in comparison_dict:
            vec1 = pretrained_model.wv[anchor_word]
            vec2 = pretrained_model.wv[term]
            values_array.append(1 - cosine_sim(vec1, vec2))
        term_one = 1 / len(values_array) * sum(values_array)

        similarity_b = model.wv.most_similar(positive=[anchor_word], topn=k,
                                             restrict_vocab=len(model.wv.vocab) // 4)
        temp_dict_b = [term[0] for term in similarity_b]
        comparison_dict_b = [term for term in temp_dict_b if term in pretrained_model.wv.vocab]
        print(anchor_word)
        print(comparison_dict_b)
        for term in comparison_dict_b:
            vec1 = pretrained_model.wv[anchor_word]
            vec2 = pretrained_model.wv[term]
            values_array_b.append(1 - cosine_sim(vec1, vec2))
        term_two = 1 / len(values_array_b) * sum(values_array_b)
        return 1 / 2 * (term_one + term_two)

    elif not use_cocurrence and not use_pretrained_comparison:
        similarity = opposite.wv.most_similar(positive=[anchor_word], topn=k,
                                              restrict_vocab=len(opposite.wv.vocab) // 4)
        temp_dict = [term[0] for term in similarity]
        comparison_dict = [term for term in temp_dict if term in opposite.wv.vocab]
        print(anchor_word)
        print(comparison_dict)
        for term in comparison_dict:
            vec1 = model.wv[anchor_word]
            vec2 = opposite.wv[term]
            values_array.append(1 - cosine_sim(vec1, vec2))
        term_one = 1 / len(values_array) * sum(values_array)

        similarity_b = model.wv.most_similar(positive=[anchor_word], topn=k,
                                             restrict_vocab=len(model.wv.vocab) // 4)
        temp_dict_b = [term[0] for term in similarity]
        comparison_dict_b = [term for term in temp_dict_b if term in model.wv.vocab]
        print(anchor_word)
        print(comparison_dict_b)
        for term in comparison_dict_b:
            vec1 = opposite.wv[anchor_word]
            vec2 = model.wv[term]
            values_array_b.append(1 - cosine_sim(vec1, vec2))
        term_two = 1 / len(values_array_b) * sum(values_array_b)
        return 1 / 2 * (term_one + term_two)

    else:
        return -1


def driver(dictionary_of_models, dictionary_of_co, list_of_terms, k=5, use_cocurrence=True, use_pretrained=False):
    list_of_models = list(dictionary_of_models.values())
    list_of_matrices = list(dictionary_of_co.values())
    terms = [term[0] for term in list_of_terms if
             term[0] in list_of_models[0].wv.vocab and term[0] in list_of_models[1].wv.vocab]
    tfidf_values = [term[1] for term in list_of_terms if
                    term[0] in list_of_models[0].wv.vocab and term[0] in list_of_models[1].wv.vocab]
    tightness = []
    dispersion = []
    for term in terms:
        term_tightness = find_tightness(list_of_models, list_of_matrices, term, k, use_cocurrence, use_pretrained)
        term_dispersion = find_dispersion_2(list_of_models, list_of_matrices, term, k, use_cocurrence, use_pretrained)
        tightness.append((term, term_tightness))
        dispersion.append((term, term_dispersion))

    print_list = []
    values = []
    weight = []
    max_tfidf = max(tfidf_values)
    for i in range(len(terms)):
        # measure = float(tfidf_values[i] / max_tfidf) * float((dispersion[i][1] * tightness[i][1]) / 2)
        measure = float(tightness[i][1])
        print_list.append((terms[i], measure))
        values.append(float((dispersion[i][1] * tightness[i][1]) / 2))
        weight.append(float(tfidf_values[i] / max_tfidf))

    print_list = sorted(print_list, key=operator.itemgetter(1))
    print_list = print_list[::-1]

    weighted_avg = np.average(values, weights=weight)
    tightness_ = [term[1] for term in tightness]
    dispersion_ = [term[1] for term in dispersion]
    weighted_tight = np.average(tightness_, weights=weight)
    weighted_disp = np.average(dispersion_, weights=weight)
    print(weighted_avg)

    with open('the_list_{}_{}_{}'.format(k, use_cocurrence, use_pretrained), 'w', encoding='utf-8') as fp:
        print(*print_list, sep='\n', file=fp)

    tightness = sorted(tightness, key=operator.itemgetter(1))
    tightness = tightness[::-1]
    dispersion = sorted(dispersion, key=operator.itemgetter(1))
    dispersion = dispersion[::-1]

    with open('tight_{}_{}'.format(use_cocurrence, use_pretrained), 'w', encoding='utf-8') as fp:
        print(*tightness, sep='\n', file=fp)

    with open('disp_{}_{}'.format(use_cocurrence, use_pretrained), 'w', encoding='utf-8') as fp:
        print(*dispersion, sep='\n', file=fp)


def maxmink(dictionary_of_models, list_of_terms, range_in=1, range_out=101):
    k_ = []
    k_arr = []
    for k in range(range_in, range_out):
        t = time()
        k_arr.append(driver(dictionary_of_models, list_of_terms, k))
        print('Time to build {}-nearest: {} mins'.format(k, round((time() - t) / 60, 2)))
    k_np = np.array([k for k in range(range_in, range_out)])
    k_arr_np = np.array(k_arr)
    print(k_np)
    print(k_arr_np)
    pyplot.plot(k_np, k_arr_np)
    pyplot.show()
    k_arr = sorted(k_arr)
    k_arr = k_arr[::-1]
    print(k_arr)
