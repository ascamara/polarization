import os
import pandas as pd
import numpy as np
import gensim
import scipy
import nearestk
import math
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA
from matplotlib import pyplot
from align import smart_procrustes_align_gensim
from statistics import stdev
from nltk.corpus import stopwords
from collections import Counter


def cosine_sim(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norma = np.linalg.norm(vec1)
    normb = np.linalg.norm(vec2)
    cos = dot / (norma * normb)
    return cos


def cos_sim_common_vocab(model_left, model_right_aligned, vocab=None, title=None):
    if not vocab:
        vocab = list(filter(lambda w: w in model_left.wv.vocab, model_right_aligned.wv.vocab))
    else:
        vocab = list(filter(lambda w: w in model_left.wv.vocab and w in model_right_aligned.wv.vocab, vocab))
    print(len(vocab))
    cos_sim = [cosine_sim(model_left[w], model_right_aligned[w]) for w in vocab]
    print(len(cos_sim))
    result = pd.DataFrame(list(zip(vocab, cos_sim)), columns=['vocab', 'cos_sim'])
    print(result.shape)
    result.boxplot(column=['cos_sim'])
    if not title:
        title = 'common vocab'
    pyplot.title(title)
    pyplot.savefig('box_plot_{}.png'.format(title))
    pyplot.show()
    result.to_csv('cos_sim.csv', index=None, header=True)


def PCA_plot(model_l, model_r, words, title):
    words = [word for word in words if word in model_l.wv.vocab and word in model_r.wv.vocab]
    X_l = model_l[words]
    X_r = model_r[words]
    X = np.concatenate((X_l, X_r), axis=0)
    partisanship = [1] * len(words) + [0] * len(words)
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1], c=partisanship)
    words = words + words
    for i, word in enumerate(words):
        # if i%2 == 0:
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.title(title)
    pyplot.savefig('2d_{}.png'.format(title))
    pyplot.show()


def PCA_plot_diff(model_l, model_r, model_l_words, model_r_words, title):
    X_l = model_l[model_l_words]
    X_r = model_r[model_r_words]
    len_words = len(model_r_words) + len(model_l_words)
    X = np.concatenate((X_l, X_r), axis=0)
    partisanship = [1] * len(model_l_words) + [0] * len(model_r_words)
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1], c=partisanship)
    model_l_words = [word + "_d" for word in model_l_words]
    model_r_words = [word + "_r" for word in model_r_words]
    words = model_l_words + model_r_words
    for i, word in enumerate(words):
        # if i%2 == 0:
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.title(title)
    pyplot.savefig('2d_{}.png'.format(title))
    pyplot.show()


# let's try some shit
# importance v. vector distance
def vector_distance_2_models(dictionary_of_models, list_of_terms, title='1'):
    x, y = [], []
    list_of_models = list(dictionary_of_models.values())
    list_of_terms = [term for term in list_of_terms[:20] if
                     term[0] in list_of_models[0].wv.vocab and term[0] in list_of_models[1].wv.vocab]
    for term in list_of_terms:
        vec1 = list_of_models[0].wv[term[0]]
        vec2 = list_of_models[1].wv[term[0]]
        x.append(np.linalg.norm(vec1 - vec2))
        y.append(term[1])
    pyplot.scatter(x, y)
    for i, term[0] in enumerate(list_of_terms):
        pyplot.annotate(term[0], (x[i], y[i]))
    pyplot.xlabel('Euclidean distance')
    pyplot.ylabel('Weighted tf-idf importance of term')
    pyplot.title(title)
    pyplot.savefig('test_{}.png'.format(title))
    pyplot.show()


# tightness vs dispersion, size is importance
def tight_disp_scatter(dictionary_of_models, dictionary_of_co_occurence, list_of_terms, k=20, n=99, title='1',
                       use_cocurrence=False,
                       use_pretrained_comparison=False,
                       pct_terms=25):
    x, y, s = [], [], []
    list_of_models = list(dictionary_of_models.values())
    list_of_matrices = list(dictionary_of_co_occurence.values())
    list_of_terms_ = list_of_terms[:math.ceil(len(list_of_terms)*pct_terms*.01)]
    if use_pretrained_comparison:
        list_of_terms_ = [term for term in list_of_terms_ if term[0] in list_of_models[-1].wv.vocab]
    else:
        list_of_terms_ = [term for term in list_of_terms_ if term[0] in list_of_models[0].wv.vocab
                          and term[0] in list_of_models[1].wv.vocab]
    list_of_terms = [term for term in list_of_terms_ if
                     term[0] in list_of_models[0].wv.vocab and term[0] in list_of_models[1].wv.vocab]
    max_tfidf = max([term[1] for term in list_of_terms])
    for term in list_of_terms_:
        x.append(nearestk.find_tightness(list_of_models, list_of_matrices, term[0], k, n,
                                         use_cocurrence, use_pretrained_comparison))
        y.append(nearestk.find_dispersion_2(list_of_models, list_of_matrices, term[0], k, n,
                                            use_cocurrence, use_pretrained_comparison))
        s.append(2 * 2 ** 10 * (term[1] ** 2) / (max_tfidf ** 2))
    pyplot.scatter(x, y, s=s)
    for i, term[0] in enumerate(list_of_terms_):
        pyplot.annotate(term[0][0], (x[i], y[i]))
    pyplot.xlabel('Tightness')
    pyplot.ylabel('Dispersion')
    pyplot.title(title)
    pyplot.savefig('test_{}.png'.format(title))
    pyplot.show()


def get_last_x(list_of_terms, x):
    return list_of_terms[:-x]


def get_first_x(list_of_terms, x):
    return list_of_terms[:x]


# works for two models only!
def cos_sim_plot_2_models(dictionary_of_models, list_of_terms, title='a'):
    x, y = [], []
    list_of_models = list(dictionary_of_models.values())
    list_of_terms = [term for term in list_of_terms if
                     term[0] in list_of_models[0].wv.vocab and term[0] in list_of_models[1].wv.vocab]
    list_of_terms_ = list_of_terms[0:21] + list_of_terms[len(list_of_terms) - 21:len(list_of_terms) - 1]
    print(len(list_of_terms_))
    for term in list_of_terms_:
        vec1 = list_of_models[0].wv[term[0]]
        vec2 = list_of_models[1].wv[term[0]]
        x.append(cosine_sim(vec1, vec2))
        y.append(term[1])
    pyplot.scatter(x, y)
    for i, term[0] in enumerate(list_of_terms_):
        pyplot.annotate(term[0], (x[i], y[i]))
    pyplot.xlabel('Cosine similarity')
    pyplot.ylabel('Weighted tf-idf importance of term')
    pyplot.title(title)
    pyplot.savefig('firstandlast_{}.png'.format(title))
    pyplot.show()


# finds cos sim with respect to the mean and std deviation cos sim of a word
def cos_sim_plot_2_models_wrt_mean(dictionary_of_models, list_of_terms):
    x, y = [], []
    list_of_models = list(dictionary_of_models.values())
    title = 'Fox News and NBC News Terms from Late March to Early June 2020'
    list_of_terms = [term for term in list_of_terms if
                     term[0] in list_of_models[0].wv.vocab and term[0] in list_of_models[1].wv.vocab]
    # find average and std dev for cosine sim
    sample = []
    for term in list_of_terms:
        vec1 = list_of_models[0].wv[term[0]]
        vec2 = list_of_models[1].wv[term[0]]
        sample.append(cosine_sim(vec1, vec2))
    mean = sum(sample) / len(sample)
    stddev = np.std(sample)
    indices_to_mark = []
    for i, term in enumerate(list_of_terms):
        vec1 = list_of_models[0].wv[term[0]]
        vec2 = list_of_models[1].wv[term[0]]
        p = scipy.stats.norm.sf(abs(cosine_sim(vec1, vec2) - mean) / stddev)
        if p <= .05:
            x.append(cosine_sim(vec1, vec2))
            y.append(term[1])
            indices_to_mark.append(term[0])
    pyplot.scatter(x, y)
    for i, term in enumerate(indices_to_mark):
        pyplot.annotate(term, (x[i], y[i]))
    pyplot.xlabel('Z score')
    pyplot.ylabel('Weighted tf-idf importance of term')
    pyplot.title(title)
    pyplot.savefig('cos_sim_z-score_{}.png'.format(title))
    pyplot.show()


def plot_word_with_context(dictionary_of_models, word, k):
    list_of_models = list(dictionary_of_models.values())

    model_1_similarity = list_of_models[0].wv.most_similar(positive=[word], topn=k)
    model_1_comp_dict = [term[0] for term in model_1_similarity]
    model_1_words = [word] + model_1_comp_dict

    model_2_similarity = list_of_models[1].wv.most_similar(positive=[word], topn=k)
    model_2_comp_dict = [term[0] for term in model_2_similarity]
    model_2_words = [word] + model_2_comp_dict
    PCA_plot_diff(list_of_models[0], list_of_models[1], model_1_words, model_2_words, title=word)


def plot_word_with_context_base(dictionary_of_models, word, k):
    list_of_models = list(dictionary_of_models.values())

    model_1_similarity = list_of_models[0].wv.most_similar(positive=[word], topn=k)
    model_1_comp_dict = [term[0] for term in model_1_similarity]
    model_1_comp_dict = [term for term in model_1_comp_dict if term in list_of_models[-1].wv.vocab]
    model_1_words = [word] + model_1_comp_dict

    model_2_similarity = list_of_models[1].wv.most_similar(positive=[word], topn=k)
    model_2_comp_dict = [term[0] for term in model_2_similarity]
    model_2_comp_dict = [term for term in model_2_comp_dict if term in list_of_models[-1].wv.vocab]
    model_2_words = [word] + model_2_comp_dict
    PCA_plot_diff(list_of_models[-1], list_of_models[-1], model_1_words, model_2_words, title=word)


def plot_word_with_cocurrence(dictionary_of_models, dictionary_of_co_occurence, anchor_word, n):
    models = list(dictionary_of_models.values())
    matrices = list(dictionary_of_co_occurence.values())
    # model 1
    top_n_pct_1 = np.percentile([matrices[0][anchor_word][term] for term in matrices[0][anchor_word]], n)
    print(n)
    print(top_n_pct_1)
    model_1_comp_dict = []
    for key in matrices[0][anchor_word]:
        if matrices[0][anchor_word][key] > top_n_pct_1 \
                and key in models[0].wv.vocab \
                and key not in stopwords.words('english'):
            model_1_comp_dict.append(key)
    model_1_words = [anchor_word] + model_1_comp_dict
    # model 2
    top_n_pct_2 = np.percentile([matrices[1][anchor_word][term] for term in matrices[1][anchor_word]], n)
    model_2_comp_dict = []
    for key in matrices[1][anchor_word]:
        if matrices[1][anchor_word][key] > top_n_pct_1 \
                and key in models[1].wv.vocab \
                and key not in stopwords.words('english'):
            model_2_comp_dict.append(key)
    model_2_words = [anchor_word] + model_2_comp_dict
    PCA_plot_diff(models[0], models[1], model_1_words, model_2_words, title=anchor_word)


def plot_word_with_cocurrence_base(dictionary_of_models, dictionary_of_co_occurence, anchor_word, n):
    models = list(dictionary_of_models.values())
    matrices = list(dictionary_of_co_occurence.values())
    # model 1
    top_n_pct_1 = np.percentile([matrices[0][anchor_word][term] for term in matrices[0][anchor_word]], n)
    model_1_comp_dict = []
    for key in matrices[0][anchor_word]:
        if matrices[0][anchor_word][key] > top_n_pct_1 \
                and key in models[-1].wv.vocab \
                and key not in stopwords.words('english'):
            model_1_comp_dict.append(key)
    model_1_words = [anchor_word] + model_1_comp_dict
    # model 2
    top_n_pct_2 = np.percentile([matrices[1][anchor_word][term] for term in matrices[1][anchor_word]], n)
    model_2_comp_dict = []
    for key in matrices[1][anchor_word]:
        if matrices[1][anchor_word][key] > top_n_pct_1 \
                and key in models[-1].wv.vocab \
                and key not in stopwords.words('english'):
            model_2_comp_dict.append(key)
    model_2_words = [anchor_word] + model_2_comp_dict
    PCA_plot_diff(models[-1], models[-1], model_1_words, model_2_words, title=anchor_word)


'''
def main(path):
    models_path = 'models_pretrained_flock_1'
    # models_path = 'models'
    read_saved = True
    if read_saved:
        model_right = KeyedVectors.load_word2vec_format('./{}/model_right.txt'.format(models_path), binary=False)
        model_left = KeyedVectors.load_word2vec_format('./{}/model_left.txt'.format(models_path), binary=False)
    else:
        all_data = pd.read_csv(os.path.join(path, 'kaggle.csv'))
        print(all_data.columns)
        data_left = all_data[all_data['political_side'] == 'l']
        print(data_left.shape)
        data_right = all_data[all_data['political_side'] == 'r']
        print(data_right.shape)

        # pretrained_path = 'GoogleNews-vectors-negative300.bin'
        pretrained_path = None
        model_left = w2v(data_left.title.astype('str').tolist(), pretrained_path)
        model_left.wv.save_word2vec_format('./{}/model_left.txt'.format(models_path), binary=False)
        print('model_left is saved.')

        model_right = w2v(data_right.title.astype('str').tolist(), pretrained_path)
        model_right.wv.save_word2vec_format('./{}/model_right.txt'.format(models_path), binary=False)
        print('model_right is saved.')

    model_right_aligned = smart_procrustes_align_gensim(model_left, model_right)
    model_right_aligned.wv.save_word2vec_format('./{}/model_right_aligned.txt'.format(models_path), binary=False)
    print('model_right_aligned is saved.')

    cos_sim_common_vocab(model_left, model_right, title='all common vocab')
    # print(model_left.most_similar(positive='president'))
    # print(model_right_aligned.most_similar(positive='president'))
    all_mft_expand = pd.read_csv(os.path.join(path, 'mft_expand_678.csv'))
    print(all_mft_expand.columns)
    words = ['climate', 'trump', 'abortion', 'obama', 'clinton']
    PCA_plot(model_l=model_left, model_r=model_right_aligned, words=words, title='sample words')
'''
