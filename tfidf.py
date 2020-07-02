import operator
import os
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict


# create set of words over a set of docs
def create_set(folder, filename):
    file_to_open = r'.\{}\{}'.format(folder, filename)
    raw = open(file_to_open).read()
    words = raw.split()
    return sorted(set(words))


def create_set_corpora(direct, folder):
    words_set = set()

    for file in os.listdir(direct):
        file_to_open = r'.\{}\{}'.format(folder, file)
        raw = open(file_to_open).read()
        words = raw.split()
        words_set.update(words)

    return sorted(words_set)


def tf(word, folder, filename):
    file_to_open = r'.\{}\{}'.format(folder, filename)
    raw = open(file_to_open).read()
    words = raw.split()

    # check to see if this is good
    return words.count(word) / len(words)


def doc_contains_word(word, folder, filename):
    file_to_open = r'.\{}\{}'.format(folder, filename)
    raw = open(file_to_open).read()
    words = raw.split()
    return words.count(word)


def no_docs_with_word(word, folder, direct):
    number_docs_with_word = 0
    for file in os.listdir(direct):
        if doc_contains_word(word, folder, file):
            number_docs_with_word += 1
    return number_docs_with_word


# old func
def dictionary_word_in_corpus_per_doc_old(directory, folder):
    word_set = create_set_corpora(directory, folder)
    word_in_docs = defaultdict(int)
    for word in word_set.copy():
        if word in stopwords.words('english'):
            word_set.remove(word)
    for word in word_set:
        word_in_docs[word] = no_docs_with_word(word, folder, directory)
        print("Searching: " + word)
    return word_in_docs


def dictionary_word_in_corpus_per_doc(directory, folder, total):
    word_set = create_set_corpora(directory, folder)
    word_in_docs = defaultdict(int)
    for word in word_set.copy():
        if word in stopwords.words('english'):
            word_set.remove(word)
        else:
            word_in_docs[word] = 0
    progress = 0
    for file in os.listdir(directory):
        progress += 1
        file_to_open = r'.\{}\{}'.format(folder, file)
        raw = open(file_to_open).read()
        words = raw.split()
        set_to_search = set(words)
        for elt in set_to_search:
            word_in_docs[elt] += 1
        print("Completed: " + str(progress) + " out of " + str(total) + " documents.")
        print(progress / total)
    return word_in_docs


def tfidf_corpus(directory, folder):
    AT_LEAST_PCT = .2

    total = 0
    progress = 0
    for filename in os.listdir(directory):
        total += 1

    word_in_docs = dictionary_word_in_corpus_per_doc(directory, folder, total)
    tfidf_pairs = defaultdict(float)

    for filename in os.listdir(directory):
        doc_set = create_set(folder, filename)
        for elt in doc_set:
            if word_in_docs[elt] > (total * AT_LEAST_PCT):
                # tf * idf
                tfidf_pairs[elt] += tf(elt, folder, filename) * math.log(total / word_in_docs[elt])

        progress += 1
        print("Completed: " + str(progress) + " out of " + str(total) + " documents.")
        print(progress / total)

    sorted_tfidf_pairs = []
    for key, value in tfidf_pairs.items():
        temp_pair = [key, value / total]
        sorted_tfidf_pairs.append(temp_pair)
    sorted_tfidf_pairs = sorted(sorted_tfidf_pairs, key=operator.itemgetter(1))

    with open('tfidf_{}'.format(folder), 'w', encoding='utf-8') as fp:
        print(*sorted_tfidf_pairs, sep='\n', file=fp)

    return sorted_tfidf_pairs


def tfidf_corpora(directory_stem, folders):
    tfidf_docs = defaultdict(float)
    ensure_word_appears_in_each_corpus = defaultdict(int)
    array_of_corpora = []
    for folder in folders:
        path = directory_stem + '\{}'.format(folder)
        corpus_ = tfidf_corpus(path, folder)
        array_of_corpora.append(corpus_)
        for pair in corpus_:
            ensure_word_appears_in_each_corpus[pair[0]] += 1
    for corpus in array_of_corpora:
        for pair in corpus:
            if ensure_word_appears_in_each_corpus[pair[0]] == len(array_of_corpora):
                tfidf_docs[pair[0]] += pair[1]
    sorted_tfidf_docs = []
    for key, value in tfidf_docs.items():
        temp_pair = [key, value / len(folders)]
        sorted_tfidf_docs.append(temp_pair)
    sorted_tfidf_docs = sorted(sorted_tfidf_docs, key=operator.itemgetter(1))
    sorted_tfidf_docs = sorted_tfidf_docs[::-1]

    with open('tfidf_final', 'w', encoding='utf-8') as fp:
        print(*sorted_tfidf_docs, sep='\n', file=fp)

    return sorted_tfidf_docs


def read_tfidf(file):
    terms = []
    with open(file, 'r', encoding='utf-8') as fp:
        for line in fp:
            term = line[2:line.rfind("'")]
            value = float(line[line.rfind(' ') + 1:-2])
            terms.append([term, value])
    return terms
