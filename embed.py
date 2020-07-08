import re
import os
import csv
import multiprocessing
import pandas as pd
from time import time
from collections import defaultdict
from gensim.test.utils import datapath
from gensim import utils
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import spacy

import logging

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)


def create_corpus(directory, folder, corpora_name="a", skip=25):
    # get words into an array of words
    words = []
    for filename in os.listdir(directory):
        file_to_open = r'.\{}\{}'.format(folder, filename)
        with open(file_to_open, 'r', encoding='utf-8') as fp:
            for line in fp:
                words.extend(line.split())
            fp.close()

    # add words to array of sentences
    lines = []
    count = 0
    temp_str = ''
    for elt in range(len(words)):
        words[elt] = words[elt].replace(',', '')
        if count < skip - 1:
            temp_str += words[elt] + ' '
            count += 1
        else:
            # add last elt
            temp_str += words[elt]
            lines.append(temp_str)
            temp_str = ''
            count = 0
    if temp_str != '':
        lines.append(temp_str)

    # write to corpora file
    file_to_write_name = 'corpora_{}.csv'.format(corpora_name)
    with open(file_to_write_name, 'w', encoding='utf-8') as csvfile:
        csvfile.write('lines')
        csvfile.write('\n')
        for line in lines:
            csvfile.write(line)
            csvfile.write('\n')


def create_corpus_file(directory_stem, partition_name):
    directory = directory_stem + '\{}.'.format(partition_name)
    folder = '{}'.format(partition_name)
    create_corpus(directory, folder, corpora_name=partition_name)


def create_corpora_files(directory, list_of_corpora):
    for corpus in list_of_corpora:
        create_corpus_file(directory, corpus)


def create_sentence_arrays(corpora_file_stem, list_of_corpora):
    dictionary_of_sentence_arrays = {}
    for corpus in list_of_corpora:
        dictionary_of_sentence_arrays[corpus] = create_sentence_array(corpora_file_stem, corpus)
    return dictionary_of_sentence_arrays


def create_sentence_array(corpora_file_stem, corpus):
    df = pd.read_csv(corpora_file_stem.format(corpus), sep=',')
    df = df.dropna().reset_index(drop=True)
    sent = [sentences.split() for sentences in df['lines']]
    phrases = Phrases(sent, min_count=40, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    return sentences

def fix_arrays(sent):
    phrases = Phrases(sent, min_count=40, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    return sentences

def create_models(dictionary_of_sentences, pretrain=True):
    dictionary_of_models = {}
    for k in dictionary_of_sentences:
        if pretrain:
            dictionary_of_models[k] = create_model_pretrained(dictionary_of_sentences[k])
        else:
            dictionary_of_models[k] = create_model(dictionary_of_sentences[k])
        path = 'model_{}.txt'.format(k)
        dictionary_of_models[k].wv.save_word2vec_format(path, binary=False)
    dictionary_of_models['base'] = create_model_base(dictionary_of_sentences)
    path = 'model_base.txt'
    dictionary_of_models['base'].wv.save_word2vec_format(path, binary=False)
    return dictionary_of_models


def create_model_base(dictionary_of_sentences, pretrained_path='GoogleNews-vectors-negative300.bin.gz'):
    sentences_tokenized = []
    for key, value in dictionary_of_sentences.items():
        sentences_tokenized += dictionary_of_sentences[key]
    if pretrained_path:
        model_2 = Word2Vec(size=300, min_count=5)
        model_2.build_vocab(sentences_tokenized)
        print(len(model_2.wv.vocab))
        total_examples = model_2.corpus_count
        model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        model_2.build_vocab([list(model.vocab.keys())], update=True)
        print(len(model_2.wv.vocab))
        # todo play with lockf
        model_2.intersect_word2vec_format(pretrained_path, binary=True, lockf=1)
        model_2.train(sentences_tokenized, total_examples=total_examples, epochs=model_2.iter)
    else:
        model_2 = Word2Vec(sentences_tokenized, size=300, min_count=10)
    return model_2


def create_model_pretrained(sentences, pretrained_path='GoogleNews-vectors-negative300.bin.gz'):
    sentences_tokenized = sentences
    if pretrained_path:
        model_2 = Word2Vec(size=300, min_count=5)
        model_2.build_vocab(sentences_tokenized)
        print(len(model_2.wv.vocab))
        total_examples = model_2.corpus_count
        model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        model_2.build_vocab([list(model.vocab.keys())], update=True)
        print(len(model_2.wv.vocab))
        # todo play with lockf
        model_2.intersect_word2vec_format(pretrained_path, binary=True, lockf=1)
        model_2.train(sentences_tokenized, total_examples=total_examples, epochs=model_2.iter)
    else:
        model_2 = Word2Vec(sentences_tokenized, size=300, min_count=10)
        print(len(model_2.wv.vocab))
    return model_2


def create_model_pretrained_without_sentences():
    return KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


def create_model(sentences):
    cores = multiprocessing.cpu_count()
    model = Word2Vec(min_count=20,
                     window=10,
                     size=300,
                     sample=6e-5,
                     alpha=.03,
                     min_alpha=.0007,
                     negative=20,
                     workers=cores - 1,
                     sorted_vocab=1)
    t = time()
    model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    return model


def create_dictionary_of_sentences(directory_stem, list_of_corpora):
    corpora_file_stem = directory_stem + '\corpora_{}.csv'
    create_corpora_files(directory_stem, list_of_corpora)
    return create_sentence_arrays(corpora_file_stem, list_of_corpora)
