from collections import defaultdict
import math
from tqdm import tqdm
import pickle
import os

lv = True
ns = True


def create_set_document(document):
    word_set = set()
    for sentence in document:
        word_set.update([w for w in sentence])
    return sorted(word_set)


def create_set_corpus(list_of_documents):
    #tqdm here?
    word_set = set()
    for document in list_of_documents:
        for sentence in document:
            word_set.update([w for w in sentence])
    return sorted(word_set)


def find_docs_with_word_dict(list_of_documents):
    docs_with_word = defaultdict(int)
    corpus_set = create_set_corpus(list_of_documents)
    for document in list_of_documents:
        document_set = create_set_document(document)
        for word in document_set:
            docs_with_word[word] += 1
    return docs_with_word

def find_docs_with_word_dict_LINE_LEVEL(list_of_documents):
    docs_with_word = defaultdict(int)
    for document in list_of_documents:
        document_set = set(document)
        for word in document_set:
            docs_with_word[word] += 1
    return docs_with_word


def word_in_doc(document, word):
    sum = 0
    for sentence in document:
        sum += sentence.count(word)
    return sum


def run_tfidf(list_of_raw_documents, name, DOCUMENT_AT_LINE_LEVEL=True):
    list_of_documents = []
    if DOCUMENT_AT_LINE_LEVEL:
        for document in list_of_raw_documents:
            for sentence in document:
                temp_doc = sentence.split()
                if len(temp_doc) > 0:
                    list_of_documents.append(sentence.split())
    else:
        for document in list_of_raw_documents:
            temp_doc = []
            for sentence in document:
                temp_doc.append(sentence.split())
            list_of_documents.append(temp_doc)

    if DOCUMENT_AT_LINE_LEVEL:
        docs_with_word = find_docs_with_word_dict_LINE_LEVEL(list_of_documents)
    else:
        docs_with_word = find_docs_with_word_dict(list_of_documents)

    tfidf = defaultdict(float)
    # Conduct TFIDF on each document
    for document in tqdm(list_of_documents, ascii=True, desc='TFIDF_{}'.format(name), position=1):
        if DOCUMENT_AT_LINE_LEVEL:
            document_set = set(document)
        else:
            document_set = create_set_document(document)
        for word in document_set:
            tfidf[word] += \
                (word_in_doc(document, word) / len(document)) * math.log(len(list_of_documents) / docs_with_word[word])
    # Averaged over number of documents
    for key, value in tfidf.items():
        tfidf[key] = value / len(list_of_documents)
    return tfidf


def dict_sum(tfidf_subcorpus, key):
    sum = 0
    for subcorpus in tfidf_subcorpus:
        sum += subcorpus[key]
    return sum


def tfidf_corpus(source_dictionary, DOCUMENT_AT_LINE_LEVEL):
    tfidf_corpus = defaultdict(float)
    tfidf_subcorpus = []
    for key, value in tqdm(source_dictionary.items(), ascii=True, desc="TFIDF", position=0):
        if os.path.isfile('tfidf_{}'.format(key)):
            temp = pickle.load(open('tfidf_{}'.format(key), 'rb'))
            tfidf_subcorpus.append(temp)
        else:
            temp_tfidf = run_tfidf(value, key)
            with open('tfidf_{}'.format(key), 'wb') as fp:
                pickle.dump(temp_tfidf, fp)
            tfidf_subcorpus.append(temp_tfidf)
    # for each tfidf_subcorpus, ensure the word appears in all and then add to final
    ensure_word_appears = defaultdict(int)
    for subcorpus in tfidf_subcorpus:
        for key, value in subcorpus.items():
            ensure_word_appears[key] += 1
    # build the final tfidf dictionary
    for key, value in ensure_word_appears.items():
        if ensure_word_appears[key] == len(tfidf_subcorpus):
            tfidf_corpus[key] = dict_sum(tfidf_subcorpus, key) / len(tfidf_subcorpus)
    return tfidf_corpus
