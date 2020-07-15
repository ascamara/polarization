import os
import re
import shutil
import numpy as np
import nltk
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import decimal
from tqdm import tqdm

def lower(tokens):
    return [w.lower() for w in tokens]


def numberize(tokens):
    def has_number(token):
        return bool(re.search(r'\d', token))

    def ordinal(token):
        ordins = ['th', 'rd', 'st', 'nd']
        if token[-2:] in ordins and has_number(token[-3]):
            return True
        else:
            return False

    def has_letters(tok):
        for t in tok:
            if not has_number(t):
                return True
        return False

    tokens = list(filter(None, tokens))

    for token in tokens:
        try:
            # normal case - fifty-three
            if has_number(token) and not ordinal(token):
                # if it has letters, let her go (n95, covid19)
                if not has_letters(token):
                    tokens[tokens.index(token)] = str(num2words(int(token)))
            # unless its an ordinal!
            elif has_number(token) and ordinal(token):
                tokens[tokens.index(token)] = str(num2words(token[:-2], ordinal=True))
        except decimal.InvalidOperation:
            print('something didnt convert good')
    return tokens



def remove_punctuation(tokens):
    for token in tokens:
        tokens[tokens.index(token)] = re.sub(r'\W+', '', token)
    return tokens


def lemmatize_and_stem(tokens, stemming):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for token in tokens:
        try:
            tokens[tokens.index(token)] = lemmatizer.lemmatize(token)
            if stemming:
                tokens[tokens.index(token)] = stemmer.stem(token)
        except ValueError as e:
            continue
    return tokens


def remove_stopwords(tokens):
    for token in tokens:
        if token in stopwords.words('english'):
            tokens.remove(token)
    return tokens


def remove_single_chars(tokens):
    for token in tokens:
        if len(token) <= 1:
            tokens.remove(token)
    return tokens


def swap_period_for_newline(line, idx):
    return line[:idx] + '\n' + line[idx + 1:]


def clean(directory_stem, folder, filename, is_fox, stemming):
    file_to_open = r'.\{}\{}'.format(folder, filename)
    file_to_write_name = 'clean_{}'.format(filename)

    tokens = []
    if not is_fox:
        with open(file_to_open, 'r', encoding='utf-8') as fp:
            for line in fp:
                tokens.extend(line.split())
            fp.close()
    else:
        quote = False
        speakers = ['laura:', 'tucker:', 'sean:']
        with open(file_to_open, 'r', encoding='utf-8') as fp:
            # fix lines
            for line in fp:
                line = line.lower().split()
                try:
                    # case one - general
                    if (next(i for i in reversed(range(len(line))) if line[i] == line[1]) -
                        next(i for i in reversed(range(len(line))) if line[i] == line[0]) == 1) & (
                            next(i for i in reversed(range(len(line))) if line[i] == line[0]) != 0):
                        line = line[:len(line) // 2]
                    # case two - weird sentences where one of the words is used again later
                    elif (next(i for i in reversed(range(len(line))) if line[i] == line[2]) -
                          next(i for i in reversed(range(len(line))) if line[i] == line[0]) == 2) & (
                            next(i for i in reversed(range(len(line))) if line[i] == line[0]) != 0):
                        line = line[:len(line) // 2]
                    elif (next(i for i in reversed(range(len(line))) if line[i] == line[3]) -
                          next(i for i in reversed(range(len(line))) if line[i] == line[1]) == 2) & (
                            next(i for i in reversed(range(len(line))) if line[i] == line[1]) != 0):
                        line = line[:len(line) // 2]
                except IndexError as e:
                    pass
                try:
                    # case two - only two things
                    if line[0].strip() == line[1].strip():
                        line = line[:len(line) // 2]
                except IndexError as e:
                    pass
                # check quote
                if line[0] == ">>" and line[1] in speakers:
                    del line[0:2]
                    quote = False
                elif line[0] == ">>":
                    quote = True
                # if quote, delete that!
                if not quote:
                    '''line = ' '.join(line)
                    for elt in range(0, len(line)):
                        if line[elt] == "." and (line[elt + 1] == '' or line[elt + 1] == ' ') and line[elt - 2] != ".":
                            line = swap_period_for_newline(line, elt)
                            print(line)'''
                    tokens.extend(line)

    # now that we have tokens, we can clean!
    # lower case
    tokens = lower(tokens)

    # single chars
    tokens = remove_single_chars(tokens)

    # remove punctuation
    tokens = remove_punctuation(tokens)
    tokens = list(filter(None, tokens))

    # stop words
    tokens = remove_stopwords(tokens)

    # lemmatization and stemming
    tokens = lemmatize_and_stem(tokens, stemming)

    # numbers to words
    tokens = numberize(tokens)

    # fix for nbc corpus
    for token in tokens:
        if token == 'gt' or token == 'gtgt' or token == 'gtgtgt':
            del token

    with open(file_to_write_name, 'w', encoding='utf-8') as f:
        print(*tokens, sep=' ', file=f)

    source = directory_stem + '\{}'.format(file_to_write_name)
    dest = directory_stem + '\clean_{}\{}'.format(folder, filename)
    os.rename(source, dest)


def clean_folder(directory_stem, directory, folder, stemming=True):
    new_path = directory_stem + '\\' + 'clean_{}'.format(folder)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    is_fox = False
    if folder == 'fox':
        is_fox = True

    for filename in tqdm(os.listdir(directory), ascii=True, desc='Cleaning'):
        clean(directory_stem, folder, filename, is_fox, stemming)
    return 'clean_{}'.format(folder)


