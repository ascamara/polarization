import os
import re
import csv
import pickle
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from gensim.models.phrases import Phrases, Phraser
from collections import defaultdict
import decimal
from tqdm import tqdm

lv = True
ns = True


def numberize(line):
    def has_number(tok):
        return bool(re.search(r'\d', tok))

    def ordinal(tok):
        ordins = ['th', 'rd', 'st', 'nd']
        if tok[-2:] in ordins and has_number(tok[-3]):
            return True
        else:
            return False

    def has_letters(tok):
        for t in tok:
            if not has_number(t):
                return True
        return False

    line = list(filter(None, line))
    new_line = []
    for w in line:
        try:
            if has_number(w):
                if has_letters(w):
                    new_line.append(w)
            else:
                new_line.append(w)
        except decimal.InvalidOperation:
            print('Error: Numberize')
            '''
            # normal case - fifty-three
            if has_number(w) and not ordinal(w):
                # if it has letters, let her go (n95, covid19)
                if not has_letters(w):
                    line[line.index(w)] = str(num2words(int(w)))
            # unless its an ordinal!
            elif has_number(w) and ordinal(w):
                line[line.index(w)] = str(num2words(w[:-2], ordinal=True))
            '''
    return new_line


def lemmatize_and_stem(line):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for w in line:
        try:
            line[line.index(w)] = lemmatizer.lemmatize(w)
            line[line.index(w)] = stemmer.stem(w)
        except ValueError as e:
            continue
    return line


def clean_line(line):
    # remove punctuation
    for w in line:
        line[line.index(w)] = re.sub(r'\W+', '', w)
    line = list(filter(None, line))

    # lower-case
    line = [w.lower() for w in line]

    # num2word
    line = numberize(line)

    # lemma and stem
    # line = lemmatize_and_stem(line)

    # remove single characters and stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update('and')
    line = [w for w in line if w not in stop_words]
    line = [w for w in line if len(w) > 2]

    return line


def bigram(doc_clean):
    sent = [' '.join(sentence).split() for sentence in doc_clean]
    phrases = Phrases(sent, min_count=10)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    return sentences


def clean_file(directory_stem, folder, filename):
    old_filename = r'.\{}\{}'.format(folder, filename)
    new_filename = 'clean_{}'.format(filename)

    # open up and clean
    doc_dirty = []
    doc_clean = []
    with open(old_filename, 'r', encoding='utf-8') as fp:
        for line in fp:
            doc_dirty.append(line.split())
        fp.close()

    for line in doc_dirty:
        c_line = clean_line(line)
        if c_line and len(c_line) != 0:
            doc_clean.append(c_line)

    # find bigrams
    doc_clean = bigram(doc_clean)

    with open(new_filename, 'w', encoding='utf-8') as fp:
        wr = csv.writer(fp, delimiter=' ')
        wr.writerows(doc_clean)

    source = directory_stem + '\{}'.format(new_filename)
    dest = directory_stem + '\clean_{}\{}'.format(folder, new_filename)
    os.rename(source, dest)

def word_in_doc_pct(new_path):
    docs_with_word = defaultdict(int)
    file_size = 0
    for filename in tqdm(os.listdir(new_path), ascii=True, desc='Finding word counts', leave=True):
        file_size += 1
        # create file set
        file_set = set()
        with open(filename, 'r', encoding='utf-8') as fp:
            for sentence in fp:
                sentence = sentence.rstrip()
                file_set.update([w for w in sentence])
            fp.close()
        for word in file_set:
            docs_with_word[word] += 1
    unsorted_dict = {k: v / file_size for k, v in docs_with_word.items()}
    return {k: v for k, v in sorted(unsorted_dict.items(), key=lambda item: item[1])}


def clean_source(directory_stem, source):
    folder = source
    new_path = directory_stem + '\\' + 'clean_{}'.format(source)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    # clean file
    for filename in tqdm(os.listdir(directory_stem + '\\' + source), ascii=True, desc='Cleaning files', leave=True):
        clean_file(directory_stem, folder, filename)
    return new_path

def clean_corpus(directory_stem, sources):
    source_data = defaultdict(list)
    for source in tqdm(sources, ascii=True, desc='Sources', leave=True):
        if os.path.isfile('data_{}'.format(source)):
            temp = pickle.load(open('data_{}'.format(source), 'rb'))
            source_data[source] = temp
        else:
            clean_source_path = clean_source(directory_stem, source)
            assert os.path.isdir(clean_source_path)

            word_in_doc_pct_dict = word_in_doc_pct(clean_source_path)
            with open('dict_{}.txt'.format(source), 'w') as fp:
                for key, value in word_in_doc_pct_dict.items():
                    fp.write('{}: {}'.format(key, value))

            # for each file in the directory, read sentences into doc, read doc into list
            for file in tqdm(os.listdir(clean_source_path), ascii=True, desc='Files', leave=True):
                with open(r'clean_{}\{}'.format(source, file), encoding='utf-8') as f:
                    contents = [line.rstrip() for line in f]
                source_data[source].append(contents)
            with open('data_{}'.format(source), 'wb') as fp:
                pickle.dump(source_data[source], fp)
    return source_data


def bclean_source(directory_stem, source):
    clean_source_path = directory_stem + '\\' + 'clean_{}'.format(source)
    for file in tqdm(os.listdir(clean_source_path), ascii=True, desc='Files', leave=True):
        with open(r'clean_{}\{}'.format(source, file), mode='rb') as fp:
            doc_clean = []
            for line in fp:
                line = line[0:-3].decode('utf-8') + '\n'
                doc_clean.append(line)

        old_filename = r'clean_{}\{}'.format(source, file)
        new_filename = '{}'.format(file)
        with open(new_filename, 'w', encoding='utf-8') as fp:
            for element in doc_clean:
                fp.write(element)

        new_path = directory_stem + '\\' + 'bclean_{}'.format(source)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        origin = directory_stem + '\{}'.format(new_filename)
        dest = directory_stem + r'\bclean_{}\{}'.format(source, new_filename)
        os.rename(origin, dest)
    return new_path


def bclean_corpus(directory_stem, sources):
    source_data = defaultdict(list)
    for source in tqdm(sources, ascii=True, desc='Sources', leave=True):
        if os.path.isfile('data_{}'.format(source)):
            temp = pickle.load(open('data_{}'.format(source), 'rb'))
            source_data[source] = temp
        else:
            clean_source_path = clean_source(directory_stem, source)
            assert os.path.isdir(clean_source_path)
            bclean_source_path = bclean_source(directory_stem, source)
            assert os.path.isdir(bclean_source_path)

            # generate list
            word_in_doc_pct_dict = word_in_doc_pct(bclean_source_path)
            with open('dict_{}.txt'.format(source), 'w') as fp:
                for key, value in word_in_doc_pct_dict.items():
                    fp.write('{}: {}'.format(key, value))

            # for each file in the directory, read sentences into doc, read doc into list
            for file in tqdm(os.listdir(bclean_source_path), ascii=True, desc='Files', leave=True):
                with open(r'bclean_{}\{}'.format(source, file), encoding='utf-8') as f:
                    contents = [line.rstrip() for line in f]
                source_data[source].append(contents)
            with open('data_{}'.format(source), 'wb') as fp:
                pickle.dump(source_data[source], fp)
    return source_data
