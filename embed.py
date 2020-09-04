import os
import multiprocessing
import align
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm


def fix_model_dimensions(dictionary_of_models):
    base_embed = 0
    for key, value in dictionary_of_models.items():
        if base_embed == 0:
            base_embed = value
        else:
            dictionary_of_models[key] = align.smart_procrustes_align_gensim(base_embed, value)
    return dictionary_of_models


def create_models(source_dictionary, pretrain, align):
    dictionary_of_models = {}
    for k in tqdm(source_dictionary, ascii=True, desc='Creating models'):
        if os.path.isfile('model_{}.txt'.format(str(k))):
            dictionary_of_models[k] = KeyedVectors.load_word2vec_format('model_{}.txt'.format(str(k)), binary=False)
        else:
            if pretrain:
                dictionary_of_models[k] = create_model_pretrained(source_dictionary[k])
            else:
                dictionary_of_models[k] = create_model(source_dictionary[k])
            path_to_save = 'model_{}.txt'.format(str(k))
            dictionary_of_models[k].wv.save_word2vec_format(path_to_save, binary=False)
    if align:
        dictionary_of_models = fix_model_dimensions(dictionary_of_models)
    return dictionary_of_models


def create_model_pretrained(documents, pretrained_path='GoogleNews-vectors-negative300.bin.gz'):
    sentences = [sentence for document in documents for sentence in document]
    sentences_tokenized = [sentence.split() for sentence in sentences]
    if pretrained_path:
        cores = multiprocessing.cpu_count()
        model_2 = Word2Vec(min_count=20,
                           window=20,
                           size=300,
                           sample=6e-5,
                           alpha=.03,
                           min_alpha=.0007,
                           negative=20,
                           workers=cores - 1,
                           sorted_vocab=1)
        model_2.build_vocab(sentences_tokenized)
        total_examples = model_2.corpus_count
        model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        model_2.build_vocab([list(model.vocab.keys())], update=True)
        # todo play with lockf
        model_2.intersect_word2vec_format(pretrained_path, binary=True, lockf=1)
        model_2.train(sentences_tokenized, total_examples=total_examples, epochs=model_2.iter)
    else:
        model_2 = Word2Vec(sentences_tokenized, size=300, min_count=10)
    return model_2


def create_model(documents):
    sentences = [sentence for document in documents for sentence in document]
    sentences_tokenized = [sentence.split() for sentence in sentences]
    cores = multiprocessing.cpu_count()
    model = Word2Vec(min_count=20,
                     window=10,
                     size=300,
                     sample=6e-5,
                     alpha=.025,
                     min_alpha=.0007,
                     negative=20,
                     workers=cores - 1,
                     sorted_vocab=1)
    # t = time()
    model.build_vocab(sentences_tokenized)
    # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    # t = time()
    model.train(sentences_tokenized, total_examples=model.corpus_count, epochs=30)
    # print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    return model
