# IN: term, model, k
# OUT: context by term and cosine sim
from gensim.models import KeyedVectors

model_name = str(input("Provide corpus: "))
k = 50
path = 'model_{}.txt'.format(model_name)
model = KeyedVectors.load_word2vec_format(path, binary=False)

while True:
    term = str(input("Provide term: "))
    similarity = model.wv.most_similar(positive=[term],
                                       topn=k,
                                       restrict_vocab=len(model.wv.vocab) // 4)
    print(similarity)
