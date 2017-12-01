from gensim.models import KeyedVectors
import numpy as np
from collections import Counter
import random
import pickle

from gensim.models import KeyedVectors


def load_word2vec(path, limit):
    wv = KeyedVectors.load_word2vec_format(path, binary=True, limit=limit)
    wv.init_sims(replace=True)
    return wvm


def prepare_weights(wv, corpus_vocab):
    intr_vocab = {k: wv.vocab[k].index for k in wv.vocab.keys() & corpus_vocab}
    weights = wv.syn0norm[list(intr_vocab.values()), :]
    weights = np.append(weights, [np.random.uniform(
        -0.05, 0.05, weights[0].shape).astype(np.float32)], axis=0)
    vocab = dict(zip(intr_vocab.keys(), range(len(intr_mvocab.keys()))))
    return weights, vocab


# corpus_vocab: '../data/vocab'
def prepare_embedding(googlew2vpath, corpus_vocab_path='../data/vocab', limit=1000000):
    with open(corpus_vocab_path, 'rb') as fp:
        corpus_vocab = pickle.load(fp)
    wv = load_word2vec(googlew2vpath, limit=1000000)
    return prepare_weights(wv, corpus_vocab)

#Code labels by it's index in list labels.
#Return labels and prepare input in format:
#   [
#    [label,[tree, tree, tree, tree, ...]],
#    [label, [tree, tree, tree ...]], ...
#   ]
def prepare_labels(df, labels_column='topics', trees_column='trees', trees_separator='||'):
    labels = list(df['topics'].unique())
    train_texts = [[labels.index(row[labels_column]),
                   [tree.strip() for tree in row[trees_column].split(trees_separator)]]
                   for index, row in df.iterrows()]
    return labels, train_texts
