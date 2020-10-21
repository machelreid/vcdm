import torch
import torch.nn as nn
import gensim
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm

print("Loading Word2Vec")
google_wv = KeyedVectors.load_word2vec_format(
    "/path/to/GoogleNews-vectors-negative300.bin",
    binary=True,
    limit=500000,  # faster loading
)


def Word2Vec(list_of_words):

    model = google_wv

    number = 0
    embedding_matrix = np.zeros((len(list_of_words), 300))
    for i, word in tqdm(enumerate(list_of_words)):
        try:
            if word == "<pad>":
                embedding_matrix[i] = np.zeros(300)
            else:
                embedding_matrix[i] = model[word]
                number += 1
        except KeyError:
            embedding_matrix[i] = np.random.uniform(-0.05, 0.05, 300)

    embs = torch.FloatTensor(embedding_matrix)

    print("Loaded Word2Vec")
    print("Vocab Size:", embedding_matrix.shape)
    print("Coverage:", 100 * number / len(list_of_words))
    return embs
