import io
import pickle
import random
import numpy as np
from collections import defaultdict, Counter

UNK='_UNKNOWN_'
class Embeddings:
    
    def __init__(self, data_fn, out_fn=None, distance='cos'):
        self.distance = distance
        try:
            with open(data_fn, 'rb') as inf:
                print('Loading embeddings from "{}"'.format(data_fn))
                self._data = pickle.load(inf)
                self.n, self.d = len(self._data), len(self._data['dog'])
        except:
            with io.open(data_fn, 'r', encoding='utf-8', newline='\n', errors='ignore') as inf:
                self.n, self.d = map(int, inf.readline().split())
                print('Failed. Reading {} embeddings from "{}"'.format(self.n, data_fn))
                self._data = {}
                self._data[UNK] = [random.gauss(0, 1) for _ in range(self.d)]
                print('Reading {} vectors of dimension {}'.format(self.n, self.d))
                for line in inf:
                    tokens = line.rstrip().split(' ')
                    self._data[tokens[0]] = list(map(float, tokens[1:]))
            if out_fn is not None:
                with open(out_fn, 'wb') as of:
                    pickle.dump(self._data, of)
    
    def __getitem__(self, key):
        key = UNK if key not in self._data else key
        return self._data[key]

    def embed_tokens(self, tokens, token_weights=None):
        if token_weights is None:
            token_weights = np.ones((len(tokens),))
        tokens = np.array([self[tk] for tk in tokens]).T
        res = np.matmul(tokens, token_weights) / len(tokens)
        return res
    
    def embedding_similarity(self, e1, e2, distance=None):
        distance = self.distance if distance is None else distance
        if distance == 'l1':
            return 1 / np.linalg.norm(e1 - e2, ord=1)
        elif distance == 'l2':
            return 1 / np.linalg.norm(e1 - e2, ord=2)
        else:
            cosf = np.dot(e1, e2) / (np.linalg.norm(e1, ord=2) * np.linalg.norm(e2, ord=2))
            return cosf
