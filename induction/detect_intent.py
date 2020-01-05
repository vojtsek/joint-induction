import argparse
import os
import pickle
import json
from collections import Counter
from itertools import combinations
from copy import deepcopy

import editdistance
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, SpectralBiclustering, SpectralClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


from .dataset import Dataset
from .embeddings import Embeddings
from .annotated_corpus import AnnotatedCorpus

NUM_ITERATIONS=10
NUM_ACCEPTED_FRAMES=10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--domain', type=str, default='camrest')
    parser.add_argument('--data_type', type=str, default='raw')
    parser.add_argument('--embedding_file', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--clusters', type=str)

    args = parser.parse_args()

    embeddings = Embeddings(args.embedding_file)
    if args.data_type == 'raw':
        with open(args.data_fn, 'rt') as infd:
            data = json.load(infd)
        if args.domain == 'camrest':
            reader = CamRestReader()
        else:
            print('Uknown data domain "{}"'.format(args.domain))
            sys.exit(1)
        dataset = Dataset(data=data, reader=reader)
    else:
        dataset = Dataset(saved_dialogues=args.data_fn)

    annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'], data_fn=args.corpus)
    with open(args.clusters, 'rb') as inf:
        cluster_dict = pickle.load(inf)
    eval_mapping = {1: 'request', 0: 'inform', 2: 'inform'}
    correct = total = 0
    for turn in dataset.test_set:
        cluster_assignment = []
        for fr_name, chunk in annotated_corpus.get_chunks_for_turn(turn, embeddings):
            maxim = 0
            cluster = 0
            chunk = (chunk[0].replace('\'d', 'would').replace('\'s', 'is').lower(), fr_name)
            for cluster_no, dct in cluster_dict.items():
                if chunk in dct:
                    count = dct[chunk]
                    if count > maxim:
                        maxim = count
                        cluster = cluster_no
            cluster_assignment.append(cluster)
            predicted_cluster = np.argmax(np.bincount(cluster_assignment))
        correct += turn.usr_slu[0].intent == eval_mapping[predicted_cluster]
        total += 1
    print(correct, total, correct/total)

