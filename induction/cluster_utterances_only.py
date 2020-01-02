import argparse
import pickle
import json
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

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--domain', type=str, default='camrest')
    parser.add_argument('--data_type', type=str, default='raw')
    parser.add_argument('--embedding_file', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--distance_file', type=str)
    parser.add_argument('--compute_distances', action='store_true')
    args = parser.parse_args()

    annotated_corpus = AnnotatedCorpus(
                            allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'],
                            data_fn=args.corpus)
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


    dataset_list = [dataset]
    for iteration in range(NUM_ITERATIONS):
        tmp_dataset_list = []
        for dataset in dataset_list:
        annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'])
        # annotated_corpus.filter_frames(lambda f: f.count > 1)
        # frame_ranks = annotated_corpus.compute_frame_rank()
        # selected_frames = [next(frame_ranks) for n in range(NUM_ACCEPTED_FRAMES)]
        chunks = list(annotated_corpus.get_chunks(dataset.turns, embeddings))
        if args.compute_distances:
            print('computing distance matrix')
            distance_matrix = np.zeros((len(chunks), len(chunks)))
            for i in range(len(chunks)):
                for j in range(len(chunks)):
                    if i == j:
                        distance_matrix[i][j] = 0
                    elif distance_matrix[j][i] != 0:
                        distance_matrix[i][j] = distance_matrix[j][i]
                    else:
                        distance_matrix[i][j] = chunks[i].distance(chunks[j])

            with open(args.distance_file, 'wb') as of:
                pickle.dump(distance_matrix, of)
        else:
            with open(args.distance_file, 'rb') as inf:
                distance_matrix = pickle.load(inf)
        data = np.array([chunk.get_feats() for chunk in chunks])
        print('fitting clustering')
        no_clusters = 4
        for no_clusters in range(2,3):
            clustering = AgglomerativeClustering(n_clusters=no_clusters,
                                                 linkage='complete',
                                                 affinity='precomputed')
#    clustering = SpectralClustering(n_clusters=no_clusters, affinity='precomputed')
            #clustering = clustering.fit(distance_matrix)
            clustering = clustering.fit(distance_matrix)
            db_score = davies_bouldin_score(data, clustering.labels_)
            sil_score = silhouette_score(data, clustering.labels_)
            print(no_clusters, db_score, sil_score)
            #plot_dendrogram(clustering, truncate_mode='level', p=3)
            stats_per_class = {}
            stats_per_frame = {}
            for chunk, label in zip(chunks, clustering.labels_):
                if label not in stats_per_class:
                    stats_per_class[label] = {}
                if chunk.chunk[1] not in stats_per_frame:
                    stats_per_frame[chunk.chunk[1]] = {}
                if chunk.chunk[1] not in stats_per_class[label]:
                    stats_per_class[label][chunk.chunk[1]] = 0
                if label not in stats_per_frame[chunk.chunk[1]]:
                    stats_per_frame[chunk.chunk[1]][label] = 0
                stats_per_class[label][chunk.chunk[1]] += 1
                stats_per_frame[chunk.chunk[1]][label] += 1

            for frame, dct in stats_per_frame.items():
                for label, count in dct.items():
                    print(frame, label, count)
            print('-'*100)

