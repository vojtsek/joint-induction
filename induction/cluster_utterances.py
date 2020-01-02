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


def compute_statistics_for_turns(turns):
    stats = Counter()
    for t in turns:
        for s in t.semantics:
            stats[s[1]] += 1
    print(stats.most_common(20))

NUM_ITERATIONS=10
NUM_ACCEPTED_FRAMES=10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--domain', type=str, default='camrest')
    parser.add_argument('--data_type', type=str, default='raw')
    parser.add_argument('--embedding_file', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--distance_file', type=str)
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--compute_distances', action='store_true')

    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        os.makedirs(args.work_dir)

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

    dataset_list = [list(dataset.turns)]
    selected_frames = None
    previously_merged = {}
    count_previously_selected = 0
    for iteration in range(NUM_ITERATIONS):
        tmp_dataset_list = []
        new_selected_frames = set()
        stats_frame_cluster = {x: {} for x in range(3)}
        for n, turns in enumerate(dataset_list):
            print(len(turns))
            annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'])
            annotated_corpus.merged_frames = previously_merged
# because of this the frames are preselected and thus some semantics is omitted
            if selected_frames is not None:
                annotated_corpus.selected_frames = selected_frames
            annotated_corpus.extract_semantic_frames(turns, replace_srl=False)
            annotated_corpus.compute_frame_embeddings(embeddings)
            # find merge candidates
            len_before = len(annotated_corpus.merged_frames)
            for fr1, fr2 in combinations(annotated_corpus.frames_dict.values(), 2):
                similarity = fr1.similarity(fr2, embeddings)
                if similarity > 0.9:
                    annotated_corpus.merge_frames(fr1, fr2)
            previously_merged.update(annotated_corpus.merged_frames)
            if len(annotated_corpus.merged_frames) > len_before:
                print('RECOMPUTING')
                annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'])
                annotated_corpus.merged_frames = previously_merged
                annotated_corpus.extract_semantic_frames(turns, replace_srl=False)
                annotated_corpus.compute_frame_embeddings(embeddings)
            # rank
            annotated_corpus.filter_frames(lambda fr: fr.count > 1)
            frame_ranks = list(annotated_corpus.compute_frame_rank())
            annotated_corpus.frame_stats()
            for frame in annotated_corpus.frames_dict.values():
                stats_frame_cluster[n][frame.name] = frame.count
            num_selected = min(round(NUM_ACCEPTED_FRAMES - iteration), len(annotated_corpus.frames_dict))

            selected_frames_this_iteration = []
            _, frame_scores = zip(*frame_ranks)
            limit_score = np.mean(frame_scores) * 5 / 4
            prev_score = -1
            for m, (frame, score) in enumerate(frame_ranks):
                # if m < num_selected or prev_score == score:
                if score < limit_score:
                    selected_frames_this_iteration.append(frame)
                prev_score = score
            new_selected_frames.update(selected_frames_this_iteration)
            print('selected', new_selected_frames)
            annotated_corpus.save(os.path.join(args.work_dir, 'corpus-iter-{}-cluster-{}.pkl'.format(iteration, n)))
        
        selected_frames = new_selected_frames
        annotated_corpus.selected_frames = selected_frames
        annotated_corpus.extract_semantic_frames(dataset.turns)
        print('Iteration {}, {} of {}; selected: {}'.format(iteration, len(selected_frames), count_previously_selected, selected_frames))
        count_previously_selected = len(selected_frames)
        chunks = list(annotated_corpus.get_chunks(dataset.turns, embeddings))
        print('Chunks', len(chunks))
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
        no_clusters = 2
        clustering = AgglomerativeClustering(n_clusters=no_clusters,
                                             linkage='ward',
                                             affinity='euclidean')
        clustering = clustering.fit(data)
        #db_score = davies_bouldin_score(data, clustering.labels_)
        #sil_score = silhouette_score(data, clustering.labels_)
        #print(no_clusters, db_score, sil_score)
        new_datasets = {x: set() for x in range(no_clusters)}
        for chunk, label in zip(chunks, clustering.labels_):
            new_datasets[label].add(chunk.to_turn())
        for d in new_datasets.values():
            if len(d) > 0:
                tmp_dataset_list.append(d)
        dataset_list = tmp_dataset_list
    annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'])
    annotated_corpus.selected_frames = selected_frames
    annotated_corpus.merged_frames = previously_merged
    annotated_corpus.get_corpus_srl_iob(args.work_dir, dataset.turns, 1000)

