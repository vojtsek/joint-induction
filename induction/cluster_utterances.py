import argparse
import os
import sys
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


def save_cluster_stats(chunks, f):
    verb_stats = {}
    for chunk in chunks:
        verb, frame = chunk.chunk
        if not (verb, frame) in verb_stats:
            verb_stats[(verb, frame)] = 1
        else:
            verb_stats[(verb, frame)] += 1
    
    for pair, count in sorted(verb_stats.items(), key=lambda it: it[1]):
        print(pair, count, file=f)
    return verb_stats


def get_cv_folds(dataset, folds):
    chunk_len = round(len(dataset._dialogues) / folds)
    for i in range(folds):
        train_set = list(dataset.turns_from_chunk([x for x in range(i * chunk_len)])) + \
                    list(dataset.turns_from_chunk([x for x in range((i+1) * chunk_len, len(dataset.dialogues))]))
        test_set = list(dataset.turns_from_chunk([x for x in range(i * chunk_len, (i+1) * chunk_len -1 )]))
        print('Train: ' + str(len(train_set)))
        print('Test: ' + str(len(test_set)))
        #train_set = train_set[:round(len(train_set)/2)]
        yield deepcopy(train_set), deepcopy(test_set)


MAX_NUM_ITERATIONS=10
CV_FOLDS=2
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
    parser.add_argument('--no_clusters', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compute_distances', action='store_true')

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

    for t in dataset.turns:
        del t.semantics
    annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'])
    annotated_corpus.extract_semantic_frames(dataset.turns, replace_srl=True)
    with open('frame_stats.txt', 'wt') as f:
        annotated_corpus.frame_stats(f)
    dataset.save_dialogues(args.data_fn)

    dataset.permute(args.seed)
    for fold, (train_set, test_set) in enumerate(get_cv_folds(dataset, CV_FOLDS)):
        work_dir = args.work_dir + '-fold-' + str(fold)
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        with open(os.path.join(work_dir, 'train_set'), 'wb') as f:
            pickle.dump(train_set, f)
        with open(os.path.join(work_dir, 'test_set'), 'wb') as f:
            pickle.dump(test_set, f)
        dataset_list = [train_set]
        selected_frames = None
        previously_merged = {}
        count_previously_selected = 0

        for iteration in range(MAX_NUM_ITERATIONS):
            cluster_verb_stats = {}
            tmp_dataset_list = []
            new_selected_frames = set()
            for n, turns in enumerate(dataset_list):
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
                    similarity -= (len(fr1.name.split('-')) + len(fr2.name.split('-'))) / 40
                    if similarity > 0.9 + (.02) * iteration:
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
                num_selected = min(round(NUM_ACCEPTED_FRAMES - iteration), len(annotated_corpus.frames_dict))

                selected_frames_this_iteration = []
                if len(frame_ranks) > 0:
                    _, frame_scores = zip(*frame_ranks)
                else:
                    frame_scores = []
                limit_score = np.mean(frame_scores) * 4 / 3
                prev_score = -1
                for m, (frame, score) in enumerate(frame_ranks):
                    if score < limit_score:
                        selected_frames_this_iteration.append(frame)
                    prev_score = score
                new_selected_frames.update(selected_frames_this_iteration)
                annotated_corpus.selected_frames = selected_frames_this_iteration
                with open(os.path.join(work_dir, 'cluster_stats-iter-{}-{}.txt'.format(iteration, n)), 'wt') as of:
                    annotated_corpus.frame_stats(of)
                    cluster_verb_stats[len(cluster_verb_stats)] = save_cluster_stats(annotated_corpus.get_chunks(turns, embeddings), of)
                print('selected', new_selected_frames)
                annotated_corpus.save(os.path.join(work_dir, 'corpus-iter-{}-cluster-{}.pkl'.format(iteration, n)))
            
            selected_frames = new_selected_frames
            annotated_corpus.selected_frames = selected_frames
            annotated_corpus.extract_semantic_frames(train_set)
            print('Iteration {}, {} of {}; selected: {}'.format(iteration, len(selected_frames), count_previously_selected, selected_frames))
            if len(selected_frames) == count_previously_selected:
                break
            count_previously_selected = len(selected_frames)
            chunks = list(annotated_corpus.get_chunks(train_set, embeddings))
            print('Chunks', len(chunks))
            data = np.array([chunk.get_feats() for chunk in chunks])
            print('fitting clustering')
            clustering = AgglomerativeClustering(n_clusters=args.no_clusters,
                                                 linkage='ward',
                                                 affinity='euclidean')
            clustering = clustering.fit(data)
            new_datasets = {x: set() for x in range(args.no_clusters)}
            for chunk, label in zip(chunks, clustering.labels_):
                new_datasets[label].add(chunk.to_turn())
            for d in new_datasets.values():
                if len(d) > 0:
                    tmp_dataset_list.append(d)
            dataset_list = tmp_dataset_list
        annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'])
        annotated_corpus.selected_frames = selected_frames
        print('FINAL SELECTED FRAMES:', selected_frames)
        annotated_corpus.merged_frames = previously_merged
        annotated_corpus.get_corpus_srl_iob(work_dir, train_set, 1000)
        annotated_corpus.save(os.path.join(work_dir, 'corpus-final.pkl'.format(iteration, n)))
        with open(os.path.join(work_dir, 'clustering-final.pkl'.format(iteration, n)), 'wb') as of:
            pickle.dump(cluster_verb_stats, of)

