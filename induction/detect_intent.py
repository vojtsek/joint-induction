import argparse
import os
import pickle
import json
import random
from collections import Counter
from itertools import combinations
from copy import deepcopy

import editdistance
import networkx as nx
import numpy as np
import spacy
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

nlp = spacy.load('en_core_web_sm')
def extract_turn_chunks_dummy(turn, annotated_turns):
    chunks = []
    tagger_annotations = annotated_turns[turn.user.lower()]
    doc = nlp(turn.user)
    last_verb = None
    for tk in doc:
        if tk.pos_ == 'VERB':
            last_verb = tk.lemma_.lower()
        for tag, val in tagger_annotations.items():
            if tk.text.lower() == val.lower() and last_verb is not None:
                chunks.append((tag, (last_verb, val)))
    return chunks


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
    eval_mapping = {
        0: 'flight',
        1: 'flight',
        2: 'flight',
        3: 'flight',
        4: 'flight',
        5: 'flight',
        6: 'flight',
        7: 'flight',
    }
    eval_mapping_c = {2: 'request', 0: 'inform', 1: 'inform', 3: 'inform', 4: 'inform'}
    #eval_mapping_cc = {2: 'inform', 0: 'request', 1: 'inform', 3: 'inform', 4: 'inform'}
    #eval_mapping_c = {2: 'inform', 0: 'inform', 1: 'inform', 3: 'request', 4: 'inform'}
    eval_mapping_cc = {2: 'inform', 0: 'inform', 1: 'inform', 3: 'inform', 4: 'reqalts'}
    eval_mapping_ccc = {2: 'inform', 0: 'inform', 1: 'request', 3: 'inform', 4: 'inform'}
    eval_mapping_cccc = {2: 'inform', 0: 'inform', 1: 'inform', 3: 'inform', 4: 'inform'}
    correct = total = 0
    c = cc = ccc = cccc = 0
    with open('annotated_turns.pkl', 'rb') as f:
        annotated_turns = pickle.load(f)
    for dial in dataset.dialogues[int(.8 * len(dataset.dialogues)):]:
        state = {}
        for turn in dial.turns:
            if len(turn.usr_slu) == 0:
                continue
            slu = []
            predicted_cluster = 0
            cluster_assignment = []
            print(turn.user, turn.usr_slu[0].intent)
            turn_chunks = list(annotated_corpus.get_chunks_for_turn(turn, embeddings))
            if len(turn_chunks) == 0:
                turn_chunks = extract_turn_chunks_dummy(turn, annotated_turns)
                print('alternative', turn_chunks)
            #if len(turn_chunks) == 0:
             #   continue
            for fr_name, chunk in turn_chunks:
                maxim = 0
                cluster = 0
                chunk = (chunk[0].replace('\'d', 'would').replace('\'s', 'is').lower(), fr_name)
                for cluster_no, dct in cluster_dict.items():
                    if chunk in dct:
                        count = dct[chunk]
                        if count > maxim:
                            maxim = count
                            cluster = cluster_no
                print(fr_name, chunk, cluster)
                cluster_assignment.append(cluster)
                predicted_cluster = np.argmax(np.bincount(cluster_assignment))
            print('state', state)
            for s in turn.usr_slu:
                if not s.name in state or state[s.name] != s.val:
                    slu.append(s)
                state[s.name] = s.val
            print(slu)
            if len(slu) == 0:
                print('cont')
                continue
            print('-' * 80)
            # correct += slu[0].intent == eval_mapping[predicted_cluster]
            c += slu[0].intent == eval_mapping_c[predicted_cluster]
            cc += slu[0].intent == eval_mapping_cc[predicted_cluster]
            ccc += slu[0].intent == eval_mapping_ccc[predicted_cluster]
            cccc += slu[0].intent == eval_mapping_cccc[predicted_cluster]
            # c += slu[0].intent == random.choice(['inform','request'])
            total += 1
    print(c, total, c/total)
    print(cc, total, cc/total)
    print(ccc, total, ccc/total)
    print(cccc, total, cccc/total)

