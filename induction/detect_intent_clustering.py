import argparse
import os
import pickle
import json
from collections import Counter
from itertools import combinations
from copy import deepcopy

import editdistance
import spacy
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

nlp = spacy.load("en_core_web_sm")


def get_turn_feats(turn, annotated_corpus, annotated_turns, embeddings):
    def extract_verbs(txt):
        doc = nlp(txt)
        verbs = []
        for tk in doc:
            if tk.pos_.startswith('V'):
                verbs.append(tk.text)
        return verbs

    def _embed(verbs, frames):
        print(verbs, frames)
        if len(verbs) + len(frames) == 0:
            return embeddings.embed_tokens(['unk'])
        verb_e = embeddings.embed_tokens(verbs)
        all_instance_embs = []
        for fr in frames:
            instance_embeddings, counts = zip(*annotated_corpus.frames_dict[fr].instance_embeddings)
            total_instances = sum(counts)
            instance_embedding = sum([count * emb / total_instances for emb, count in zip(instance_embeddings, counts)])
            all_instance_embs.append(instance_embedding)
        frame_e = np.mean(all_instance_embs, axis=0)

        if len(verbs) == 0:
            return embeddings.embed_tokens(['unk'])
        if len(frames) == 0:
            return verb_e
        return verb_e


    return embeddings.embed_tokens([tk.lower() for tk in turn.user.split()])
    turn_frames = {}
    if turn.user.lower() in annotated_turns:
        tagger_annotation = annotated_turns[turn.user.lower()]
        print(turn.user)
        for name, val in tagger_annotation.items():
            tagger_contexts = []
            for rl in turn.role_labeling:
                if val in rl[1]:
                    tagger_contexts.append((rl[0], name, len(rl[1])))
            if len(tagger_contexts) > 0:
                most_specific = [(ctx[0], ctx[1]) for ctx in sorted(tagger_contexts, key=lambda c : c[2])]
                turn_frames[most_specific[0][0]] = most_specific[0][1]
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
            if chunk[0] not in turn_frames:
                turn_frames[chunk[0]] = annotated_corpus._real_frame_name(chunk[1])
            # cluster_assignment.append(cluster)
        if len(turn_frames) == 0:
            verbs = extract_verbs(turn.user)
            frames = tagger_annotation.keys()
        else:
            verbs, frames = zip(*turn_frames.items())
            # predicted_cluster = np.argmax(np.bincount(cluster_assignment))
        return _embed(verbs, frames)


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
    #eval_mapping_c = {2: 'request', 0: 'inform', 1: 'inform', 3: 'inform', 4: 'inform'}
    #eval_mapping_cc = {2: 'inform', 0: 'request', 1: 'inform', 3: 'inform', 4: 'inform'}
    eval_mapping_c = {2: 'inform', 0: 'inform', 1: 'inform', 3: 'request', 4: 'inform'}
    eval_mapping_cc = {2: 'inform', 0: 'inform', 1: 'inform', 3: 'inform', 4: 'request'}
    eval_mapping_ccc = {2: 'inform', 0: 'inform', 1: 'request', 3: 'inform', 4: 'inform'}
    eval_mapping_cccc = {2: 'inform', 0: 'inform', 1: 'inform', 3: 'inform', 4: 'inform'}
    correct = total = 0
    c = cc = ccc = cccc = 0
    with open('annotated_turns.pkl', 'rb') as f:
        annotated_turns = pickle.load(f)
    annotated_corpus.extract_semantic_frames(dataset.turns)
    annotated_corpus.compute_frame_embeddings(embeddings)
    test_set = dataset.dialogues[int(.8 * len(dataset.dialogues)):]
    all_feats = [get_turn_feats(turn, annotated_corpus, annotated_turns, embeddings)
                 for dial in test_set for turn in dial.turns]
    clustering = AgglomerativeClustering(n_clusters=5,
                                             linkage='ward',
                                             affinity='euclidean')
    clustering = clustering.fit(all_feats)

    for dial, label in zip(test_set, clustering.labels_):
        state = {}
        for turn in dial.turns:
            if len(turn.usr_slu) == 0:
                continue
            turn.intent = turn.usr_slu[0].intent
            slu = []
            print(turn.user, turn.intent, label)
            predicted_cluster = label
            cluster_assignment = []
            for s in turn.usr_slu:
                if not s.name in state or state[s.name] != s.val:
                    slu.append(s)
                state[s.name] = s.val
            if len(slu) == 0:
                continue
            #correct += slu[0].intent == eval_mapping[predicted_cluster]
            c += turn.intent == eval_mapping_c[predicted_cluster]
            cc += turn.intent == eval_mapping_cc[predicted_cluster]
            ccc += turn.intent == eval_mapping_ccc[predicted_cluster]
            cccc += turn.intent == eval_mapping_cccc[predicted_cluster]
            total += 1

    print(c, total, c/total)
    print(cc, total, cc/total)
    print(ccc, total, ccc/total)
    print(cccc, total, cccc/total)

