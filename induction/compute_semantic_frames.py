import argparse
import networkx as nx
import json
import editdistance
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .dataset import Dataset
from .embeddings import Embeddings
from .annotated_corpus import AnnotatedCorpus
from .helpers import BTM

NUM_ACCEPTED_FRAMES=20

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--domain', type=str, default='camrest')
    parser.add_argument('--data_type', type=str, default='raw')
    parser.add_argument('--embedding_file', type=str)
    parser.add_argument('--corpus_output', type=str)
    parser.add_argument('--dataset_output', type=str)
    parser.add_argument('--output_iob', type=str)
    args = parser.parse_args()

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

    annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'])
    print('Constructing semantic frames.')
    annotated_corpus.extract_semantic_frames(dataset)
    # annotated_corpus.get_corpus_iob('train-corpus.txt')
    print('Computing frame distributed representations')
    embeddings = Embeddings(args.embedding_file)
    annotated_corpus.compute_frame_embeddings(embeddings)
    print('Computing page rankings.')
    annotated_corpus.filter_frames(lambda f: f.count > 1)
    frame_ranks = annotated_corpus.compute_frame_rank()
    selected_frames = [next(frame_ranks) for n in range(NUM_ACCEPTED_FRAMES)]
    annotated_corpus.selected_frames = selected_frames
    print(selected_frames)
    # annotated_corpus.get_verb_arg_pairs(dataset.turns, selected_frames)
    annotated_corpus.get_corpus_srl_iob(args.output_iob, dataset.turns, train_len=500)
    dataset.save_dialogues(args.dataset_output)
    annotated_corpus.save(args.corpus_output)

