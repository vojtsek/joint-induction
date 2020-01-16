import argparse
import os
import sys
import pickle

import numpy as np
import spacy
from nltk.corpus import stopwords
from ner.corpus import Corpus
from ner.utils import tokenize, lemmatize
from ner.network import NER

from .annotated_corpus import AnnotatedCorpus
from .evaluators import GenericEvaluator, camrest_eval_mapping, movies_eval_mapping, woz_hotel_eval_mapping, carslu_eval_mapping
from .dataset import Dataset

nlp = spacy.load("en_core_web_sm")

def prepare_data_dict(data_path):
    data_types = ['train', 'test']
    dataset_dict = dict()
    for data_type in data_types:

        with open(os.path.join(data_path, data_type), 'rt') as f:
            xy_list = list()
            tokens = list()
            tags = list()
            for line in f:
                items = line.split()
                if len(items) > 1:
                    token, tag = items
                    if token[0].isdigit():
                        tokens.append('#')
                    else:
                        tokens.append(token)
                    tags.append(tag)
                elif len(tokens) > 0:
                    xy_list.append((tokens, tags,))
                    tokens = list()
                    tags = list()
            dataset_dict[data_type] = xy_list
            if data_type == 'test':
                dataset_dict['valid'] = xy_list
    return dataset_dict


def print_dataset(dataset_dict):
    for key in dataset_dict:
        print('Number of samples (sentences) in {:<5}: {}'.format(key, len(dataset_dict[key])))

    print('\nHere is a first two samples from the train part of the dataset:')
    first_two_train_samples = dataset_dict['train'][:2]
    for n, sample in enumerate(first_two_train_samples):
        # sample is a tuple of sentence_tokens and sentence_tags
        tokens, tags = sample
        print('Sentence {}'.format(n))
        print('Tokens: {}'.format(tokens))
        print('Tags:   {}'.format(tags))


def print_predict(sentence, network, f=sys.stdout, threshold=0.2):
    # Split sentence into tokens
    tokens = tokenize(sentence)

    # Lemmatize every token
    tokens_lemmas = lemmatize(tokens)

    tags, logits = network.predict_for_token_batch([tokens_lemmas])
    tags, logits = tags[0], logits[0]
    o_idx = network.corpus.tag_dict.toks2idxs(['O'])


    predicted_tags = []
    last_number = None
    for token, tag, l in zip(tokens, tags, logits):
        if is_number(token.lower()):
            last_number = normalize_num(token)
        second_best =  np.argsort(l)[-2]
        third_best =  np.argsort(l)[-3]
        if tag == 'O':
            ratio = l[second_best] / l[third_best]
            #if ratio * l[second_best] > .2 and token not in '.?,\':!':
            if ratio * l[second_best] > threshold and token not in '.?,\':!' and token not in stopwords.words('english'):
                tag = network.corpus.tag_dict.idxs2toks([second_best])[0]
        elif tag.startswith('B') and (token in stopwords.words('english') or token in '.?,\':!'):
            tag = 'O'
#        print(token, tag, file=f)
        if tag.startswith('B') and token not in stopwords.words('english') and token not in '.?,\':!':
            if 'calendric' in tag and 'night' in token and last_number is not None:
                predicted_tags.append(('nights', last_number))
                last_number = None
            elif 'people' in tag and last_number is not None:
                predicted_tags.append(('people', last_number))
                last_number = None
            elif 'performers' in tag and last_number is not None:
                predicted_tags.append(('stars', last_number))
                last_number = None
            else:
                predicted_tags.append((tag[2:], token))
    return predicted_tags


def augment_parser_numbers(parse, txt):
    parse_dct = {k: v for k, v in parse}
    inv_parse_dct = {v: k for k, v in parse}
    last_number = None
    for tk in txt.split():
        if is_number(tk):
            last_number = normalize_num(tk)
        elif tk.lower() in inv_parse_dct:
            fr = inv_parse_dct[tk.lower()]
            if 'calendric' in fr and 'night' in tk and last_number is not None:
                if fr in parse_dct:
                    del parse_dct[fr]
                parse_dct['nights'] = last_number
                last_number = None
            elif 'people' in fr and last_number is not None:
                if fr in parse_dct:
                    del parse_dct[fr]
                parse_dct['people'] = last_number
                last_number = None
            elif 'performers' in fr and last_number is not None:
                if fr in parse_dct:
                    del parse_dct[fr]
                parse_dct['stars'] = last_number
                last_number = None

    return list(parse_dct.items())


def normalize_num(n):
    known_nos = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    dct = {x: str(y) for y, x in enumerate(known_nos)}
    if n.lower() in dct:
        return dct[n.lower()]
    return n


def is_number(n):
    known_nos = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    n = n.strip().lower()
    return n.isdigit() or n in known_nos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--domain', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--train_size', type=int)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.85)
    parser.add_argument('--hidden_size', type=float, default=100)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--type', type=str, default='cnn')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--result_file', type=str)
    args = parser.parse_args()

    model_params = {"filter_width": 5,
                    "embeddings_dropout": True,
                    "n_filters": [
                        args.hidden_size
                    ],
                    "token_embeddings_dim": 100,
                    "char_embeddings_dim": 25,
                    "use_batch_norm": True,
                    "use_crf": False,
                    "net_type": args.type,
                    "cell_type": 'gru',
                    "use_capitalization": True,
                   }
    if args.domain == 'camrest':
        slot_names = ['food', 'pricerange', 'area', 'slot']
        eval_mapping = camrest_eval_mapping
    elif args.domain == 'movies':
        slot_names = ['spatial_relation', 'timerange', 'object_location_type', 'movie_type', 'location_name', 'object_type', 'movie_name']
        eval_mapping = movies_eval_mapping
    elif args.domain == 'woz-hotel':
        eval_mapping = woz_hotel_eval_mapping
        slot_names = ['req-stars', 'choice', 'req-phone', 'req-address', 'req-area', 'req-price', 'day', 'type', 'area', 'price', 'parking', 'internet', 'people', 'stars', 'stay']
        slot_names = ['slot', 'day', 'type', 'area', 'price', 'people', 'stars', 'stay']
    elif args.domain == 'carslu':
        eval_mapping = carslu_eval_mapping
        slot_names = ['food', 'pricerange', 'area', 'phone', 'type', 'address']
        slot_names = ['food', 'pricerange', 'area', 'slot', 'type']
    else:
        slot_names = []
        eval_mapping = {}

# TODO: replace dialogue dataset with stored turns - train and test

    if not args.predict:
        if args.corpus is not None:
            with open(os.path.join(args.exp_dir, 'train_set'), 'rb') as f:
                train_set = pickle.load(f)
            annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'], data_fn=args.corpus)
            selected = [fr for fr in annotated_corpus.selected_frames if any([el in fr for el in eval_mapping.keys()])]
            print(selected)
            print('Creating corpus in', args.data_dir)
            annotated_corpus.get_corpus_srl_iob(args.data_dir, train_set, args.train_size, selected=selected)
        dataset_dict = prepare_data_dict(args.data_dir)
        corpus = Corpus(dataset_dict, embeddings_file_path=None)
        print_dataset(dataset_dict)
        net = NER(corpus, **model_params)
        learning_params = {'dropout_rate': args.dropout,
                           'epochs': 10,
                           'learning_rate': 0.005,
                           'batch_size': 8,
                           'learning_rate_decay': 0.707,
                           'model_file_path': args.model_dir}
        results = net.fit(**learning_params)
    else:
        dialogue_dataset = Dataset(saved_dialogues=args.dataset_path)
        total = 0
        with open(os.path.join(args.exp_dir, 'test_set'), 'rb') as f:
            test_set = pickle.load(f)
        dataset_dict = prepare_data_dict(args.data_dir)
        corpus = Corpus(dataset_dict, embeddings_file_path=None)
        annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'], data_fn=args.corpus)
        network = NER(corpus, verbouse=False, **model_params)

        import tensorflow as tf

        saver = tf.train.Saver()
        saver.restore(network._sess, os.path.join(args.model_dir, 'ner_model.ckpt'))

        model_evaluator = GenericEvaluator('NN MODEL', eval_mapping, slot_names)
        parser_evaluator = GenericEvaluator('PARSER', eval_mapping, slot_names)
        for n, turn in enumerate(test_set):
            state = {}
            turn_slu = {}
            for slot in turn.usr_slu:
                if slot.intent == 'request':
                    slot.name = 'slot'
                if not slot.name in state or state[slot.name] != slot.val:
                    turn_slu[slot.name] = slot.val
                state[slot.name] = slot.val

            by_model = print_predict(turn.user, network, threshold=args.threshold)
            # doc = nlp(turn.user)
            semantics = set(turn.user_semantic_parse_sesame)
            semantics.update(turn.user_semantic_parse_semafor)
            print(annotated_corpus.selected_frames)
            by_parser = augment_parser_numbers([(annotated_corpus._real_frame_name(f[1]), f[0]) for f in semantics if annotated_corpus._real_frame_name(f[1]) in annotated_corpus.selected_frames], turn.user)
            print(turn.user, turn_slu, by_model)
            by_model = sorted(by_model, key=lambda x: x[0])
            by_parser = sorted(by_parser, key=lambda x: x[0])
            model_evaluator.add_turn(turn_slu, by_model)
            total += len(by_model)
            parser_evaluator.add_turn(turn_slu, by_parser)
            if n % 30 == 0:
                model_evaluator.eval(sys.stdout)
                parser_evaluator.eval(sys.stdout)
        print('writing results')
        print('TOTAL RECOGNIZED', total)
        with open(args.result_file, 'wt') as of:
            model_evaluator.eval(of)
            parser_evaluator.eval(of)
        annotated_turns = {}
        print('annotating turns')
        for turn in dialogue_dataset.turns:
            continue
            by_model = print_predict(turn.user, network, threshold=args.threshold)
            ann = {}
            for recognized in by_model:
                if any([sub in recognized[0] for sub in ['origin','direction','expensiveness']]):
                    ann[recognized[0]] = recognized[1]
            annotated_turns[turn.user.lower()] = ann
        print('writing turns')
#        with open('annotated_turns.pkl', 'wb') as of:
 #           pickle.dump(annotated_turns, of)

