import argparse
import os
import sys
import pickle

import numpy as np
from nltk.corpus import stopwords
from ner.corpus import Corpus
from ner.utils import tokenize, lemmatize
from ner.network import NER
from .annotated_corpus import AnnotatedCorpus

from .dataset import Dataset

class Evaluator:

    class SlotEvaluator:
        def __init__(self, name='dummy'):
            self.tp = 0.000001
            self.fp = 0.000001
            self.tn = 0
            self.fn = 0.000001

        @property
        def precision(self):
            return self.tp / (self.tp + self.fp)

        @property
        def recall(self):
            return self.tp / (self.tp + self.fn)

        @property
        def f1(self):
            return 2 * self.precision * self.recall / (self.precision + self.recall)

        @property
        def instances(self):
            return self.tp + self.fn

    def __init__(self, name):
        self.name = name
        self.eval_mapping = {
            'origin': 'food',
            'expensiveness': 'pricerange',
            'direction': 'area',
            'contacting': 'slot',
            'quantity': 'slot',
            'topic': 'slot'
        }
        self.slot_evaluators = {
            'food': self.SlotEvaluator(),
            'pricerange': self.SlotEvaluator(),
            'area': self.SlotEvaluator(),
            'slot': self.SlotEvaluator(),
        }

    def add_turn(self, turn_slu, recognized):
        slu_hyp = {}
        for name, val in recognized:
            for substr, slot in self.eval_mapping.items():
                if substr in name:
                    slu_hyp[slot] = val
                    break

        for gold_slot, gold_value in list(turn_slu.items()):
            if gold_slot not in slu_hyp:
                self.slot_evaluators[gold_slot].fn += 1
                continue
            if slu_hyp[gold_slot].lower() in gold_value.lower():
                self.slot_evaluators[gold_slot].tp += 1
                del slu_hyp[gold_slot]
                continue
            else:
                self.slot_evaluators[gold_slot].fp += 1
                del slu_hyp[gold_slot]
        for predicted_slot, predicted_value in slu_hyp.items():
            self.slot_evaluators[predicted_slot].fp += 1

    def eval(self, result):
        print(self.name, file=result)
        mean_precision = mean_recall = mean_f1 = 0
        w_mean_precision = w_mean_recall = w_mean_f1 = 0
        for name, evaluator in self.slot_evaluators.items():
            print(name, evaluator.precision, evaluator.recall, evaluator.f1, file=result)
            mean_precision += evaluator.precision
            mean_recall += evaluator.recall
            mean_f1 += evaluator.f1
            total_instances = sum([evaltr.instances for evaltr in self.slot_evaluators.values()])
            w_mean_precision += evaluator.precision * (evaluator.instances) / total_instances
            w_mean_recall += evaluator.recall * (evaluator.instances) / total_instances
            w_mean_f1 += evaluator.f1 * (evaluator.instances) / total_instances
        
        print('mean', mean_precision / len(self.slot_evaluators), mean_recall / len(self.slot_evaluators), mean_f1 / len(self.slot_evaluators), file=result)
        print('weighted-mean', w_mean_precision, w_mean_recall, w_mean_f1, file=result)
        print('-' * 80, file=result)


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
    for token, tag, l in zip(tokens, tags, logits):
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
            predicted_tags.append((tag[2:], token))
    return predicted_tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--train_size', type=int)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.85)
    parser.add_argument('--hidden_size', type=float, default=100)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--type', type=str, default='cnn')
    parser.add_argument('--model_dir', type=str)
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


    if not args.predict:
        if args.corpus is not None:
            dialogue_dataset = Dataset(saved_dialogues=args.dataset_path)
            annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'], data_fn=args.corpus)
            annotated_corpus.get_corpus_srl_iob(args.data_dir, dialogue_dataset.turns, args.train_size)
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
        dialogue_dataset = Dataset(saved_dialogues=args.dataset_path)
    else:
        dataset_dict = prepare_data_dict(args.data_dir)
        corpus = Corpus(dataset_dict, embeddings_file_path=None)
        annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'], data_fn=args.corpus)
        dialogue_dataset = Dataset(saved_dialogues=args.dataset_path)
        network = NER(corpus, verbouse=False, **model_params)

        import tensorflow as tf

        saver = tf.train.Saver()
        saver.restore(network._sess, os.path.join(args.model_dir, 'ner_model.ckpt'))

        model_evaluator = Evaluator('NN MODEL')
        parser_evaluator = Evaluator('PARSER')
        for n, d in enumerate(dialogue_dataset.dialogues[int(.8 * len(dialogue_dataset.dialogues)):]):
            state = {}
            for turn in d.turns:
                turn_slu = {}
                for slot in turn.usr_slu:
                    if not slot.name in state or state[slot.name] != slot.val:
                        turn_slu[slot.name] = slot.val
                    state[slot.name] = slot.val

                by_model = print_predict(turn.user, network, threshold=args.threshold)
                semantics = set(turn.user_semantic_parse_sesame)
                semantics.update(turn.user_semantic_parse_semafor)
                by_parser = [(annotated_corpus._real_frame_name(f[1]), f[0]) for f in semantics]
                model_evaluator.add_turn(turn_slu, by_model)
                parser_evaluator.add_turn(turn_slu, by_parser)
                if n % 30 == 0:
                    model_evaluator.eval(sys.stdout)
                    parser_evaluator.eval(sys.stdout)
        with open(args.result_file, 'wt') as of:
            model_evaluator.eval(of)
            parser_evaluator.eval(of)
        annotated_turns = {}
        for turn in dialogue_dataset.turns:
            by_model = print_predict(turn.user, network, threshold=args.threshold)
            ann = {}
            for recognized in by_model:
                if any([sub in recognized[0] for sub in ['origin','direction','expensiveness']]):
                    ann[recognized[0]] = recognized[1]
            annotated_turns[turn.user.lower()] = ann
        with open('annotated_turns.pkl', 'wb') as of:
            pickle.dump(annotated_turns, of)

