import argparse
import os
import sys

import numpy as np
from nltk.corpus import stopwords
from ner.corpus import Corpus
from ner.utils import tokenize, lemmatize
from ner.network import NER

from .dataset import Dataset

def prepare_data_dict(data_path):
    data_types = ['train', 'test', 'valid']
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


def print_predict(sentence, network, f=sys.stdout):
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
            if ratio * l[second_best] > 0.2 and token not in '.?,\':!' and token not in stopwords.words('english'):
                tag = network.corpus.tag_dict.idxs2toks([second_best])[0]
        elif tag.startswith('B') and (token in stopwords.words('english') or token in '.?,\':!'):
            tag = 'O'
        print(token, tag, file=f)
        if tag.startswith('B') and token not in stopwords.words('english') and token not in '.?,\':!':
            predicted_tags.append((tag[2:], token))
    print(predicted_tags)
    return predicted_tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--model_dir', type=str)
    args = parser.parse_args()

    model_params = {"filter_width": 7,
                    "embeddings_dropout": True,
                    "n_filters": [
                        80
                    ],
                    "token_embeddings_dim": 100,
                    "char_embeddings_dim": 25,
                    "use_batch_norm": True,
                    "use_crf": False,
                    "net_type": 'rnn',
                    "cell_type": 'gru',
                    "use_capitalization": True,
                   }

    dataset_dict = prepare_data_dict(args.data_dir)
    corpus = Corpus(dataset_dict, embeddings_file_path=None)
    if not args.predict:
        print_dataset(dataset_dict)
        net = NER(corpus, **model_params)
        learning_params = {'dropout_rate': 0.7,
                           'epochs': 10,
                           'learning_rate': 0.005,
                           'batch_size': 8,
                           'learning_rate_decay': 0.707}
        results = net.fit(**learning_params)
        dialogue_dataset = Dataset(saved_dialogues=args.dataset_path)
        for turn in dialogue_dataset.turns:
            by_model = print_predict(turn.user, net)
            semantics = set(turn.user_semantic_parse_sesame)
            semantics.update(turn.user_semantic_parse_semafor)
            by_parser = [f for f in semantics]
            print('Recognized by parsers: ' + str(by_parser))
#            for role in turn.role_labeling:
#                utt = role[0] + ' ' + role[1]
#                if len(utt) > 2:
#                    print_predict(utt, net)
            print('-' * 80)
    else:
        sentence='I am looking for a cheap restaurant serving chinese food'
        network = NER(corpus, verbouse=False, **model_params)

        import tensorflow as tf

        saver = tf.train.Saver()
        #saver.restore(network._sess,tf.train.latest_checkpoint('model/'))
        saver.restore(network._sess, './model/ner_model.ckpt')

        while True:
            sentence = input('>').strip()
            print_predict(sentence, network)

