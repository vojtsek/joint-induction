import argparse
import os
import sys

import tensorflow as tf
from ner.corpus import Corpus
from ner.utils import tokenize, lemmatize
from ner.network import NER
from .ner import print_predict, prepare_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--hidden_size', type=float, default=150)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--type', type=str, default='cnn')
    parser.add_argument('--model_dir', type=str)
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

    dataset_dict = prepare_data_dict(args.model_dir)
    corpus = Corpus(dataset_dict, embeddings_file_path=None)
    network = NER(corpus, verbouse=False, **model_params)

    saver = tf.train.Saver()
    saver.restore(network._sess, os.path.join(args.model_dir, 'ner_model.ckpt'))

    while 1:
        utt = input('>')
        by_model = print_predict(utt, network, threshold=args.threshold)
        print(by_model)
