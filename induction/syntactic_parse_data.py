import json
import argparse

import spacy
import stanfordnlp
from allennlp.predictors import Predictor

from .dataset import Dataset, CamRestReader, MultiWOZReader, MovieReader, AtisReader, CarsluReader


def dependency_parsing(parser):
    def f(turn):
        doc = parser(turn.user)
        turn.user_dependency_parse = []
        turn.user_tokens = []
        turn.sys_dependency_parse = []
        turn.sys_tokens = []
        for sent in doc.sentences:
            for word in sent.words:
                turn.user_tokens.append(word)
                turn.user_dependency_parse.append(
                    (word, sent.words[word.governor-1], word.dependency_relation))
        return
        doc = stanford_nlp(turn.system)
        for sent in doc.sentences:
            for word in sent.words:
                turn.sys_tokens.append(word)
                turn.sys_dependency_parse.append(
                    (word, sent.words[word.governor-1], word.dependency_relation))
    return f


def constituency_parsing(parser):
    def f(turn):
        turn.const_parse_string = ''
        tree = parser.parse(turn.user)
        print(tree)
        for vp_subtree in tree.subtrees(lambda t: t.label().startswith('VP')):
            print([s for s in vp_subtree.subtrees(lambda t: t.label().startswith('VB'))])
            print([s for s in vp_subtree.subtrees(lambda t: t.height() == 2 and not t.label().startswith('V'))])
        turn.const_parse_string += str(tree)
    return f


def semantic_role_label(srl_predictor):
    def f(turn):
        turn.role_labeling = []
        def parse_description(desc):
            verb = []
            arg = []
            for tk in desc.split('['):
                tk = tk.split()
                if len(tk) < 1 or ':' not in tk[0]:
                    continue
                tpe = tk[0].strip(':')
                content = ' '.join(tk[1:])
                content = content[:content.find(']')]
                if tpe == 'V':
                    verb.append(content)
                elif tpe == 'ARGM-NEG':
                    verb.append('not')
                elif tpe == 'ARGM-MOD':
                    verb.append(content)
                elif tpe in ['ARG1', 'ARGM-LOC']:
                    arg.append(content)
            turn.role_labeling.append((' '.join(verb), ' '.join(arg)))
            print((' '.join(verb), ' '.join(arg)))

        prediction = srl_predictor.predict(turn.user)
        for verb_dict in prediction['verbs']:
            # turn.role_label_pairs.append()
            parse_description(verb_dict['description'])
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--domain', type=str, default='camrest')
    parser.add_argument('--data_type', type=str, default='raw')
    parser.add_argument('--const_parse', action='store_true')
    parser.add_argument('--srl_parse', action='store_true')
    parser.add_argument('--dep_parse', action='store_true')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    if args.data_type == 'raw':
        with open(args.data_fn, 'rt') as infd:
            data = json.load(infd)
        if args.domain == 'camrest':
            reader = CamRestReader()
        elif args.domain == 'woz-hotel':
            reader = MultiWOZReader(['hotel'])
        elif args.domain == 'woz-attr':
            reader = MultiWOZReader(['attraction'])
        elif args.domain == 'woz-multi':
            reader = MultiWOZReader(['hotel','restaurant'])
        elif args.domain == 'movies':
            reader = MovieReader()
        elif args.domain == 'atis':
            reader = AtisReader()
        elif args.domain == 'carslu':
            reader = CarsluReader()
        else:
            print('Uknown data domain "{}"'.format(args.domain))
            sys.exit(1)
        dataset = Dataset(data=data, reader=reader)
    else:
        dataset = Dataset(saved_dialogues=args.data_fn)
    print([t.user for t in dataset.turns])
    if args.dep_parse:
        stanford_nlp = stanfordnlp.Pipeline()
        dataset.apply_to_turns(dependency_parsing(stanford_nlp))
    if args.srl_parse:
        srl_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
        dataset.apply_to_turns(semantic_role_label(srl_predictor))

    dataset.save_dialogues(args.output)
