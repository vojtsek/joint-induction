import json
import argparse

from .dataset import Dataset, CamRestReader

def print_separator(char='-'):
    print(char * 120)


def get_turn_attribute(turn, attr_name):
    if hasattr(turn, attr_name):
        return getattr(turn, attr_name)
    else:
        return 'Object does not have requested attribute {}'.format(attr_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--domain', type=str, default='camrest')
    parser.add_argument('--data_type', type=str, default='raw')
    parser.add_argument('--output_type', type=str)
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
    if args.output_type == 'user':
        for t in dataset.turns:
            print(get_turn_attribute(t, 'user'))
            print([(s.name, s.intent) for s in get_turn_attribute(t, 'usr_slu')])
    elif args.output_type == 'filter':
        for d in dataset.dialogues:
            for t in d.turns:
                t.user = t.user.strip()
                t.user = t.user.replace('\n', ' ')
                print(get_turn_attribute(t, 'user'))
        dataset.save_dialogues(args.data_fn)

    elif args.output_type == 'semantic':
        for t in sorted(dataset.turns, key=lambda t: len(t.user_semantic_parse_semafor)):
            print(t.user)
            print(get_turn_attribute(t, 'user_semantic_parse_semafor'))
            print(get_turn_attribute(t, 'user_semantic_parse_sesame'))
            print_separator()
    else:
        print('Unknown output type {}'.format(args.output_type))
