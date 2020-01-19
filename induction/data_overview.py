import json
import argparse

from .dataset import Dataset, CamRestReader, Slot

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
        def transform(name, val):
            if val == '?':
                val = name
                if name == 'phone':
                    val = 'phone number'
                name = 'req-' + name
            return name, val

        for t in dataset.turns:
            print(get_turn_attribute(t, 'intent'))
#            print(get_turn_attribute(t, 'usr_slu'))
            slu = []
#            for s in get_turn_attribute(t, 'usr_slu'):
#                print(s.name, s.val, s.intent)
#                name, val = transform(s.name, s.val)
 #               slu.append(Slot(name, val, 'request' if name.startswith('req') else 'inform'))
#            t.usr_slu = slu
#        dataset.save_dialogues(args.data_fn)
    elif args.output_type == 'filter':
        for d in dataset._dialogues:
            state = {}
            for t in d.turns:
                new_slu = []
                for s in t.usr_slu:
                    if s.name not in state or state[s.name] != s.val:
                        state[s.name] = s.val
                        new_slu.append(Slot(s.name, s.val, s.intent))
                t.usr_slu = new_slu
                print(t.user, t.usr_slu)
#        dataset.save_dialogues(args.data_fn)

    elif args.output_type == 'semantic':
        for t in sorted(dataset.turns, key=lambda t: len(t.user_semantic_parse_semafor)):
            print(t.user)
            print(get_turn_attribute(t, 'user_semantic_parse_semafor'))
            print(get_turn_attribute(t, 'user_semantic_parse_sesame'))
            print_separator()
    elif args.output_type == 'count':
        turns = len([t for t in dataset.turns])
        print(turns)
    else:
        print('Unknown output type {}'.format(args.output_type))
