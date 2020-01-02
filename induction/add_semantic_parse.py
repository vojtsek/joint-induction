import argparse
import json
import networkx as nx
import editdistance
from .dataset import Dataset
from .helpers import read_conll
from .conll09 import VOCDICT, LEMDICT,  FRAMEDICT, LUDICT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--domain', type=str, default='camrest')
    parser.add_argument('--data_type', type=str, default='raw')
    parser.add_argument('--semantic_parse', type=str)
    parser.add_argument('--type', type=str, default='semafor')
    parser.add_argument('--output', type=str)
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
  
    if args.type == 'sesame':
        semantic_parse, _, _ = read_conll(args.semantic_parse)
        vocab = VOCDICT._inttostr
        frame_vocab = FRAMEDICT._inttostr
        lu_vocab = LUDICT._inttostr
        lem_vocab = LEMDICT._inttostr
        previous_sum = -1
        last_sentence = None
        turn_iterator = dataset.turns
        current_parses = []
        total_parses = total_turns = total_sents = total_skips = 0
        for parsed_sentence in semantic_parse:
            total_parses += 1
            sentence = parsed_sentence.sentence
            sentence_sum = sum(sentence.tokens)
            if previous_sum == sentence_sum:
                pass
            elif previous_sum > 0:
                sentence_tokens = [vocab[tk].lower() for tk in last_sentence.tokens]
                while True:
                    turn = next(turn_iterator)
                    total_turns += 1
                    user_tokens = [w.text.lower() for w in turn.user_tokens]
                    distance = editdistance.eval(' '.join(sentence_tokens), ' '.join(user_tokens))
                    if distance / len(' '.join(user_tokens)) < 0.3:
                        total_sents += 1
                        turn.user_semantic_parse_sesame = current_parses
                        current_parses = []
                        break
                    else:
                        total_skips += 1
                        turn.user_semantic_parse_sesame = []
            previous_sum = sentence_sum
            last_sentence = parsed_sentence
            #print(vars(last_sentence))
            targetlu = lu_vocab[last_sentence.lu.id]
            start_token = 0
            for n, tk in enumerate(last_sentence.lemmas):
                if lem_vocab[tk] == targetlu:
                    start_token = n
            targetframe = frame_vocab[last_sentence.frame.id]
            current_parses.append((targetlu, targetframe, start_token))

        for t in turn_iterator:
            t.user_semantic_parse_sesame = []
        print('Processed {0} parses of {1} sentences and skipped {2} turns out of {3} total ({4:.2f}%).'.format(
            total_parses, total_sents, total_skips, total_turns, 100 * total_skips / total_turns))
    elif args.type == 'semafor':
        with open(args.semantic_parse, 'rt') as semf:
            for line, turn in zip(semf, dataset.turns):
                parse = json.loads(line)
                turn.user_semantic_parse_semafor = []
                for frame in parse['frames']:
                    turn.user_semantic_parse_semafor.append((frame['target']['spans'][0]['text'], frame['target']['name'], frame['target']['spans'][0]['start']))
    else:
        print('Unknown parse type: {}'.format(args.type))
    dataset.save_dialogues(args.output)
