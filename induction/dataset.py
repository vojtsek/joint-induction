import pickle
import numpy

from collections import Counter

class Dataset:
    def __init__(self, data=None, reader=None, saved_dialogues=None, train=.6):
        if saved_dialogues is not None:
            with open(saved_dialogues, 'rb') as fd:
                print('Loading data from "{}"'.format(saved_dialogues))
                self._dialogues = pickle.load(fd)
                self.length = len(self._dialogues)
        else:
            self.reader = reader
            self._parse_data(data)
        self.train = train
        self.permutation = list(range(len(self._dialogues)))

    def permute(self, seed=0):
        numpy.random.seed(seed)
        self.permutation = numpy.random.permutation(len(self.dialogues))

    @property
    def dialogues(self):
        dials = []
        for idx in self.permutation:
            dials.append(self._dialogues[idx])
        return dials

    def _parse_data(self, data):
        self._dialogues = [d for d in self.reader.parse_dialogues(data)]
        self.length = len(self._dialogues)

    def apply_to_dialogues(self, fun):
        for d in self.dialogues:
            fun(d)

    def apply_to_turns(self, fun):
        print('Applying {}'.format(fun))
        for dialogue in self.dialogues:
            for turn in dialogue.turns:
                fun(turn)

    def _turns(self, dials):
        for d in dials:
            for t in d.turns:
                yield t

    def turns_from_chunk(self, chunk_idxs):
        for i in chunk_idxs:
            d = self._dialogues[self.permutation[i]]
            for turn in d.turns:
                yield turn

    @property
    def turns(self):
        return self._turns(self.dialogues)
    
    @property
    def test_set(self):
        train_size = round(self.train * self.length)
        return self._turns(self.dialogues[train_size:])

    @property
    def train_set(self):
        train_size = round(self.train * self.length)
        return self._turns(self.dialogues[:train_size])
    
    def user_utterances(self):
        for t in self.turns:
            yield t.user
    
    def save_dialogues(self, output_fn):
        with open(output_fn, 'wb') as fd:
            pickle.dump(self.dialogues, fd)


class Dialogue:
    
    def __init__(self):
        self.turns = []

    def add_turn(self, turn):
        self.turns.append(turn)


class Turn:

    def __init__(self):
        self.user = None
        self.system = None
        self.usr_slu = None
        self.sys_slu = None
        self.parse = None
        self.intent = None
    
    def add_user(self, utt):
        self.user = utt

    def add_system(self, utt):
        self.system = utt

    def add_usr_slu(self, usr_slu):
        self.usr_slu = usr_slu

    def add_sys_slu(self, sys_slu):
        self.sys_slu = sys_slu

    def add_intent(self, intent):
        self.intent = intent


class Slot:

    def __init__(self, name, val, intent):
        self.name = name
        self.val = val
        self.intent = intent


class CamRestReader:
    
    def __init__(self):
        pass

    def parse_dialogues(self, data):
        for dial in data:
            dialogue = Dialogue()
            turns = dial['dial']
            for t in turns:
                turn = Turn()
                turn.add_user(t['usr']['transcript'])
                turn.add_system(t['sys']['sent'])
                slu = self.parse_slu(t['usr']['slu'])
                turn.add_usr_slu(slu)
                intent_counter = Counter()
                for slot in slu:
                    intent_counter[slot.intent] += 1
                if len(intent_counter) > 0:
                    turn.add_intent(intent_counter.most_common(1)[0][0])
                else:
                    turn.add_intent(None)
                dialogue.add_turn(turn)
            yield dialogue

    def parse_slu(self, slu):
        usr_slu = []
        for da in slu:
            for s in da['slots']:
                slot = Slot(s[0], s[1], da['act'])
                usr_slu.append(slot)
        return usr_slu


class MultiWOZReader:
    
    def __init__(self, allowed_domains):
        self.allowed_domains = allowed_domains

    def parse_dialogues(self, data):
        for dial in data.values():
            dialogue = Dialogue()
            turns = dial['log']
            i = 0
            for t in turns:
                i += 1
                if i % 2 == 0:
                    continue
                turn = Turn()
                text = t['text'].strip().replace('\n', ' ')
                turn.add_user(text)
                turn.add_system('dummy')
                if not 'dialog_act' in t:
                    print('skipping')
                    continue
                slu, are_other_domains = self.parse_slu(t['dialog_act'])
                if are_other_domains:
                    continue
                if len(slu) == 0:
                    continue
                turn.add_usr_slu(slu)
                intent_counter = Counter()
                for slot in slu:
                    intent_counter[slot.intent] += 1
                if len(intent_counter) > 0:
                    turn.add_intent(intent_counter.most_common(1)[0][0])
                else:
                    turn.add_intent(None)
                print(turn.user)
                dialogue.add_turn(turn)
            yield dialogue

    def parse_slu(self, slu):
        usr_slu = []
        others = False
        for intent_domain, val in slu.items():
            domain, intent = intent_domain.split('-')
            intent = intent.lower()
            domain = domain.lower()
            if domain not in self.allowed_domains:
                if len(val) > 0:
                    others = True
                continue
            for s in val:
                slot = Slot(s[0].lower(), s[1], intent)
            usr_slu.append(slot)
        return usr_slu, others

class MovieReader:
    
    def __init__(self):
        pass

    def parse_dialogues(self, data):
        for dial in data['SearchScreeningEvent']:
            dialogue = Dialogue()
            text, slu = self.extract_turn(dial['data'])
            text = text.strip().replace('\n', '')
            print(text, slu)
            turn = Turn()
            turn.add_user(text)
            turn.add_system('dummy')
            turn.add_usr_slu(slu)
            dialogue.add_turn(turn)
            yield dialogue

    def extract_turn(self, data):
        text = ''.join([tk['text'] for tk in data])
        entities = [tk for tk in data if 'entity' in tk]
        slu = [ Slot(e['entity'].lower(), e['text'], 'unk') for e in entities]
        return text, slu

class AtisReader:
    
    def __init__(self):
        pass

    def parse_dialogues(self, data):
        for dial in data['rasa_nlu_data']['common_examples']:
            dialogue = Dialogue()
            text = dial['text'].strip().replace('\n', '')
            turn = Turn()
            turn.add_user(text)
            turn.add_system('dummy')
            intent = dial['intent']
            slu = []
            for ent in dial['entities']:
                slu.append(Slot(ent['entity'], ent['value'], intent))
            turn.add_usr_slu(slu)
            dialogue.add_turn(turn)
            yield dialogue

    def extract_turn(self, data):
        text = ''.join([tk['text'] for tk in data])
        entities = [tk for tk in data if 'entity' in tk]
        slu = [ Slot(e['entity'].lower(), e['text'], 'unk') for e in entities]
        return text, slu


class CarsluReader:
    
    def __init__(self):
        pass

    def parse_dialogues(self, data):
        for t in data:
            dialogue = Dialogue()
            turn = Turn()
            if 'text' not in t:
                continue
            turn.add_user(t['text'])
            turn.add_system('dummy')
            intent, slots = t['slu']
            print(slots)
            slu = []
            for sl in slots:
                if len(sl) == 1:
                    slu.append(Slot(sl[0], sl[0], intent))
                else:
                    slu.append(Slot(sl[0], sl[1], intent))
            turn.add_usr_slu(slu)
            dialogue.add_turn(turn)
            yield dialogue

