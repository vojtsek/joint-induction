import pickle

from collections import Counter

class Dataset:
    def __init__(self, data=None, reader=None, saved_dialogues=None):
        if saved_dialogues is not None:
            with open(saved_dialogues, 'rb') as fd:
                print('Loading data from "{}"'.format(saved_dialogues))
                self.dialogues = pickle.load(fd)
        else:
            self.reader = reader
            self._parse_data(data)

    def _parse_data(self, data):
        self.dialogues = [d for d in self.reader.parse_dialogues(data)]

    def apply_to_dialogues(self, fun):
        for d in self.dialogues:
            fun(d)

    def apply_to_turns(self, fun):
        print('Applying {}'.format(fun))
        for dialogue in self.dialogues:
            for turn in dialogue.turns:
                fun(turn)

    @property
    def turns(self):
        for d in self.dialogues:
            for t in d.turns:
                yield t
    
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
            for t in turns:
                turn = Turn()
                turn.add_user(t['text'])
                turn.add_system('dummy')
                if not 'dialog_act' in t:
                    print('skipping')
                    continue
                slu = self.parse_slu(t['dialog_act'])
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
                sys_turn = next
                dialogue.add_turn(turn)
            yield dialogue

    def parse_slu(self, slu):
        usr_slu = []
        for intent_domain, val in slu.items():
            domain, intent = intent_domain.split('-')
            intent = intent.lower()
            domain = domain.lower()
            if domain not in self.allowed_domains:
                continue
            for s in val:
                slot = Slot(s[0].lower(), s[1], intent)
                usr_slu.append(slot)
        return usr_slu

