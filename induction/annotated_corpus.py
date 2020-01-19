import itertools
import pickle
import functools
import os
import sys
from collections import Counter
import copy

import yake
import networkx as nx
import numpy as np

class SemanticFrame:

    def __init__(self, name):
        self.name = name
        self.instances = Counter()
        self.pos_instances = Counter()
        self.ctx = []
        self.dependencies = Counter()
        self.frame_dependencies = Counter()
        self.heads = Counter()
        self.count = 0
        self.orders = []
        self.coherence = None
        self.instance_embeddings = []
        self.texts = []

    def add_instance(self, word, lemma, pos, ctx):
        self.instances[word] += 1
        self.pos_instances[(lemma, pos)] += 1
        self.count += 1
        self.ctx.append(ctx)

    def add_dependency(self, dep_relation):
        self.dependencies[dep_relation] += 1

    def add_frame_dependency(self, frame, rel):
        self.frame_dependencies[(frame, rel)] += 1

    def add_head(self, relation):
        self.heads[relation] += 1

    def append_text(self, text):
        self.texts.append(text)

    def compute_embeddings(self, embeddings):
        for instance, count in self.instances.items():
            instance_tks = instance.split()
            self.instance_embeddings.append((embeddings.embed_tokens(instance_tks), count))
        if len(self.instance_embeddings) > 0:
            instance_embeddings, _ = zip(*self.instance_embeddings)
            self.content_embedding = np.mean(instance_embeddings, axis=0)
        else:
            self.content_embedding = embeddings.embed_tokens(['unk'])

    def compute_coherence(self, embeddings):
        embedding_pairs = itertools.combinations(self.instance_embeddings, 2)
        coherence = 0
        n = 0
        for e1, e2 in embedding_pairs:
            coherence += embeddings.embedding_similarity(e1[0], e2[0], 'cos') * min(e1[1], e2[1])
            n += min(e1[1], e2[1])
        if n == 1:
            coherence = 0.5
        # coherence divided by num combinations
        coherence = 2 * coherence / n if n > 1 else coherence
        self.coherence = coherence
    
    def print(self, f=sys.stdout):
        score = self.score if hasattr(self, 'score') else 0
        print(self.name, self.coherence, list(self.pos_instances.items()), self.count, self.relative_frequency, self.orders, score, file=f)
        # print('{0}\n\tcoherence:\t{1:.2f}\n\tinstances:\t{2}\n\tcount:\t\t{3}\n\tfrequency:\t{4}\n\torders:\t\t{5}\n\tscore:\t\t{6}'.format(self.name, self.coherence, list(self.pos_instances.items()), self.count, self.relative_frequency, self.orders, score), file=f)
        print(self.dependencies.most_common(5), file=f)
        print(self.heads.most_common(5), file=f)
        for dep in self.frame_dependencies:
            ratio = self.frame_dependencies[dep] / self.count if self.count > 0 else 0
            if ratio > 0.5:
                print(dep, ratio, file=f)
        print(self.frame_dependencies.most_common(5), file=f)

    def similarity(self, other, embeddings):
        if len(self.instances) == 0 or len(other.instances) == 0:
            return 0
        embedding_distance = embeddings.embedding_similarity(self.content_embedding, other.content_embedding, 'cos')
        same_context = 0
        all_dependencies = 0
        for dep1 in self.dependencies.items():
            for dep2 in other.dependencies.items():
                all_dependencies += dep1[1] + dep2[1]
                if dep1[0][1] == dep2[0][1] and dep1[0][2] == dep2[0][2]:
                    same_context += min(dep1[1], dep1[1])
        if all_dependencies == 0:
            return 0
        return embedding_distance + same_context / all_dependencies


class AnnotatedCorpus:
    
    def __init__(self, allowed_pos, data_fn=None):
        self.frames_dict = {}
        self.frame_order_dict = {}
        self.raw_corpus = []
        self.frame_corpus = []
        self.allowed_pos = allowed_pos
        self.selected_frames = None
        self.merged_frames = {}
        if data_fn is not None:
            with open(data_fn, 'rb') as inf:
                self.frames_dict, self.raw_corpus, self.frame_corpus, self.selected_frames, self.merged_frames = pickle.load(inf)

    def _real_frame_name(self, name):
        name = name.lower()
        if name in self.merged_frames:
            return self.merged_frames[name]
        return name

    def extract_semantic_frames(self, turns, replace_srl=True):
        for t in turns:
            turn_text = t.user
            if not hasattr(t, 'semantics'):
                if hasattr(t,' user_semantic_parse_sesame'):
                    semantics = set(t.user_semantic_parse_sesame)
                else:
                    semantics = set()
                semantics.update(t.user_semantic_parse_semafor)
                # semantics.update(t.ner)
                t.semantics = semantics
            dep_parse = t.user_dependency_parse
            if replace_srl:
                self._filter_out_turn_frames(t)
            for fr in t.semantics:
                turn_text = turn_text.replace(fr[0], self._real_frame_name(fr[1]))
                semframe = self.get_frame_reference(self._real_frame_name(fr[1]))
                for w in dep_parse:
                    node0 = node1 = None
                    if w[2] == 'root':
                        continue
                    if w[2] not in self.allowed_pos:
                        continue
                    if w[0].text in fr[0] and w[0].pos in ['NN', 'JJ', 'NNP']:
                        for fr_dep in t.semantics:
                            if fr_dep[0] in [w[1].lemma, w[1].text]:
                                semframe.add_frame_dependency(self._real_frame_name(fr_dep[1]), w[2])
                                break
                        semframe.add_instance(fr[0], w[0].lemma, w[0].pos, t.user)
                        semframe.add_dependency((self._real_frame_name(fr[1]), w[1].lemma, w[2]))
                    elif w[1].text in fr[0]:
                        for fr_dep in t.semantics:
                            if fr_dep[0] in [w[0].lemma, w[0].text]:
                                semframe.add_frame_dependency(self._real_frame_name(fr_dep[1]), w[2])
                                break
                        semframe.add_instance(fr[0], w[1].lemma, w[1].pos, t.user)
                        semframe.add_head((w[0].lemma, self._real_frame_name(fr[1]), w[2]))
                semframe.append_text([self._real_frame_name(fr[1]) if tk.lemma == fr[0] else tk.text for tk in t.user_tokens])
            self.add_utterance(t.user, turn_text)
        self._normalize_frame_frq()

    def compute_frame_rank(self):
        if len(self.frames_dict) == 0:
            return []
        graph_rank = self.compute_graph_rank()
        graph_rank_frames_only = [(k, v) for k, v in graph_rank.items() if k in self.frames_dict]
        graph_based_order = sorted(graph_rank_frames_only, key=lambda f: f[1], reverse=True)
        if len(graph_based_order) > 0:
            graph_based_order, _ = zip(*graph_based_order)
        frq_rank = sorted(self.frames_dict.items(),
            key=lambda f: f[1].relative_frequency, reverse=True)
        frq_based_order, _ = zip(*frq_rank)
        coherence_rank = sorted(self.frames_dict.items(),
            key=lambda f: f[1].coherence, reverse=True)
        alpha = 0.2
        rank = sorted(self.frames_dict.items(),
                      key=lambda f: (1 - alpha) * np.log(f[1].relative_frequency) +
                                    alpha * np.log(f[1].coherence), reverse=True)
        # return rank
        coherence_based_order, _ = zip(*coherence_rank)
        keywords = self.extract_keywords() # sorted already
        keywords_frames_only = [(k, v) for k, v in keywords if k in self.frames_dict]
        if len(keywords_frames_only) == 0:
            kw_based_order = []
        else:
            kw_based_order, _ = zip(*keywords_frames_only)
        for ordering in [kw_based_order, graph_based_order, coherence_based_order, frq_based_order]:
            if len(ordering) != len(self.frames_dict):
                for fr in self.frames_dict:
                    if fr not in ordering:
                        self.frames_dict[fr].orders.append(len(self.frames_dict) - len(ordering))
            for n, frame_name in enumerate(ordering):
                self.frames_dict[frame_name].orders.append(n+1)

        for frame in self.frames_dict.values():
            frame.score = self._order_merging_policy(frame.orders)
        frames_with_scores = [(frame_name, frame.score) for frame_name, frame in self.frames_dict.items()]
        frames_with_scores_sorted = map(lambda x: (x.name, x.score), sorted(self.frames_dict.values(), key=lambda f: f.score))
        return frames_with_scores_sorted

    def merge_frames(self, fr1, fr2):
        if fr1.name in self.merged_frames:
            name1 = self.merged_frames[fr1.name]
        else:
            name1 = fr1.name
        if fr2.name in self.merged_frames:
            name2 = self.merged_frames[fr2.name]
        else:
            name2 = fr2.name
        names = set(name1.split('-') + name2.split('-'))
        new_name = '-'.join(sorted(names))
        print('MERGING ', new_name)
        self.merged_frames[fr1.name] = new_name
        self.merged_frames[fr2.name] = new_name
        for name in names:
            self.merged_frames[name] = new_name

    def get_verb_arg_pairs(self, turns, frame_names):
        for t in turns:
            for role_pair in t.replaced_role_labeling:
                print(role_pair)

    def _order_merging_policy(self, orders):
        orders = list(sorted(orders))[:-1]
        return functools.reduce(lambda a,b: a + b, orders)

    def _coherence_rank_f(self, frame):
        frame = frame[1]
        alpha = .8
        return alpha * frame.coherence + (1 - alpha) * frame.relative_frequency

    def add_utterance(self, text, text_with_frames):
        self.raw_corpus.append(text)
        self.frame_corpus.append(text_with_frames)

    def get_frame_reference(self, frame_name):
        frame_name = frame_name.lower()
        if frame_name in self.frames_dict:
            return self.frames_dict[frame_name]
        else:
            semframe = SemanticFrame(frame_name)
            self.frames_dict[frame_name] = semframe
            return semframe

    def filter_frames(self, pred):
        to_del = []
        for frame_name, frame in self.frames_dict.items():
            if not pred(frame):
                to_del.append(frame_name)
        for del_frame in to_del:
            del self.frames_dict[del_frame]

    def compute_graph_rank(self):
        graph = nx.Graph()
        for semframe in self.frames_dict.values():
            def item_gen():
                for it in semframe.dependencies.items():
                    yield it
                for it in semframe.heads.items():
                    yield it

            for dep, count in item_gen():
                node0 = dep[0].lower()
                node1 = dep[1].lower()
                if graph.has_edge(node0, node1):
                    graph[node0][node1]['weight'] += count
                else:
                    graph.add_edge(node0, node1, weight=count)

        ranks = nx.pagerank(graph)
        return ranks

    def compute_frame_embeddings(self, embeddings):
        for semframe in self.frames_dict.values():
            semframe.compute_embeddings(embeddings)
            semframe.compute_coherence(embeddings)

    def extract_keywords(self):
        kw_extractor = yake.KeywordExtractor(n=1, top=100)
        keywords = kw_extractor.extract_keywords('. '.join(self.frame_corpus))
        return keywords

    def save(self, out_name):
        with open(out_name, 'wb') as outf:
            pickle.dump((self.frames_dict, self.raw_corpus, self.frame_corpus, self.selected_frames, self.merged_frames), outf)

    def _normalize_frame_frq(self):
        for fr in self.frames_dict.values():
            fr.relative_frequency = fr.count / len(self.raw_corpus)

    def frame_stats(self, of):
        print('-'*80, file=of)
        for name, semframe in self.frames_dict.items():
            semframe.print(of)
        print('-'*80, file=of)
    
    def get_corpus_iob(self, out_file):
        with open(out_file, 'wt') as of:
            for raw, parsed in zip(self.raw_corpus, self.frame_corpus):
                last_tk = None
                for tk_raw, tk_parsed in zip(raw.split(), parsed.split()):
                    tk_raw = tk_raw.strip('!?.,')
                    tk_parsed = tk_parsed.strip('!?.,').lower()
                    tag_type =  'I' if tk_parsed == last_tk else 'B'
                    if tk_parsed in self.frames_dict:
                        tag = '{}-{}'.format(tag_type, tk_parsed)
                    else:
                        tag = 'O'
                    last_tk = tk_parsed
                    print(tk_raw, tag, file=of)
                print(file=of)

    def get_corpus_srl_iob(self, out_dir, turns, train_len, selected=None):
        def rank_f(turn):
            semantics = set(turn.user_semantic_parse_semafor + turn.user_semantic_parse_sesame)
            return len([f for f in semantics if self._real_frame_name(f[1]).lower() in self.selected_frames])

        count = 0
        train_written = False
        sorted_turns = sorted(turns, key=lambda t: rank_f(t), reverse=True)
        of = open(os.path.join(out_dir, 'train'), 'wt')
        if selected is None:
            selected = self.selected_frames
        for turn in sorted_turns:
            if count > train_len:
                of.close()
                of =  open(os.path.join(out_dir, 'test'), 'wt')
                count = 0
                train_written = True
            if not train_written:
                count += 1
            for raw, parsed in zip(turn.role_labeling, turn.replaced_role_labeling):
                raw = raw[0] + ' ' + raw[1]
                parsed = parsed[0] + ' ' + parsed[1]
                last_tk = None
                for tk_raw, tk_parsed in zip(raw.split(), parsed.split()):
                    tk_raw = tk_raw.strip('!?.,')
                    tk_parsed = tk_parsed.strip('!?.,').lower()
                    tag_type =  'I' if tk_parsed == last_tk else 'B'
                    if self._real_frame_name(tk_parsed) in selected:
                        tag = '{}-{}'.format(tag_type, self._real_frame_name(tk_parsed))
                    else:
                        tag = 'O'
                    last_tk = tk_parsed
                    print(tk_raw, tag, file=of)
                print(file=of)

    def get_chunks(self, turns, embeddings):
        for turn in turns:
            for fr_name, chunk in self.get_chunks_for_turn(turn, embeddings):
                if not self._real_frame_name(fr_name) in self.frames_dict:
                    continue
                yield RichChunk((chunk[0], self._real_frame_name(fr_name)), embeddings, self.frames_dict[self._real_frame_name(fr_name)], turn)

    def get_chunks_for_turn(self, turn, embeddings):
            candidates = {}
            for chunk in turn.replaced_role_labeling:
                for frame in self.selected_frames:
# because of merged frames
                    for fr in frame.split('-'):
                        fr = fr[0].upper() + fr[1:]
                        if fr in chunk[1] and len(chunk[1]) > 0 and len(chunk[0]) > 0:
                            if fr.lower() in candidates:
                                if len(chunk[1]) < candidates[fr.lower()][1]:
                                    candidates[fr.lower()] = (chunk[0], len(chunk[1]))
                            else:
                                candidates[fr.lower()] = (chunk[0], len(chunk[1]))
            for fr_name, chunk in candidates.items():
                yield self._real_frame_name(fr_name), chunk

    def _filter_out_turn_frames(self, t):
        if self.selected_frames is not None:
            t.semantics = [fr for fr in t.semantics if self._real_frame_name(fr[1].lower()) in self.selected_frames]
        t.replaced_role_labeling = []
        replaced_role_labeling = t.role_labeling
        for fr in t.semantics:
            replaced_role_labeling = [(role[0], role[1].replace(fr[0], fr[1])) for role in replaced_role_labeling]
        t.replaced_role_labeling = replaced_role_labeling


class RichChunk:
    def __init__(self, chunk, embeddings, semframe, turn_reference):
        chunk0 = chunk[0].replace('\'d', 'would').replace('\'s', 'is').lower()
        self.chunk = (chunk0, chunk[1])
        self.embeddings = embeddings
        self.embeddings.distance = 'l2'
        self.verb_e = embeddings.embed_tokens(chunk[0].split())
        self.content_e = embeddings.embed_tokens(chunk[1].split('_'))
        if len(semframe.instance_embeddings) > 0:
            instance_embeddings, counts = zip(*semframe.instance_embeddings)
            total_instances = sum(counts)
            self.instance_embedding = sum([count * emb / total_instances for emb, count in semframe.instance_embeddings])
        else:
            self.instance_embedding = embeddings.embed_tokens(['unk'])
        self.turn_reference = turn_reference

    def distance(self, chunk2):
        verb_similarity = self.embeddings.embedding_similarity(self.verb_e, chunk2.verb_e)
        content_similarity = self.embeddings.embedding_similarity(self.content_e, chunk2.content_e)
        # return 1/verb_similarity
        return 1 / verb_similarity + np.linalg.norm(self.instance_embedding - chunk2.instance_embedding, ord=2)

    def get_feats(self):
        # return self.verb_e
        alpha = .5
        return alpha * self.verb_e + (1 - alpha) * self.instance_embedding
        return np.concatenate((alpha * self.verb_e, (1 - alpha) * self.instance_embedding), axis=0)

    def to_turn(self):
        t = copy.deepcopy(self.turn_reference)
        t.semantics = set([s for s in t.semantics if s[1].lower() in self.chunk[1]])
        t.replaced_role_labeling = t.role_labeling
        for fr in t.semantics:
            t.replaced_role_labeling = [(v, c.replace(fr[0], fr[1])) for v, c in t.replaced_role_labeling]
        return t

