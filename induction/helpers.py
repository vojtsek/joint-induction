import codecs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
from .conll09 import *
from .sentence import *

def read_conll(conll_file, syn_type=None):
    sys.stderr.write("\nReading {} ...\n".format(conll_file))

    read_depsyn = read_constits = False
    if syn_type == "dep":
        read_depsyn = True
    elif syn_type == "constit":
        read_constits = True
        cparses = read_brackets(CONSTIT_MAP[conll_file])


    examples = []
    elements = []
    missingargs = 0.0
    totalexamples = 0.0

    next_ex = 0
    with codecs.open(conll_file, "r", "utf-8") as cf:
        snum = -1
        for l in cf:
            l = l.strip()
            if l == "":
                if elements[0].sent_num != snum:
                    sentence = Sentence(syn_type, elements=elements)
                    if read_constits:
                        sentence.get_all_parts_of_ctree(cparses[next], CLABELDICT, True)
                    next_ex += 1
                    snum = elements[0].sent_num
                e = CoNLL09Example(sentence, elements)
                examples.append(e)
                if read_depsyn:
                    sentence.get_all_paths_to(sorted(e.targetframedict.keys())[0])
                elif read_constits:
                    sentence.get_cpath_to_target(sorted(e.targetframedict.keys())[0])

                if e.numargs == 0:
                    missingargs += 1

                totalexamples += 1

                elements = []
                continue
            elements.append(CoNLL09Element(l, read_depsyn))
        cf.close()
    sys.stderr.write("# examples in %s : %d in %d sents\n" %(conll_file, len(examples), next_ex))
    sys.stderr.write("# examples with missing arguments : %d\n" %missingargs)
    if read_constits:
        analyze_constits_fes(examples)
    return examples, missingargs, totalexamples


def analyze_constits_fes(examples):
    matchspan = 0.0
    notmatch = 0.0
    matchph = {}
    for ex in examples:
        for fe in ex.invertedfes:
            if fe == FEDICT.getid(EMPTY_LABEL):
                continue
            for span in ex.invertedfes[fe]:
                if span in ex.sentence.constitspans:
                    matchspan += 1
                    phrases = ex.sentence.constitspans[span]
                    for phrase in phrases:
                        if phrase not in matchph:
                            matchph[phrase] = 0
                        matchph[phrase] += 1
                else:
                    notmatch += 1
    tot = matchspan + notmatch
    sys.stderr.write("matches = %d %.2f%%\n"
                     "non-matches = %d %.2f%%\n"
                     "total = %d\n"
                     % (matchspan, matchspan*100/tot, notmatch, notmatch*100/tot, tot))
    sys.stderr.write("phrases which are constits = %d\n" %(len(matchph)))


class BTM:

    def __init__(self, documents, num_topics=20):
        self.documents = documents
        self.num_topics = num_topics
        self.vocab = None
        self.biterms = None
        self.topics = None
        self._analyze_texts()
        self.compute_topics()

    def _analyze_texts(self):
        vec = CountVectorizer(stop_words='english')
        self.X = vec.fit_transform(self.documents).toarray()
        self.vocab = np.array(vec.get_feature_names())
        self.biterms = vec_to_biterms(self.X)

    def compute_topics(self):
        self.btm = oBTM(num_topics=self.num_topics, V=self.vocab)
        self.topics = self.btm.fit_transform(self.biterms, iterations=10)
        topic_summuary(self.btm.phi_wz.T, self.X, self.vocab, 5)
        return self.topics

