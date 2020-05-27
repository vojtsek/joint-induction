import os
import argparse
import pickle

import numpy as np
from .annotated_corpus import AnnotatedCorpus
from .embeddings import Embeddings

class SlotEvaluator:
    def __init__(self, name='dummy'):
        self.tp = 0.000001
        self.fp = 0.000001
        self.tn = 0
        self.fn = 0.000001

    @property
    def precision(self):
        return round(self.tp) / (self.tp + self.fp)

    @property
    def recall(self):
        return round(self.tp) / (self.tp + self.fn)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall + .000000000000001)

    @property
    def instances(self):
        return self.tp + self.fn


class GenericEvaluator:

    def __init__(self, name, eval_mapping, slot_names):
        self.name = name
        self.eval_mapping = eval_mapping
        self.slot_evaluators = {x: SlotEvaluator() for x in slot_names}

    def add_turn(self, turn_slu, recognized, replace_request=False):
        slu_hyp = {}
        already_used_frames = set()
        for name, val in recognized:
            for substr, slot in self.eval_mapping.items():
                if substr in name and not name in already_used_frames:
                    already_used_frames.add(name)
                    slu_hyp[slot] = val.replace('range', 'price').replace('number','phone')
                    if slot == 'slot' and replace_request:
                        slu_hyp[slot] = '?'
                    break

        for gold_slot, gold_value in list(turn_slu.items()):
            if gold_slot not in self.slot_evaluators:
                continue
            if gold_slot not in slu_hyp:
                self.slot_evaluators[gold_slot].fn += 1
                continue
            if slu_hyp[gold_slot].lower() in gold_value.lower():
                self.slot_evaluators[gold_slot].tp += 1
                del slu_hyp[gold_slot]
                continue
            else:
                self.slot_evaluators[gold_slot].fp += 1
                del slu_hyp[gold_slot]
        for predicted_slot, predicted_value in slu_hyp.items():
            if predicted_slot not in self.slot_evaluators:
                continue
            self.slot_evaluators[predicted_slot].fp += 1

    def eval(self, result):
        print(self.name, file=result)
        mean_precision = mean_recall = mean_f1 = 0
        w_mean_precision = w_mean_recall = w_mean_f1 = 0
        for name, evaluator in self.slot_evaluators.items():
            print(name, evaluator.precision, evaluator.recall, evaluator.f1, file=result)
            mean_precision += evaluator.precision
            mean_recall += evaluator.recall
            mean_f1 += evaluator.f1
            total_instances = sum([evaltr.instances for evaltr in self.slot_evaluators.values()])
            w_mean_precision += evaluator.precision * (evaluator.instances) / total_instances
            w_mean_recall += evaluator.recall * (evaluator.instances) / total_instances
            w_mean_f1 += evaluator.f1 * (evaluator.instances) / total_instances
        
        print('mean', mean_precision / len(self.slot_evaluators), mean_recall / len(self.slot_evaluators), mean_f1 / len(self.slot_evaluators), file=result)
        print('weighted-mean', w_mean_precision, w_mean_recall, w_mean_f1, file=result)
        print('-' * 80, file=result)


def compute_ap(frames, sorted_list):
    spoted = 0
    i = 0
    precision_sum = 0
    cut_off = 20
    for fr_name in sorted_list:
        i += 1
        spoted_any = False
        for fr in fr_name.split('-'):
            if fr in frames:
                spoted += 1
                print(fr, 'adding {}/{}'.format(spoted, i))
                precision_sum += spoted / i
                i += 1
        if spoted == len(frames):
            break
    # return precision_sum
    return precision_sum / len(frames)


def compute_mean_precision(mapping, sorted_list):
    all_mapped_frames = []
    for slot, frames in mapping.items():
        all_mapped_frames.extend(frames)
    print(all_mapped_frames)
    print(sorted_list)
    ap = compute_ap(all_mapped_frames, sorted_list)
    # return np.mean([compute_ap(frames, sorted_list) for _, frames in mapping.items()])
    return ap


camrest_eval_mapping = {
    'origin': 'food',
    'expensiveness': 'pricerange',
    'direction': 'area',
    'contacting': 'slot',
    'quantity': 'slot',
    'topic': 'slot'
}

carslu_eval_mapping = {
    'origin': 'food',
    'expensiveness': 'pricerange',
    'locale_by_use': 'type',
    'direction': 'area',
    'contacting': 'slot',
    'quantity': 'slot',
    'topic': 'slot'
}

movies_eval_mapping = {
#    'spatial_relation', 'timeRange', 'object_location_type',
    'adorning': 'movie_type',
    'building': 'location_name',
    'businesses': 'location_name',
    'experiencer': 'movie_name',
    'people': 'movie_name'
}

atis_eval_mapping = {
    'calendric_unit': 'depart_date.day_name',
    'toloc.city_name': 'toloc.city_name',
    'fromloc.city_name': 'fromloc.city_name',
    'natural_features': 'airline_name',
    'ent_time': 'depart_time.period_mod',
    'ent_time': 'flight_mod',
    'relative_time': 'depart_time.period_mod',
}

woz_hotel_eval_mapping = {
#    'contacting': 'req-phone',
    'contacting': 'slot',
    'buildings': 'type',
#    'quantity': 'req-phone',
    'quantity': 'slot',
    'range': 'req-price',
    'range': 'slot',
#    'speak_on_topic': 'req-address',
    'speak_on_topic': 'slot',
    'direction': 'area',
    'calendric': 'day',
    'expensiveness': 'price',
#    'locale': 'req-area',
    'locale': 'slot',
    'placing': 'parking',
    'nights': 'stay',
    'people': 'people',
    'performers': 'stars',
    'stars': 'stars'
}

woz_attr_eval_mapping = {
    'contacting': 'slot',
    'quantity': 'slot',
    'range': 'slot',
    'speak_on_topic': 'slot',
    'locale': 'slot',
    'part_inner_outer': 'area',
    'locative_relation': 'area',
    'arriving-statement': 'slot',
    'locale_by_use': 'type',
    'buildings': 'type',

}

ap_atis_mapping = {
    'depart_date.day_name': ['calendric_unit', 'ent_date'],
    'toloc.city_name': ['ent_gpe'],
    'fromloc.city_name': ['ent_gpe'],
    'airline_name': ['cause_to_amalgamate', 'natural_features', 'origin', 'part_orientational'],
    'depart_time.period_mod': ['ent_time'],
    'flight_mod': ['ent_time'],
    'depart_time.period_mod': ['relative_time'],
}
ap_woz_mapping = {
    'slot': ['locale', 'commerce_scenario', 'contacting', 'quantity', 'speak_on_topic', 'topic', 'natural_features', 'sending'],
    'type': ['building', ''],
    'area': ['locative_relation', 'locale_by_use', 'direction', 'part_inner_outer', 'part_orientational'],
    'day': ['calendric_unit'],
    'price': ['expensiveness'],
    'stars': ['performers_and_roles'],
    'people': ['people', 'visiting', ],
}

ap_camrest_mapping = {
    'food': ['origin', 'natural_features', 'custom', 'temporal_collocation', 'people_by_origin', 'stage_of_progress'],
    'price': ['expensiveness', 'ordinal_numbers'],
    'area': ['direction', 'locative_relation', 'part_orientational', 'part_inner_outer'],
    'slot': ['quantity', 'contacting', 'artifact', 'topic', 'statement', 'speak_on_topic', 'sending']
}

ap_carslu_mapping = {
    'food': ['origin', 'natural_features', 'custom', 'temporal_collocation', 'people_by_origin', 'stage_of_progress'],
    'price': ['expensiveness', 'ordinal_numbers'],
    'area': ['direction', 'locative_relation', 'part_orientational', 'part_inner_outer'],
    'slot': ['quantity', 'contacting', 'artifact', 'topic', 'statement', 'speak_on_topic', 'sending'],
    'type': ['locale_by_use']
}

ap_attr_mapping = {
    'slot': ['contacting', 'quantity', 'range', 'speak_on_topic', 'topic', 'locale', 'arriving', 'statement'],
    'area': ['locative_relation', 'part_inner_outer', 'direction', 'part_orientational'],
    'type': ['locale_by_use', 'buildings'],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--train_set', type=str)
    parser.add_argument('--embeddings', type=str)
    parser.add_argument('--join', action='store_true')
    args = parser.parse_args()


    if args.domain == 'camrest':
        mapping = ap_camrest_mapping
    elif args.domain == 'carslu':
        mapping = ap_carslu_mapping
    elif args.domain == 'woz-hotel':
        mapping = ap_woz_mapping
    elif args.domain == 'woz-attr':
        mapping = ap_attr_mapping
    elif args.domain == 'atis':
        mapping = ap_atis_mapping

    if args.embeddings is not None:
        embeddings = Embeddings(args.embeddings)
    annotated_corpus = AnnotatedCorpus(allowed_pos=['amod', 'nmod', 'nsubj', 'compound', 'conj'], data_fn=args.corpus)
    with open(os.path.join(args.train_set), 'rb') as f:
        train_set = pickle.load(f)
    selected_frames = annotated_corpus.selected_frames
    #annotated_corpus.extract_semantic_frames(train_set, replace_srl=False)
    #annotated_corpus.compute_frame_embeddings(embeddings)
    #annotated_corpus.compute_frame_rank()
    #selected_frames = [f[0] for f in sorted(annotated_corpus.frames_dict.items(),
     #                                       key=lambda x: x[1].score)]
    mean_ap = compute_mean_precision(mapping, selected_frames)
    print(mean_ap)
    with open(args.result_file, 'wt') as of:
        print(mean_ap, file=of)

