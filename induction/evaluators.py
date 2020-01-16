
class SlotEvaluator:
    def __init__(self, name='dummy'):
        self.tp = 0.000001
        self.fp = 0.000001
        self.tn = 0
        self.fn = 0.000001

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def instances(self):
        return self.tp + self.fn


class GenericEvaluator:

    def __init__(self, name, eval_mapping, slot_names):
        self.name = name
        self.eval_mapping = eval_mapping
        self.slot_evaluators = {x: SlotEvaluator() for x in slot_names}

    def add_turn(self, turn_slu, recognized):
        slu_hyp = {}
        already_used_frames = set()
        for name, val in recognized:
            for substr, slot in self.eval_mapping.items():
                if substr in name and not name in already_used_frames:
                    already_used_frames.add(name)
                    slu_hyp[slot] = val.replace('range', 'price').replace('number','phone')
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
