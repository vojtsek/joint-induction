import pickle

selected = set([
    'direction-locative_relation-part_orientational-part_inner_outer',
    'expensiveness',
    'origin-natural_features-custom-temporal_collocation-people_by_origin-stage_of_progress',
    'quantity-contacting-artifact',
    'topic-statement-speak_on_topic',
    'time_vector-sending',
    'locale-dimension',
    ])

merged = {}
for slot in selected:
    frames = slot.split('-')
    for fr in frames:
        merged[fr] = slot

with open('data/faked_corpus.pkl', 'wb') as of:
    pickle.dump(({}, [], [], selected, merged), of)
