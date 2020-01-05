#! /bin/bash
set -x
. env.sh
#. env-camrest.sh
. env-woz-hotel.sh
# run with ./
WD=`pwd`${0%/*}

cd ${PROJECT_ROOT}
python -m induction.syntactic_parse_data --data_fn "${DATASET_ROOT}" --domain ${DOMAIN} --data_type raw --srl_parse --dep_parse --output "${DATASET_PICKLED}"
cp "${DATASET_PICKLED}" "${DATASET_PICKLED}.bak"
cd ${WD}
bash run_semantic_parsing.sh "${DATASET_PICKLED}"
cd ${PROJECT_ROOT}
cp ${OPEN_SESAME_ROOT}/predicted-args.conll "${SESAME_RESULT}"
python -m induction.add_semantic_parse --data_fn "${DATASET_PICKLED}" --domain ${DOMAIN} --data_type pickle --output "${DATASET_PICKLED}" --semantic_parse "${SEMAFOR_RESULT}" --type semafor
python -m induction.add_semantic_parse --data_fn "${DATASET_PICKLED}" --domain ${DOMAIN} --data_type pickle --output "${DATASET_PICKLED}" --semantic_parse "${SESAME_RESULT}" --type sesame

# creates undex exp directory:
# - corpus-[iteration-x,final].pkl # for use in ner.py and detect_intent.py
# - clustering-final.pkl for use in detect_intent
python -m induction.cluster_utterances --data_fn data/camrest_dialogues.pkl --data_type pickle --domain camrest --embedding_file data/fasttext_embs.pkl --corpus data/annotated_corpus.pkl --work_dir exp/camrest-$$
python -m induction.detect_intent --data_fn data/camrest_dialogues.pkl --data_type pickle --domain camrest --embedding_file data/fasttext_embs.pkl --corpus exp/camrest-3712/corpus-final.pkl --clusters exp/camrest-3712/clustering-final.pkl
# computes annotated turns for usage in SEDST
python -m induction.ner --data_dir exp/3712/size50_drop0.85_train500_thr0.1_arch-cnn --dataset_path data/camrest_dialogues.pkl --model_dir models/model_size50_drop0.85_train500_thr0.1_arch-cnn --hidden_size 50 --dropout 0.85 --train_size 500 --threshold 0.1 --type cnn --corpus exp/3712/corpus-final.pkl --predict --result_file results/test.txt

