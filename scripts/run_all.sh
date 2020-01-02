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

