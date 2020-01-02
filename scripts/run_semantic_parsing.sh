set -x
. env.sh
tmp_fn=/tmp/utterances.$$
trap "rm -f $tmp_fn" EXIT
cd "${PROJECT_ROOT}"
python -m induction.data_overview --data_fn $1 --data_type pickle --output_type user > $tmp_fn
deactivate
source "${OPEN_SESAME_ENV}/bin/activate"
cd ${OPEN_SESAME_ROOT}
python -m sesame.targetid --mode predict \
                          --model_name fn1.7-pretrained-targetid \
                          --raw_input $tmp_fn
python -m sesame.frameid --mode predict \
                         --model_name fn1.7-pretrained-frameid \
                         --raw_input predicted-targetid.conll
python -m sesame.argid --mode predict \
                       --model_name fn1.7-pretrained-argid \
                       --raw_input predicted-frameid.conll
