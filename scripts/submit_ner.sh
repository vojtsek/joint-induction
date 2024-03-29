cd /home/hudecek/hudecek/dialogues-semi-sup/joint-induction
source venv-joint-induction/bin/activate
mkdir -p "${DATA_DIR}/${exp_name}"
mkdir -p "results/${domain}"
python -m induction.ner --data_dir "${DATA_DIR}/${exp_name}" --dataset_path "${DATASET}" --model_dir "models/model_${exp_name}" --type $arch --threshold $thr --hidden_size $size --dropout $dropout --train_size $train_size --corpus "${DATA_DIR}/corpus-final.pkl" --domain ${domain}
python -m induction.ner --data_dir "${DATA_DIR}/${exp_name}" --dataset_path "${DATASET}" --model_dir "models/model_${exp_name}" --type $arch --threshold $thr --hidden_size $size --dropout $dropout --train_size $train_size --corpus "${DATA_DIR}/corpus-final.pkl" --result_file "results/${domain}/ner_${exp_name}.txt" --predict --domain ${domain}
