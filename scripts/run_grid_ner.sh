DATA_DIR=exp/3712/
DATASET=data/camrest_dialogues.pkl
for arch in cnn rnn; do
    for size in 16 32 50 100 150 200; do
        for dropout in 0.5 0.6 0.7 0.8 0.85; do
            for train_size in 500 1000 1500 2000; do
                for thr in 0.1 0.15 0.2 0.25; do
                    exp_name="size${size}_drop${dropout}_train${train_size}_thr${thr}_arch-${arch}"
                    echo $exp_name
                    qsub -cwd -j y -N "${exp_name}" -v DATA_DIR="${DATA_DIR}" -v DATASET="${DATASET}" -v arch=$arch -v size=$size -v dropout=$dropout -v train_size=$train_size -v thr=$thr -v exp_name=$exp_name submit_ner.sh
                done
            done
        done
    done
done
