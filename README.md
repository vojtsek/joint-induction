This is a project for unsupervised detection of semantic slots in natural language sentences.
The code can be found in the *induction/* directory.
It requires python 3.6 with installed libraries listed in *requirements.txt*.

The method can be run in several stages.

# 1. Data preparation:
We prepared the code to work with three datasets:
 - CamRest676 https://github.com/WING-NUS/sequicity/tree/master/data/CamRest676
 - MultiWOZ https://github.com/budzianowski/multiwoz/tree/master/data
 - CamrestSLU https://aspace.repository.cam.ac.uk/handle/1810/248271

Our pipeline works with preprocessed data that are stored in the python pickle format.
There are 3 data specific readers in *induction/dataset.py* file.
To load the data and do initial preprocessing run:

```
python -m induction.syntactic_parse_data --data_fn "dowloaded_dataset_root" --domain [camrest|carslu|woz-hotel] --data_type raw --srl_parse --dep_parse --output "stored_dataset.pickle"
```

# 2. Frame semantic parsing
We use two frame semantic parsers SEMAFOR (http://www.cs.cmu.edu/~ark/SEMAFOR/) and Open-sesame(https://github.com/swabhs/open-sesame).
To obtain the individual utterances run:

```
python -m induction.data_overview --data_fn "stored_dataset.pickle" --data_type pickle --output_type user
```

Then the utterances must be parsed using the respective parser on the side.
To augment the data with semantic parses, run:

```
python -m induction.add_semantic_parse --data_fn "stored_dataset.pickle" --domain [camrest|carslu|woz-hotel] --data_type pickle --output "stored_dataset.pickle" --semantic_parse "parser_output_file" --type [semafor|sesame]
```

# 3. Slot candidates selection
Now we can run the main pipeline selection process:
```
python -m induction.cluster_utterances --data_fn "stored_dataset.pickle" --data_type pickle --domain [camrest|carslu|woz-hotel] --embedding_file data/fasttext_embs.pkl --corpus "resulted_annotated_corpus.pickle" --work_dir "experiment_directory"
```
This step creates the slot candidates and stores them as annotated corpus object in the "experiment\_directory".

# 4. Model training and prediction

```
python -m induction.ner --exp_dir "experiment_directory" --data_dir "experiment_directory/data" --dataset_path "stored_dataset.pickle" --model_dir "directory_to_save_model" --type cnn --threshold THRESHOLD --hidden_size HIDDEN --dropout DROPOUT --train_size SIZE --corpus "experiment_directory/corpus-final.pkl" --domain [camrest|carslu|woz-hotel] --result_file "result_file" [--predict]
```

# 5. Evaluation
```
echo "Average precision"
python -m induction.evaluators --domain [camrest|carslu|woz-hotel] --corpus "experiment_directory/corpus-final.pkl" --result_file "result_file"
echo "Clustering evaluation"
python -m induction.evaluate-merging.py "experiment_directory/corpus-final.pkl" [camrest|carslu|woz-hotel]
```
