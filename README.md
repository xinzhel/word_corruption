All the script should be run inside the root directory of this project.

## Preparing the Input folder for Clean Data and the Output folder for noisy data
Run the following script to create the folders and prepare SST-2 data. 
```
$ mkdir outputs_local_mixed_noise
$ mkdir data & cd data
$ mkdir sst & mkdir yelp
$ cd sst & wget https://allennlp.s3.amazonaws.com/datasets/sst/test.txt
$ mv ../generate_data.py . & python generate_data.py
```
To download the Yelp data, refer to [the instruction for the CTVAE paper](https://github.com/bidishasamantakgp/CTVAE), and the AG-News data can be directly accessed via the code, which would use the one from huggingface `datasets`.


## Evaluating the Word Corruption of Noisy Data on PLMs
The following script will generate the result as Pandas dataframe.
```
$ dataset_name=sst2
$ model_name=bert-base-uncased-SST-2
$ python word_corruption.py --dataset_name $dataset_name --model_name $model_name --mix_transform
```
We provide Python codes in `check_result.ipynp` to load the dataframe  and re-generate the result  in the paper.

## (Optional) Generating Noisy Sentences
The generated sentences are publicly available in [The Google Cloud]().
However, you can generate new noisy sentences via the `generate_noisy_data.py` script.
By default, the script would identify the context words via the files in the `outputs` folder. But you can re-identify the context words to modify for each of sentences via the `identify_context_word.py` script. 

```
# examples
$ dataset_name=sst2
$ python identify_context_word.py --dataset_name $dataset_name
$ python generate_noisy_data.py --dataset_name $dataset_name --mix_transform
```

