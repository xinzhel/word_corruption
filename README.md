## Evaluating the Word Corruption of Noisy Data on PLMs
```
$ dataset_name=sst2
$ model_name=bert-base-uncased-SST-2
$ python word_corruption.py --dataset_name $dataset_name --model_name $model_name --mix_transform
```

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

