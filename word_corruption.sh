dataset_name=sst2
for model_name in bert-base-uncased-SST-2 roberta-base-SST-2 albert-base-v2-SST-2; do
    echo $dataset_name
    echo $model_name
    python word_corruption.py --dataset_name $dataset_name --model_name $model_name 
done