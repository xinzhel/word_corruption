import resource
from resource import hf_model_bundles
import pickle
import argparse
from utils import WIRSample


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generating...")
    parser.add_argument("--dataset_name", default="sst2", type=str,)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    data, num_labels = resource.datasets[dataset_name]
    test_data = data['test']
    print(test_data[0])
    print(test_data[10])

    # wir for test data
    bert_rank = []
    roberta_rank = []
    albert_rank = []

    for i in range(test_data.num_rows):
        data = test_data[i]

        # bert 
        wir_sample = WIRSample(data, hf_model_bundles[ f'bert-base-uncased-{dataset_name}'] )
        bert_rank.append(wir_sample.wir)

        # robert
        wir_sample = WIRSample(data, hf_model_bundles[f'roberta-base-{dataset_name}'])
        roberta_rank.append(wir_sample.wir)

        # albert
        wir_sample = WIRSample(data, hf_model_bundles[f'albert-base-v2-{dataset_name}'])
        albert_rank.append(wir_sample.wir)

    total_overlap_rate = 0
    masked_accuracy = 0
    overlap_words_all = []
    for r1, r2, r3 in zip(bert_rank, roberta_rank, albert_rank):
        l = int(len(r1) * .5)
        overlap_words = set(r1[:l]) & set(r2[:l]) & set(r2[:l])
        overlap_rate = len(overlap_words) / l
        overlap_words_all.append(list(overlap_words))
        total_overlap_rate +=  overlap_rate
    avg_overlap_rate = total_overlap_rate / len(bert_rank)

    print(avg_overlap_rate)

    with open(f'outputs/{dataset_name}_test_wir.pickle', 'wb') as file:
        pickle.dump(overlap_words_all, file)