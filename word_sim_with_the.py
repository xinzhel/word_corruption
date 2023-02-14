
import resource
import pickle
import json
import torch

import pandas as pd
import argparse
from tqdm import tqdm
from torch.nn import CosineSimilarity
import numpy as np
from multiset import Multiset
from numpy.linalg import norm

def get_aggregate_sim(file_name):
    with open(file_name, 'r') as file:
        noisy_data = json.load(file)

    model_bundle = resource.hf_model_bundles[model_name]
    model_bundle.model = model_bundle.model.to("cuda:0")

    aggregate_sim = 0

    for (clean_word, _) in tqdm(noisy_data.items()):
        aggregate_sim += model_bundle.get_cos_sim("the",  clean_word).detach().item()
    return aggregate_sim, len(noisy_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating...")
    parser.add_argument("--model_name", default="albert-base-v2-SST-2", type=str,)
    args = parser.parse_args()
    model_name=args.model_name
    output_dir = "data"
    agg_sim, size = 0, 0
    for dataset_name in ["pos-typos.json", "neg-typos.json", "pos-reduplications.json", "neg-reduplications.json"]:
        file_name = f'{output_dir}/{dataset_name}'
        aggregate_sim, l = get_aggregate_sim(file_name)
        agg_sim += aggregate_sim
        size += l
    print("The average sim: ", agg_sim/size)



            


