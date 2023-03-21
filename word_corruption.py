
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

def calculate_corruption_scores(model_bundle, words, noisy_words):
    
    assert len(words) == 1
    assert words[0] != noisy_words[0]
    tokens, offsets = model_bundle.tokenize_from_words( words )
    tokens = tokens['token_str']
    tokens = [token.translate(token.maketrans('', '', '▁#Ġ')) for token in tokens] # remove substring indicators

    tokens_noisy, offsets_noisy = model_bundle.tokenize_from_words(noisy_words)  
    tokens_noisy = tokens_noisy['token_str']
    tokens_noisy = [token.translate(token.maketrans('', '', '▁#Ġ')) for token in tokens_noisy]
    
    result = {
        'tokens': tokens, 'tokens_noisy': tokens_noisy, 'type': None,
        'overlap_set': None, 'missing_set': None, 'additive_set': None, 
        
        'countM': 0, 'countO': 0, 'countA': 0, 
        # 'countA_unique':0, 
        'countM_by_countS': 0, 'countA_by_char': 0, 'avg_char_in_A':0,
        }

    for i, (offset, offset_noisy) in enumerate(zip(offsets, offsets_noisy)): # for each word 
        
        # word corruption score for a single word
        token_set = tokens[offset[0]:offset[1] + 1]
        token_set_noisy = tokens_noisy[offset_noisy[0]:offset_noisy[1] + 1]

        assert token_set != token_set_noisy # Just in case. I donot think any tokenizer which would tokenizes different words into same tokens

        overlap_set = Multiset(token_set) & Multiset(token_set_noisy) 
        missing_set = Multiset(token_set).difference(overlap_set)
        additive_set = Multiset(token_set_noisy).difference(overlap_set) #(set(token_set) | set(token_set_noisy)).difference(set(token_set))
        
        result['countA'] += len(additive_set) 
        result['countM'] += len(missing_set) 
        if result['countA'] == 0 and result['countM'] == 0 : 
            print('Overlap set: ', overlap_set)
            return None
        # result['countA_unique'] += len(additive_set.distinct_elements()) 
        
        result['countO'] += len(overlap_set) 
        result['avg_char_in_A'] +=  len(''.join(additive_set)) / len(additive_set) if len(additive_set) else -1 # average char length of additive tokens
        result['countM_by_countS'] += len(missing_set) / (len(missing_set)+len(overlap_set)) if len(missing_set)+len(overlap_set) else -1
        
    result['overlap_set'] = dict(overlap_set._elements)
    result['missing_set'] = dict(missing_set._elements)
    result['additive_set'] = dict(additive_set._elements)

    if result['countO'] == 0:
        if result['countM'] == 1:
            result['type'] = 'intact'
        else:
            result['type'] = 'complete'

    elif result['countM'] > 0 and result['countA'] > 0 and result['countO'] > 0:
        result['type'] = 'partial'
    elif result['countM'] == 0:
        assert result['countA'] > 0 and result['countO'] > 0
        result['type'] = 'additive'
    elif result['countA'] == 0:
        assert result['countM'] > 0 and result['countO'] > 0
        result['type'] = 'missing'
    else: 
        print(tokens)
        print(tokens_noisy)
        raise Exception('There is an unexpected corruption type.')
    
    return result

# additive corruption: the impact of additive subwords 单纯的增加additive subwords有影响
# 'sim': 0.8151810765266418, 'wrong_pred': 0}
# ['[CLS]', 'bad', 'd', 'dd', 'dd', 'dd', '[SEP]']

# 'sim': 0.8187543153762817, 'wrong_pred': 0
# ['[CLS]', 'baa', 'ad', '[SEP]']
# sim': 0.8022822141647339, 'wrong_pred': 0
#  ['[CLS]', 'baa', 'a', 'ad', '[SEP]']

# 'sim': 0.7644551396369934, 'wrong_pred': 0
# ['[CLS]', 'baa', 'aaa', 'd', '[SEP]'],
# sim': 0.6670364737510681, 'wrong_pred': 0
# ['[CLS]', 'baa', 'aaa', 'aaa', 'aaa', 'aaa', 'aaa', 'aaa', 'aaa', 'aaa', 'd', '[SEP]']


# 但影响有限, converge to 一个值，我认为这就是  completely lose semantics的界定


# one important additive subwords 会产生背离的semantics,比如 add会有正向含义(Hypothesis: 长的更有可能遭遇worse semantic correlation这是一点)
# sim': 0.03979116678237915, 'wrong_pred': 1
# ['[CLS]', 'ba', 'add', 'd', '[SEP]']
# 'sim': 0.3354472815990448, 'wrong_pred': 0
#  ['[CLS]', '', 'bb', 'bb', 'aaa', 'add', 'd', 'dd', '[SEP]']
# 'sim': 0.6935935616493225, 'wrong_pred': 0}
# ['[CLS]', '', 'bb', 'ba', 'a', 'aaa', 'add', 'dd', 'dd', '[SEP]']

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating...")
    parser.add_argument("--dataset_name", default="pos-abbreviations", type=str,)
    parser.add_argument("--model_name", default="roberta-base-SST-2", type=str,)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name=args.model_name
    output_dir = "data"
    file_name = f'{output_dir}/{dataset_name}.json'

    with open(file_name, 'r') as file:
        noisy_data = json.load(file)

    model_bundle = resource.hf_model_bundles[model_name]
    model_bundle.model = model_bundle.model.to("cuda:0")

    if "neg" in dataset_name:
        label = 0
    elif "pos" in dataset_name:
        label = 1
    else:
        raise Exception("Dataset name is not valid.")

    print('Example: \n')
    result_list = list()
    for (clean_word, noisy_words) in tqdm(noisy_data.items()):
        
        for noisy_word in noisy_words:
            clean_probs = model_bundle.get_probs_from_words( [clean_word],)
            noisy_probs = model_bundle.get_probs_from_words( [noisy_word],)
            result = { 
                    'clean_word': clean_word,
                    'clean_pred': torch.argmax(clean_probs).item() == label,
                    'clean_prob': clean_probs.tolist()[label], # probability for the correct class

                    'noisy_word': noisy_word,
                    'noisy_pred': torch.argmax(noisy_probs).item() == label,
                    'noisy_prob': noisy_probs.tolist()[label], # probability for the correct class}

                    'sim': model_bundle.get_cos_sim(noisy_word,  clean_word).detach().item()
                    }

            # corruption information
            res = calculate_corruption_scores(model_bundle, [clean_word], [noisy_word])
            if res is None:
                print(clean_word+' and '+noisy_word+' have the same segmentation.')
            else:
                result.update(res)
                result_list.append(result)

    file_name = f'{output_dir}/result-{dataset_name}-{model_name}.json'
    print('Saving result into: ', file_name)
    with open(file_name, 'w') as file:
        json.dump(result_list, file, indent=4)
