import resource
import pickle
import random
from tqdm import tqdm
from utils import WIRSample, visualize_text_diff, Accent, TypoSwap, AddVowel, DeleteVowel, set_seed, KeyboardNoise
import argparse



#     def wir_transform(self, wir=None, n_modify=None, pct_modify=None):
#         """ Usage: `datasets.Dataset.map(hf_transform)`
#         """
#         def transform(example):   
#             sample = WIRSample(example, wir, n_modify=n_modify, pct_modify=pct_modify)
# 
#             return self._textflint_transform(sample)
#             
#         return transform

def generate_noisy_data(text_modifier, dataset_name='ag_news', mix_transform=True):
    # load data
    dataset, num_labels = resource.datasets[dataset_name]
    test_data = dataset['test']
    test_data = test_data.rename_column('text', 'x')
    test_data = test_data.rename_column('label', 'y')
    print(f"Test data size: {len(test_data)}")

    # load wir
    with open(f'outputs/{dataset_name}_test_wir.pickle', 'rb') as file:
        test_wir = pickle.load( file)
    assert len(test_data) == len(test_wir)
    
    # transform
    noisy_data = []

    assert 'x' in test_data[0].keys()
    failed = 0
    for i in tqdm(range(len(test_data))):
        wir = test_wir[i]
        if not wir: # empty list
            failed += 1
            print(f'No words to moidy for the {i}-th sentences')
            continue
        noisy_sample = text_modifier.transform(
            test_data[i], 
            wir=wir, 
            pct_modify=pct_modify, 
            mix_transform=mix_transform)
        noisy_data.append((i, noisy_sample))

        # pint interval 100
        # if i % 100 == 0:
        #     print(noisy_sample)
    
        # test_data = test_data.map(text_modifier.wir_transform(pct_modify=None, wir=ag_news_test_wir[i]))
    print('Total examples: ', text_modifier.total_count)
    print('# of Sentences with no importance-ranking scores: ', failed)
    print('Failure cases during noising: ', text_modifier.fail)
    print('# of Failed Noising during noising: ', text_modifier.fail_total)
    return noisy_data

    
def generate_noisy_lexicon(text_modifier, n=1000): 
    # params:
    #     n: the number of words for each class (positive and negative)
    from resource import get_sentiment_lexicon
    pos_lexicon, neg_lexicon = get_sentiment_lexicon(n=int(n*1.3)) # 400 neg words, 400 pos words
    assert len(pos_lexicon) > n*1.2

    noisy_data = []

    i = 0
    for x in list(pos_lexicon):
        # if len(sample.get_words('x')) != 3:
        #     continue
        # getattr(sample, 'x').replace_mask([MODIFIED_MASK, MODIFIED_MASK, 0,])
        noisy_sample = text_modifier.transform({'x': x, 'y': 1}, wir=None, mix_transform=False)

        noisy_data.append( (i, noisy_sample) )
        i += 1
        if len(noisy_data) == n/2:
            break

    for x in list(neg_lexicon):
        noisy_sample = text_modifier.transform({'x': x, 'y': 0}, wir=None)
        noisy_data.append( (i, noisy_sample) )
        i += 1
        if len(noisy_data) == n:
            break

    print(noisy_data[0], noisy_data[100])
    return noisy_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating...")
    parser.add_argument("--dataset_name", default="sst2", type=str,)
    parser.add_argument("--pct_modify", default=None, type=int,)
    parser.add_argument("--mix_transform", action='store_true') # default False
    args = parser.parse_args()
    dataset_name = args.dataset_name
    mix_transform = args.mix_transform
    # if mix_transform and dataset_name == "sentiment-lexicon":
    #     mix_transform = False
    if mix_transform:
        print('!!!Would apply mixed noise on sentences (not work for words in lexicon)')
        output_dir = "outputs_local_mixed_noise"
    else:
        output_dir = "outputs_local_single_noise"
    pct_modify = args.pct_modify # especially use to noise all the words in sentences

    #define noisers
    trans_p  = 1 # `WIRSample` will control pct_modify by preferentially selecting words with high WIR scores
    keyboard = KeyboardNoise(trans_p=trans_p, include_upper_case=False, include_numeric=False, include_special_char=False)
    typoswap = TypoSwap(trans_p=trans_p, skip_first_char=True, skip_last_char=True)
    accent = Accent(trans_p=trans_p)
    addvowel = AddVowel(trans_p=trans_p)
    deletevowel = DeleteVowel(trans_p=trans_p)
    

    # generate noisy data
    if dataset_name == "sentiment-lexicon":
        noisers = [keyboard, typoswap, accent, addvowel, deletevowel]
        print(noisers)
        text_modifier = TextModifier(noisers)
        noisy_data = generate_noisy_lexicon(text_modifier, n=400)
    else:
        noisers = [keyboard, typoswap, addvowel, deletevowel]
        print(noisers)
        text_modifier = TextModifier(noisers)
        noisy_data = generate_noisy_data(text_modifier, dataset_name, mix_transform=mix_transform)

  
    if pct_modify is None:
        with open(f'{output_dir}/{dataset_name}_noisy.pickle', 'wb') as file:
            pickle.dump(noisy_data, file)
    else:
        with open(f'{output_dir}/{dataset_name}_noisy_{pct_modify}.pickle', 'wb') as file:
            pickle.dump(noisy_data, file)




# Yelp
# Total examples:  {'keyboard': 1000, 'typoswap': 1000, 'accent': 1000, 'addvowel': 1000, 'deletevowel': 1000}
# Failure cases:  {'typoswap': 98, 'addvowel': 5, 'deletevowel': 5, 'keyboard': 1, 'accent': 1}