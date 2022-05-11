import resource
import pickle
import random
from tqdm import tqdm

from textflint.input.component.sample import UTSample
from utils import WIRSample, visualize_text_diff, Accent, TypoSwap, AddVowel, DeleteVowel
from textflint.generation.transformation.UT import Keyboard, Typos

import argparse



class TextModifier:
    def __init__(self, noisers):
        self.noisers = noisers
        self.total_count = dict()
        self.fail = dict()

    def _textflint_transform(self, sample):

        example = {}
        example['x'] = sample.get_words('x') #sample.get_text('x')
        
        for noiser in self.noisers:
            noiser_str = type(noiser).__name__.lower()
            try:
                # transform text
                noisy_words = noiser.transform(sample, field='x', n=1)
                example[f'x_{noiser_str}'] = noisy_words[0].get_words('x')
                assert len(example['x'] ) == len(example[f'x_{noiser_str}'])

            except IndexError:
                example[f'x_{noiser_str}'] = None
                try: 
                    self.fail[noiser_str] += 1
                except KeyError:
                    self.fail[noiser_str] = 1
                
            try:
                self.total_count[noiser_str] += 1
            except KeyError:
                self.total_count[noiser_str] = 1
        
        return example

    def _mix_transform(self, sample, num_generate = 10):

        example = {}
        example['x'] = sample.get_words('x') #sample.get_text('x')
        wir = sample.wir
        
        for i in range(num_generate):
            generate_txt = []
            for j, word in enumerate(example['x']):
                if j not in wir or (not word.isalpha()):
                    generate_txt.append(word)
                    continue
                while True:
                    # randomly select a noiser
                    noiser = random.choice(self.noisers)
                    # noise the word
                    noisy_word = noiser._get_candidates( word, n=1)
                    if noisy_word and noisy_word[0] != word:
                        generate_txt.append(noisy_word[0])
                        break
                    # print(word, 'to', noisy_word)
                    # print(generate_txt)
          
            example[f'x_noise{i}'] = generate_txt
            assert len(example['x'] ) == len(example[f'x_noise{i}'])

        
        return example

    def transform(self, example, wir=None, n_modify=None, pct_modify=None, mix_transform=True):
        y = example['y']
        if wir:    
            sample = WIRSample(example, wir=wir, n_modify=n_modify, pct_modify=pct_modify)
        else:
            raise Exception
            sample = UTSample(example)
        if mix_transform:
            example = self._mix_transform(sample)
        else:
            example = self._textflint_transform(sample)
        example['y'] = y
        return example

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
    for i in tqdm(range(len(test_data))):
        wir = test_wir[i]
        if not wir: # empty list
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
    print('Failure cases: ', text_modifier.fail)
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
    if mix_transform and dataset_name == "sentiment-lexicon":
        mix_transform = False
    if mix_transform:
        print('!!!Would apply mixed noise on sentences (not work for words in lexicon)')
        output_dir = "outputs_local_mixed_noise"
    else:
        output_dir = "outputs_local_single_noise"
    pct_modify = args.pct_modify # especially use to noise all the words in sentences

    #define noisers
    trans_p  = 1 # `WIRSample` will control pct_modify by preferentially selecting words with high WIR scores
    keyboard = Keyboard(trans_p=trans_p)
    typoswap = TypoSwap(trans_p=trans_p, skip_first_char=True, skip_last_char=True)
    accent = Accent(trans_p=trans_p)
    addvowel = AddVowel(trans_p=trans_p)
    deletevowel = DeleteVowel(trans_p=trans_p)
    

    # generate noisy data
    if dataset_name == "sentiment-lexicon":
        noisers = [keyboard, typoswap, accent, addvowel, deletevowel]
        text_modifier = TextModifier(noisers)
        noisy_data = generate_noisy_lexicon(text_modifier, n=400)
    else:
        noisers = [keyboard, typoswap, addvowel, deletevowel]
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