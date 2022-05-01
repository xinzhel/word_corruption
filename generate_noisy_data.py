import resource
import pickle

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
                example[f'x_{noiser_str}'] = noiser.transform(sample, field='x', n=1)[0].get_words('x')
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

    def transform(self, example, wir=None, n_modify=None, pct_modify=None):
        y = example['y']
        if wir:    
            sample = WIRSample(example, wir=wir, n_modify=n_modify, pct_modify=pct_modify)
        else:
            sample = UTSample(example)

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

def generate_noisy_data(text_modifier, dataset_name='ag_news'):
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
    for i in range(len(test_data)):
        noisy_sample = text_modifier.transform(test_data[i], wir=test_wir[i])
        noisy_data.append((i, noisy_sample))
    
        # test_data = test_data.map(text_modifier.wir_transform(pct_modify=None, wir=ag_news_test_wir[i]))
    print('Total examples: ', text_modifier.total_count)
    print('Failure cases: ', text_modifier.fail)
    return noisy_data

    
def generate_noisy_lexicon(text_modifier, n=100): 
    # params:
    #     n: the number of words for each class (positive and negative)
    from resource import get_sentiment_lexicon
    pos_lexicon, neg_lexicon = get_sentiment_lexicon(n=n*4) # 400 neg words, 400 pos words
    assert len(pos_lexicon) > n*1.2

    noisy_data = []

    i = 0
    for x in list(pos_lexicon):
        # if len(sample.get_words('x')) != 3:
        #     continue
        # getattr(sample, 'x').replace_mask([MODIFIED_MASK, MODIFIED_MASK, 0,])
        noisy_sample = text_modifier.transform({'x': x, 'y': 1}, wir=None)

        noisy_data.append( (i, noisy_sample) )
        i += 1
        if len(noisy_data) == 100:
            break

    for x in list(neg_lexicon):
        noisy_sample = text_modifier.transform({'x': x, 'y': 0}, wir=None)
        noisy_data.append( (i, noisy_sample) )
        i += 1
        if len(noisy_data) == 200:
            break

    print(noisy_data[0], noisy_data[100])
    return noisy_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating...")
    parser.add_argument("--dataset_name", default="sst2", type=str,)
    args = parser.parse_args()
    dataset_name = args.dataset_name

    #define noisers
    trans_p  = 1 # `WIRSample` will control pct_modify by preferentially selecting words with high WIR scores
    keyboard = Keyboard(trans_p=trans_p)
    typoswap = TypoSwap(trans_p=trans_p, skip_first_char=True, skip_last_char=True)
    accent = Accent(trans_p=trans_p)
    addvowel = AddVowel(trans_p=trans_p)
    deletevowel = DeleteVowel(trans_p=trans_p)
    noisers = [keyboard, typoswap, accent, addvowel, deletevowel]
    text_modifier = TextModifier(noisers)

    # generate noisy data
    if dataset_name == "sentiment-lexicon":
        noisy_data = generate_noisy_lexicon(text_modifier, n=100)
    else:
        noisy_data = generate_noisy_data(text_modifier, dataset_name)

    with open(f'outputs/{dataset_name}_noisy.pickle', 'wb') as file:
        pickle.dump(noisy_data, file)



