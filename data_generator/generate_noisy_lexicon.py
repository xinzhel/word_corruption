from utils import get_sentiment_lexicon, TypoSwap, KeyboardNoise, TextModifier
import json 
from textflint.common.preprocess import EnProcessor

def generate_noisy_lexicon(words, text_modifier):
    noiser_names = [type(noiser).__name__.lower() for noiser in text_modifier.noisers]

    results = {}
    num_complex_words = 0
    for x in list(words):
        # no complex words, otherwise noiser_sample can be 
        # `{'x': ['2', '-', 'faced'], 'x_keyboardnoise': ['2', '-', 'facfd'], 'x_typoswap': ['2', '-', 'faecd'], 'y': 1}`
        # and `noisy_sample['x'][0]` and `noisy_sample['x_'+noiser_name][0]`` will be the same 
        if len(EnProcessor().tokenize(x, is_one_sent=False, split_by_space=False)) > 1:
            num_complex_words += 1
            continue
        
        noisy_sample = text_modifier.transform({'x': x, 'y': 1}, wir=None, mix_transform=False)
        results[noisy_sample['x'][0]] =[noisy_sample['x_'+name][0] for name in noiser_names if noisy_sample['x_'+name] is not None ] 
    print('Num Complex Words excluded: ', num_complex_words)
    return results

#define noisers
trans_p  = 1 # `WIRSample` will control pct_modify by preferentially selecting words with high WIR scores
keyboard = KeyboardNoise(min_char=5, trans_p=trans_p, include_upper_case=False, include_numeric=False, include_special_char=False)
typoswap = TypoSwap(min_char=5, trans_p=trans_p, skip_first_char=True, skip_last_char=True)
noisers = [keyboard, typoswap, ]
text_modifier = TextModifier(noisers)

pos_lexicon, neg_lexicon = get_sentiment_lexicon(simple_word=None)
neg_noisy_words = generate_noisy_lexicon(neg_lexicon, text_modifier)
json.dump(neg_noisy_words, open('neg-typos.json', 'w'), sort_keys=True, indent=4)

pos_noisy_words = generate_noisy_lexicon(pos_lexicon, text_modifier)
json.dump(pos_noisy_words, open('pos-typos.json', 'w'), sort_keys=True, indent=4)

print('Provided Positive Words: ', len(pos_lexicon)) 
print('Transformed Poistive Words: ', len(pos_noisy_words)) 
print('Provided Negative Words: ', len(neg_lexicon)) 
print('Transformed Negative Words:', len(neg_noisy_words)) 
print('Failed Transformation: ', text_modifier.fail) 

# Num Complex Words excluded:  97
# Num Complex Words excluded:  85
# Provided Positive Words:  710
# Transformed Poistive Words:  625
# Provided Negative Words:  1441
# Transformed Negative Words: 1344
# Failed Transformation:  {'keyboardnoise': 93, 'typoswap': 93}