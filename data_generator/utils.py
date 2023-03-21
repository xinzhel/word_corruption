# coding=utf-8
# Copyright 2022 Xinzhe Li All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nltk
from nltk.corpus import opinion_lexicon
from itertools import islice
from copy import Error
from typing import Iterable, List, Dict, Any
import numpy as np
import torch
import random
import os
from torch import Tensor, backends
from torch.nn.modules.loss import _Loss
from torch.utils.hooks import RemovableHandle
from allennlp.nn import util as nn_util

from textflint.common.settings import MODIFIED_MASK
from textflint.input.component.sample import Sample
from textflint.input.component.field import TextField
from textflint.common.utils.word_op import swap
from textflint.generation.transformation import WordSubstitute
from textflint.common.utils.word_op import get_start_end
from textflint.generation.transformation.UT import Keyboard
from textflint.input.component.sample import UTSample

# for visualize
from textattack.shared.attacked_text import AttackedText
from textattack.shared import utils
from IPython.core.display import display, HTML

# for interpretion
import torch
import math
from transformers.modeling_utils import PreTrainedModel
# from utils import get_grad
# from typing import Union, List, Dict
# import allennlp.nn.util as nn_util
# import numpy as np

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


"""
Visualize
============================================
"""
def visualize_text_diff(words1, words2, color_method=None):
    """Highlights the difference between two texts using color.
    Has to account for deletions and insertions from original text to
    perturbed. Relies on the index map stored in
    ``self.original_result.attacked_text.attack_attrs["original_index_map"]``.
    """

    t1 = AttackedText(' '.join(words1))
    t2 = AttackedText(' '.join(words2))
    t1._words = words1
    t2._words = words2

    if color_method is None:
        return t1.printable_text(), t2.printable_text()

    color_1 = 'blue'
    color_2 = 'purple'

    # iterate through and count equal/unequal words
    words_1_idxs = []
    t2_equal_idxs = set()
    original_index_map = t2.attack_attrs["original_index_map"]
    for t1_idx, t2_idx in enumerate(original_index_map):
        if t2_idx == -1:
            # add words in t1 that are not in t2
            words_1_idxs.append(t1_idx)
        else:
            
            w1 = t1.words[t1_idx]
            w2 = t2.words[t2_idx]
            if w1 == w2:
                t2_equal_idxs.add(t2_idx)
            else:
                words_1_idxs.append(t1_idx)

    # words to color in t2 are all the words that didn't have an equal,
    # mapped word in t1
    words_2_idxs = list(sorted(set(range(t2.num_words)) - t2_equal_idxs))

    # make lists of colored words
    words_1 = [t1.words[i] for i in words_1_idxs]
    words_1 = [utils.color_text(w, color_1, color_method) for w in words_1]
    words_2 = [t2.words[i] for i in words_2_idxs]
    words_2 = [utils.color_text(w, color_2, color_method) for w in words_2]

    t1 = t1.replace_words_at_indices(
        words_1_idxs, words_1
    )
    t2 = t2.replace_words_at_indices(
        words_2_idxs, words_2
    )

    key_color = ("bold", "underline")
    t1 = t1.printable_text(key_color=key_color, key_color_method=color_method)
    t2 = t2.printable_text(key_color=key_color, key_color_method=color_method)
    if color_method == "html":
        display(HTML(t1))
        display(HTML(t2))
    return (t1, t2)

"""
Textflint Utils
============================================
"""
class TextModifier:
    def __init__(self, noisers):
        self.noisers = noisers
        self.total_count = 0
        self.fail = dict()
        self.fail_total = 0

    def _textflint_transform(self, sample):

        example = {}
        example['x'] = sample.get_words('x') #sample.get_text('x')

        
        for noiser in self.noisers:
            noiser_str = type(noiser).__name__.lower()
            try:
                # transform text
                noisy_words = noiser.transform(sample, field='x', n=1)
                example[f'x_{noiser_str}'] = noisy_words[0].get_words('x')
                assert example[f'x_{noiser_str}'] != example['x']
                assert len(example['x'] ) == len(example[f'x_{noiser_str}'])

            except IndexError:
                example[f'x_{noiser_str}'] = None
                try: 
                    self.fail[noiser_str] += 1
                except KeyError:
                    self.fail[noiser_str] = 1
                
            self.total_count += 1
        
        return example

    def _mix_transform(self, sample, num_generate = 10, max_try = 40):
        
        example = {}
        example['x'] = sample.get_words('x') #sample.get_text('x')
        wir = sample.wir
        set_seed(22)
        for i in range(num_generate):
            generate_txt = []
            for j, word in enumerate(example['x']):
                if j not in wir or (not word.isalpha()):
                    generate_txt.append(word)
                    continue
                tried = 0

                while tried < max_try:
                    # randomly select a noiser
                    noiser = random.choice(self.noisers)
                    # print(tried, noiser)
                    # noise the word
                    noisy_word = noiser._get_candidates( word, n=1)
                    if len(noisy_word) > 0 and noisy_word[0] != word:
                        generate_txt.append(noisy_word[0])
                        break
                    # print(word, 'to', noisy_word)
                    # print(generate_txt)
                    tried += 1
                    if tried == max_try:
                        self.fail_total += 1
                        generate_txt.append(word)
            
            assert len(example['x'] ) == len( generate_txt )
            example[f'x_noise{i}'] = generate_txt
            self.total_count += 1

        
        return example

    def transform(self, example, wir=None, n_modify=None, pct_modify=None, mix_transform=True):
        y = example['y']
        if wir:    
            sample = WIRSample(example, wir=wir, n_modify=n_modify, pct_modify=pct_modify)
        else:
            sample = UTSample(example)
        if mix_transform:
            example = self._mix_transform(sample)
        else:
            example = self._textflint_transform(sample)
        example['y'] = y
        return example

"""
Modified Textflint Keyboard 
============================================
"""
class KeyboardNoise(Keyboard):
    def __init__(
        self,
        min_char=5,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.min_char= min_char

    def _get_candidates(self, word, n=5, **kwargs):
        replaced_tokens = []
        chars = self.token2chars(word)
        if len(chars)<self.min_char: # only add noise to words with length >= `min_char``
            return []
        # skip first and last char
        indices = list(range(len(chars)))[1:-1]
        
        valid_chars_idxes = [
            idx for idx in indices if chars[idx] in self.rules.rules and len(
                self.rules.predict(
                    chars[idx])) > 0]
        if not valid_chars_idxes:
            return []

        # putback sampling
        replace_char_idxes = [
            random.sample(
                valid_chars_idxes,
                1)[0] for i in range(n)]
        replace_idx_dic = {}

        for idx in set(replace_char_idxes):
            replace_idx_dic[idx] = replace_char_idxes.count(idx)

        for replace_idx in replace_idx_dic:
            sample_num = replace_idx_dic[replace_idx]
            cand_chars = self.sample_num(
                self.rules.predict(
                    chars[replace_idx]), sample_num)

            for cand_char in cand_chars:
                replaced_tokens.append(
                    self.chars2token(
                        chars[:replace_idx] + [cand_char] + chars[
                                                            replace_idx + 1:]))

        return replaced_tokens

    def __repr__(self):
        return 'KeyboardNoise'

"""
Textflint Accent Transformation
============================================
"""
class AccentRules:
    def __init__(self):
        self.rules = self.get_rules()

    def predict(self, data):
        return self.rules[data]

    @classmethod
    def get_rules(cls):
        #  Diacritics
        mapping = { 
                    # Umlaut
                    'a': ['ä'],  
                    'e': ['é'], 
                    'i': ['ï'], 
                    'o': ['ö'],  
                    'u': ['ü'], 
                    }
        # consonant_mapping = { 
        #             # cedilla
        #             'c': ['ç'],
        #             'f': ['t'], 
        #             'd': ['t'], 
        #             'k': ['g'],
        #             }

        result = {}

        # copy mapping
        for k in mapping:
            result[k] = mapping[k]

        # add reverse mapping
        for k in mapping:
            for v in mapping[k]:
                if v not in result:
                    result[v] = []

                if k not in result[v]:
                    result[v].append(k)

        return result

class Accent(WordSubstitute):
    def __init__(
        self,
        min_char=1,
        trans_min=1,
        trans_max=10,
        trans_p=0.2,
        stop_words=None,
        **kwargs
    ):
        super().__init__(
            min_char=min_char,
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words)

        self.rules = self.get_rules()

    def __repr__(self):
        return 'Ocr'

    def skip_aug(self, tokens, mask, **kwargs):
        remain_idxes = self.pre_skip_aug(tokens, mask)
        token_idxes = []

        for idx in remain_idxes:
            for char in tokens[idx]:
                if char in self.rules.rules and len(
                        self.rules.predict(char)) > 0:
                    token_idxes.append(idx)
                    break

        return token_idxes

    def _get_candidates(self, word, n=1, **kwargs):
        r"""
        Get a list of transformed tokens.
        :param str word: token word to transform.
        :param int n: number of transformed tokens to generate.
        :param kwargs:
        :return list replaced_tokens: replaced tokens list
        """
        replaced_tokens = []
        chars = self.token2chars(word)
        valid_chars_idxes = [
            idx for idx in range(
                len(chars)) if chars[idx] in self.rules.rules and len(
                self.rules.predict(
                    chars[idx])) > 0]
        if not valid_chars_idxes:
            return []

        # putback sampling
        replace_char_idxes = [
            random.sample(
                valid_chars_idxes,
                1)[0] for i in range(n)]
        replace_idx_dic = {}

        for idx in set(replace_char_idxes):
            replace_idx_dic[idx] = replace_char_idxes.count(idx)

        for replace_idx in replace_idx_dic:
            sample_num = replace_idx_dic[replace_idx]
            cand_chars = self.sample_num(
                self.rules.predict(
                    chars[replace_idx]), sample_num)

            for cand_char in cand_chars:
                replaced_tokens.append(
                    self.chars2token(
                        chars[:replace_idx] + [cand_char]
                        + chars[replace_idx + 1:]
                    )
                )

        return replaced_tokens

    @classmethod
    def get_rules(cls):
        return AccentRules()

"""
Textflint TypoSwap Transformation
============================================
"""
class TypoSwap(WordSubstitute):

    def __init__(
        self,
        min_char=5,
        trans_min=1,
        trans_max=10,
        trans_p=0.3,
        stop_words=None,
        skip_first_char=True,
        skip_last_char=True,
        **kwargs
    ):

        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words)
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char
        self.min_char = min_char

    def skip_aug(self, tokens, mask, **kwargs):
        return self.pre_skip_aug(tokens, mask)

    def _get_candidates(self, word, n=5, **kwargs):
        chars = self.token2chars(word)
        if len(chars)<self.min_char: # only add noise to words with length >= `min_char``
            return []
        # and there are at least two distinct characters in the middle
        if len(set(word[1:-1])) < 2:
            return []

        candidates = set()

        for i in range(n):
            result = word
            while result == word:
                # default operate at most one characte r in a word
                result = swap(
                    word, 1, self.skip_first_char, self.skip_last_char)
            if result:
                candidates.add(result)

        if len(candidates) > 0:
            return list(candidates)
        else:
            return []

"""
Textflint AddVowel Transformation
============================================
"""
class AddVowel(WordSubstitute):

    def __init__(
        self,
        trans_min=1,
        trans_max=10,
        trans_p=0.3,
        stop_words=None,
        skip_first_char=True,
        skip_last_char=True,
        **kwargs
    ):

        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words)
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char
    
    def __repr__(self):
        return 'AddVowel'

    def skip_aug(self, tokens, mask, **kwargs):
        return self.pre_skip_aug(tokens, mask)
    
    def _get_candidates(self, word, n=5, **kwargs):
        # TODO: Generate multiple distinct condidates
        assert n == 1 
        candidates = set()

        result = edit_vowel(word, 1, self.skip_first_char, self.skip_last_char, mode='add')
        if result is not None:
            candidates.add(result)

        if len(candidates) > 0:
            return list(candidates)
        else:
            return []
    
class DeleteVowel(WordSubstitute):

    def __init__(
        self,
        trans_min=1,
        trans_max=10,
        trans_p=0.3,
        stop_words=None,
        skip_first_char=True,
        skip_last_char=True,
        **kwargs
    ):

        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words)
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char
    
    def __repr__(self):
        return 'DeleteVowel'

    def skip_aug(self, tokens, mask, **kwargs):
        return self.pre_skip_aug(tokens, mask)
    
    def _get_candidates(self, word, n=5, **kwargs):
        # TODO: Generate multiple distinct condidates
        assert n == 1 
        candidates = set()

        result = edit_vowel(word, 1, self.skip_first_char, self.skip_last_char, mode='delete')
        if result is not None:
            candidates.add(result)

        if len(candidates) > 0:
            return list(candidates)
        else:
            return []

def edit_vowel(word, num=1, skip_first=False, skip_last=False, mode='add'):
    # TODO: support adding/deleting multiple vowels
    assert num == 1
    if len(word) <= 1:
        return None

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    # find the vowel chars
    find_char = None
    for char_idx in list(range(start, end+1)):
        if chars[char_idx] in ['a', 'e', 'i', 'o', 'u']:
            find_char = chars[char_idx]
            
            if mode == 'add':
                chars.insert(char_idx, find_char)
            elif mode == 'delete':
                chars.pop(char_idx)
            else:
                raise Exception(f'Wrong Mode for editing vowel character: {mode}')
            break
    # error num, return original word
    if find_char is None:
        return None
    return ''.join(chars)


"""
Sentiment Lexicon
============================================
"""
def get_sentiment_lexicon(n=None, simple_word=None, pos=['JJ']):
    
    tagged_neg = nltk.pos_tag(list(opinion_lexicon.negative()))
    tagged_pos = nltk.pos_tag(list(opinion_lexicon.positive()))
    pos_lexicon = [] 
    neg_lexicon = []
    for word, tag in tagged_pos:
        if tag in pos: # JJ -> adjective
            pos_lexicon.append(word)

    for word, tag in tagged_neg:
        if tag in pos:
            neg_lexicon.append(word)
    
    def is_simple_word(word):
        
        if len(tokenizer1.tokenize(word)) > 1:
            return False
        if len(tokenizer2.tokenize(word)) > 1:
            return False
        if len(tokenizer3.tokenize(word)) > 1:
            return False
        return True

    if simple_word is not None:
        from transformers import AutoTokenizer
        tokenizer1 = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer2 = AutoTokenizer.from_pretrained('roberta-base')
        tokenizer3 = AutoTokenizer.from_pretrained('albert-base-v2')
        if simple_word: # True
            pos_lexicon = [w for w in pos_lexicon if is_simple_word(w) ]
            neg_lexicon = [w for w in neg_lexicon if is_simple_word(w) ]
        else: # False
            pos_lexicon = [w for w in pos_lexicon if not is_simple_word(w) ]
            neg_lexicon = [w for w in neg_lexicon if not is_simple_word(w) ]

    if n is not None:
        random.seed(20)
        pos_lexicon, neg_lexicon = random.sample(pos_lexicon, n), random.sample(neg_lexicon, n)
        
    return pos_lexicon, neg_lexicon