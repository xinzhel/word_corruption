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
Interpret
============================================
"""
class IntegratedGradient:
    """ Code is written by refering to AllenNLP library
    """
    def __init__(self, model, embedding_layer=None) -> None:
        self.model = model
        if embedding_layer:
            self.embedding_layer = embedding_layer
        else:
            self.embedding_layer = nn_util.find_embedding_layer(self.model)

    def _register_hooks(self, embedding_layer, alpha: int, embeddings_list: List):

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach())

            # Scale the embedding by alpha
            output.mul_(alpha)

        def forward_hook_for_hf_plm(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.last_hidden_state.squeeze(0).clone().detach())

            # Scale the embedding by alpha
            # output.last_hidden_state.mul_(alpha)
        
        if isinstance(embedding_layer, PreTrainedModel):
            return embedding_layer.register_forward_hook(forward_hook_for_hf_plm)
        else:
            return embedding_layer.register_forward_hook(forward_hook)


    def _integrate_gradients(self, model_input, token_offsets: List[torch.Tensor]=None) -> Dict[str, np.ndarray]:
        
        ig_grads = None

        # List of Embedding inputs
        embeddings_list: List[torch.Tensor] = []

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10

        
        # Exclude the endpoint because we do a left point integral approximation
        for alpha in np.linspace(0, 1.0, num=steps, endpoint=False):
            handles = []
            # Hook for modifying embedding value
            handle = self._register_hooks(self.embedding_layer, alpha, embeddings_list)

            try:
                grads, _ = get_grad(model_input, self.model, self.embedding_layer)
                grads = torch.from_numpy(grads)
            finally:
                handle.remove()

            # Running sum of gradients
            if ig_grads is None:
                ig_grads = grads
            else:
                ig_grads += grads

        # Average of each gradient term
        ig_grads /= steps

        # Gradients come back in the reverse order that they were sent into the network
        # embeddings_list.reverse()
        # token_offsets.reverse()
        # embeddings_list = self._aggregate_token_embeddings(embeddings_list, token_offsets)

        # Element-wise multiply average gradient by the input
        assert len(embeddings_list) == 1
        input_embedding = embeddings_list[0]
        ig_grads *= input_embedding

        return ig_grads

    def integrate_gradients(self, model_input):
        ig_grads = self._integrate_gradients(model_input)
        ig_grads = ig_grads.numpy() # shape: bsz, seq_len, emb_hidden_size
        # The [0] here is undo-ing the batching that happens in get_gradients.
        embedding_grad = np.sum(ig_grads[0], axis=1) # shape: seq_len,
        norm = np.linalg.norm(embedding_grad, ord=1) # shape: 1
        normalized_grad = [math.fabs(e) / norm for e in embedding_grad] # shape: seq_len
        return normalized_grad



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

class WIRSample(Sample):
    def __init__(
        self, 
        data, 
        model_bundle=None, 
        wir=None, 
        pct_modify=None, 
        n_modify=None,
        origin=None,
        sample_id=None):
        
        # load text and label
        self.sentence1 = None
        self.x = None
        if 'text' in data.keys():
            data['x'] = data['text']
        
        if 'sentence2' in data.keys():
            data['x'] = data['sentence2']
            self.sentence1 = data['sentence1']
            
        if 'label' in data.keys():
            data['y'] = int(data['label'])
        
        super().__init__(data, origin=origin, sample_id=sample_id)
        

        # label, words, subwords
        self.labels = data['y']
        self.words = self.get_words('x')
        
        # modification parameters; used in `get_mask`
        self.n_modify = n_modify
        self.pct_modify = pct_modify

        if model_bundle is not None:
            self.fix_wir = False
            # discompose model bundle
            self.model_name = model_bundle.model_name
            self.model = model_bundle.model
            self.tokenizer = model_bundle.tokenizer
            self.embedding_layer = model_bundle.embedding_layer
            self.device = model_bundle.device
            self.wordpieces, self.offsets = model_bundle.tokenize_from_words(self.words, self.sentence1)
        else: 
            assert wir is not None # if not providing model to calculate wir, we should have wir as the parameter
            self._wir = wir
            self.fix_wir = True
    
    def __repr__(self):
        return 'WIRSample'

    def check_data(self, data):
        assert 'x' in data and isinstance(data['x'], str)

    def load(self, data):
        r"""
        Convert data dict which contains essential information to SASample.

        :param dict data: contains 'x' key at least.
        :return:

        """
        self.x = TextField(data['x'])

    def dump(self):
        return {'x': self.x.text, 'sample_id': self.sample_id}

    def is_legal(self):
        r"""
        Validate whether the sample is legal

        :return: bool

        """
        return True
        
    @property
    def wir(self):
        if self.fix_wir:
            return self._wir

        # token grads
        token_grads = self.token_grads

        # word grad
        word_grad_norm = []
        word_grads = []
        for offset in self.offsets:
            word_grad = token_grads[offset[0]: offset[1]+1, :]
            word_grad_norm.append(np.mean(np.linalg.norm(word_grad, axis=1)))
            word_grads.append(word_grad)
        assert len(self.words) == len(word_grad_norm)

        # wir: a list of word indices in descending order (important first)
        wir = list(np.argsort(np.array(word_grad_norm)))[::-1] 

        return wir

    @property
    def token_grads(self):
        if self.fix_wir:
            raise Exception('Cannot use token_grads when fix_wir is True')
        # token grad
        device = self.device
        wordpieces = self.wordpieces
        model_input = {
            "input_ids": torch.LongTensor(wordpieces['input_ids']).unsqueeze(0).to(device),
            "token_type_ids": torch.LongTensor(wordpieces['token_type_ids']).unsqueeze(0).to(device),
            "attention_mask": torch.LongTensor(wordpieces['attention_mask']).unsqueeze(0).to(device), 
            "labels": torch.LongTensor([self.labels]).unsqueeze(0).to(device)
        }
        
        token_grads, total_loss = get_grad(model_input, self.model, self.embedding_layer)
        return np.squeeze(token_grads)
    
    def get_mask(self, field):
        
        mask = super().get_mask(field)
        wir_mask = [MODIFIED_MASK for _ in mask]
        wir = self.wir
        # locate word_idx for transformation; 
        # this is necessary for sentence pair, where we len(mask) is for the 'x'-keyed sentence
        valid_wir = [idx for idx in wir if idx <len(mask) ]

        # find the number of words to modify
        if self.n_modify is not None:
            # print('n_modify', self.n_modify)
            n_modify = self.n_modify
        elif self.pct_modify is not None:
            # print('pct_modify', self.pct_modify)
            n_modify = int(self.pct_modify * len(mask))
        else:
            # print('mask length:', len(mask))
            n_modify = len(valid_wir)
        
        if n_modify > len(valid_wir): # not use wir
            return mask
        else: 
            for i in range(n_modify):
                word_idx = valid_wir[i]
                wir_mask[word_idx] = 0

            return wir_mask

    def loss_approximate(self, rank = 0):
        if self.fix_wir:
            raise Exception('Cannot use token_grads when fix_wir is True')
        if getattr(self, "sort_indices", None) is None:
            
            # get loss approximation scores
            if self.model_type == 'hf':
                all_special_ids = self.tokenizer.all_special_ids
                embedding_matrix = self.embedding_layer.word_embeddings.weight
                num_vocab = self.embedding_layer.word_embeddings.weight.size(0)
            else:
                raise NotImplementedError
            
            _, self.sort_indices = get_approximate_scores(self.token_grads , embedding_matrix, \
                                        all_special_ids=all_special_ids, sign=1)

        position_to_flip, what_to_modify = int(self.sort_indices[rank] // num_vocab) + 1, int(self.sort_indices[rank] % num_vocab)
        
        what_to_modify = self.tokenizer._convert_id_to_token(what_to_modify)
        token_to_modify = self.tokenizer._convert_id_to_token(self.tokens[position_to_flip])
        
        #self.model_input['input_ids'][0, position_to_flip] = what_to_modify
        #model_output = self.model(**self.model_input)
        #print(model_output['logits'], model_output['loss'])

        return what_to_modify, token_to_modify        

    def get_transformed_words(self, field='x'):
        mask = self.get_mask(field)
        words = self.get_words(field)
        transformed_words = []
        for i, m in enumerate(mask):
            if m == 0:
                transformed_words.append(words[i])
        return transformed_words

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
Search Scores
============================================
"""
def get_approximate_scores(grad, embedding_matrix, all_special_ids: List[int] =[], sign: int = -1):
    """ The objective is to minimize the first-order approximate of L_adv:
        L = L(orig_text) + [replace_token - orig_text[i]].dot(input_grad)
        ignore `orig_text` since it is constant and does not affect the result:
        minimize: replace_token.dot(input_grad)

        grad: (seq_len, embed_size); we assume all positions in the first dimension are 
            valid, i.e.,no special positions like [SEP] [PAD]
        embedding_matrix: (vocab_size, embed_size)
        all_special_ids: block invalid tokens in vocabulary (or embedding matrix)
        sign: -1 for minimization ; 1 for maximization
    """
    
    # (seq_len, vocab_size)
    first_order_dir = torch.einsum("ij,kj->ik", (torch.Tensor(grad), embedding_matrix.to('cpu'))) 

    # TODO: score in the replacement dimension for constraints or ...
    # use MLM to generate probability and then weight the above score

    # block invalid replacement
    first_order_dir[:, all_special_ids] = (-sign)*np.inf # special tokens are invalid for replacement

    scores = first_order_dir.flatten()
    if sign == -1: 
        descend = False # small score first: low -> high
    else:
        descend = True # large score first
    # get scores for the replacements for all position (seq_len*num_vocab)
    scores, indices = scores.sort(dim=-1, descending=descend)
    
    return scores, indices 

"""
Gradient
============================================
"""
def get_grad(
    dataset_tensor_dict, 
    model: torch.nn.Module, 
    layer: torch.nn.Module, 
    loss_fct: _Loss = None,
    batch_size: int = 16,
    return_tensors='np'):
    """ 
    # Parameters

    batch: A dictionary containing model input
    model: (1) the subclass of the `PreTrainedModel`  or 
           (2) Pytorch model with a method "get_input_embeddings" which return `nn.Embeddings`
    layer: the layer of `model` to get gradients, e.g., a embedding layer
    batch_size: avoid the case that `instances` may be too overloaded to perform forward/backward pass

    # Return

    return_grad: shape (batch size, sequence_length, embedding_size): gradients for all tokenized elements
        , including the special prefix/suffix and <SEP>.
    """
    
    cuda_device = next(model.parameters()).device

    gradients: List[Tensor] = []
    # register hook
    hooks: List[RemovableHandle] = _register_gradient_hooks(gradients, layer)

    # require grads for all model params 
    original_param_name_to_requires_grad_dict = {}
    for param_name, param in model.named_parameters():
        original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
        param.requires_grad = True

    # calculate grad for inference network
    orig_mode = model.training
    model.train(mode=False)
 
    # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
    with backends.cudnn.flags(enabled=False):
        
        dataset_tensor_dict = nn_util.move_to_device(dataset_tensor_dict, cuda_device)

        # update in batch 
        gradients_for_all = []
        total_loss = 0.0
        # Workaround to batch tensor_dict rather than instances 
        # in order to return a gradients list with the same sequence length
        # to be concatenated
        dataset_tensor_dict_iterator = loop_dict(dataset_tensor_dict, function_on_val=lambda val : iter(val))
        for batch in batch_dataset_tensor_dict_generator(dataset_tensor_dict_iterator,  batch_size=batch_size):
            batch.pop('metadata', None)
            outputs = model.forward(**batch)  

            if loss_fct is None:
                loss = outputs["loss"]
            else:
                raise NotImplementedError("Not support the customized loss function.")
                # labels = batch['labels'].view(-1)
                # loss = loss_fct(outputs['logits'], labels)
            # Zero gradients.
            # NOTE: this is actually more efficient than calling `model.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for p in model.parameters():
                p.grad = None
            gradients.clear()
            loss.backward()
            total_loss += loss.detach().cpu()
            gradients_for_all.append(gradients[0].detach().cpu().numpy())

    if len(gradients) != 1:
        import warnings
        warnings.warn(
            """get_grad: gradients for >1 inputs are acquired. 
            This should still work well for bidaf and 
            since the 1-st tensor is for passage.""")

    for hook in hooks:
        hook.remove()

    # restore the original requires_grad values of the parameters
    for param_name, param in model.named_parameters():
        param.requires_grad = original_param_name_to_requires_grad_dict[param_name]
    model.train(mode=orig_mode)
    
    return_grad = np.concatenate(gradients_for_all, axis=0)
    if return_tensors == 'pt':
        return_grad = torch.from_numpy(return_grad)
       
    return return_grad, total_loss

def loop_dict(input_dict, function_on_val):
    result_dict = {}
    for name, val in input_dict.items():
        if isinstance(val, dict): 
            result_dict[name] = loop_dict(val, function_on_val)
        elif isinstance(val, Iterable):
            result_dict[name] = function_on_val(val)
        else:
            raise Error("The Value has an Unknown Types.")
    return result_dict

def batch_dataset_tensor_dict_generator(dataset_tensor_dict_iterator, batch_size=16):
    ensure_iterable_value = False
    for _, value in dataset_tensor_dict_iterator.items():
        if isinstance(value, Iterable):
            ensure_iterable_value = True
    if not ensure_iterable_value:
        raise Exception('Have to ensure iterable value.')
    
    def func(iterator):
        lst = list(islice(iterator, batch_size))
        
        if len(lst) > 0 and isinstance(lst[0], torch.Tensor):
            return torch.stack(lst)
        else:
            return lst
    def runout(d):
        for _, value in d.items():
            if isinstance(value, torch.Tensor) or isinstance(value, list):
                if len(value) == 0:
                    return True
                else: 
                    return False
            elif isinstance(value, dict):
                return runout(value)
            else:
                raise Exception()

    while True:
        s = loop_dict(dataset_tensor_dict_iterator, function_on_val=func)
        if runout(s):
            break
        else:
            yield s

def _register_gradient_hooks(gradients, layer):

    def hook_layers(module, grad_in, grad_out):
        grads = grad_out[0]
        gradients.append(grads)

    hooks = []
    hooks.append(layer.register_full_backward_hook(hook_layers))
    return hooks

# class ModelFromHf(Model):
#     # Hf models -> AllenNLP models
#     def __init__(self, model_bundle) -> None:
#         
#         self.token_embedder = PretrainedTransformerEmbedder(model_name=check_valid_name(model_bundle.model_name)) # output last_hidden_state
#         assert torch.all(
#             model_bundle.model.bert.embeddings.word_embeddings.weight.eq(
#                 self.token_embedder.transformer_model.embeddings.word_embeddings.weight
#             )
#         )
#         self.transformer_model = model_bundle.model
# 
#     def forward(self):
#         sequence_output = self.token_embedder.forward() 
# 
#         # for albert
#         pooled_output = self.transformer_model.pooler_activation(self.transformer_model.pooler(sequence_output[:, 0]))
#         pooled_output = self.transformer_model.dropout(pooled_output)
#         logits = self.transformer_model.classifier(pooled_output)
# 
#         # for bert
#         pooled_output = self.transformer_model.pooler(sequence_output)
#         pooled_output = self.transformer_model.dropout(pooled_output) 
#         logits = self.transformer_model.classifier(pooled_output)
# 
#         # for roberta
#         # pooled_output = self.transformer_model.pooler(sequence_output)
#         logits = self.transformer_model.classifier(sequence_output)
# 
#         return logits
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