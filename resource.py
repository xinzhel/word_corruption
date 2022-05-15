import os
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass
from pydantic import NoneStr
from torch.nn import Module, CosineSimilarity
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from typing import Union, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
import nltk
from nltk.corpus import opinion_lexicon
import random
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.common.util import lazy_groups_of
from torch.nn import CosineSimilarity

"""
IO utils
============================================
"""
def find_load_folder(cur_dir):
    # successfully exit
    if os.path.exists(os.path.join(cur_dir, 'data')):
        return os.path.join(cur_dir, 'data')
    # fail exit
    par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
    if par_dir == cur_dir: # root dir
        return None
    
    # recursive call
    return find_load_folder(par_dir)

"""
Resource Abstract Class
============================================
"""
class LoadedResource(ABC):
    """ An abstract class for loading resource (e.g., datasets, Pytorch models)
    """
    def __init__(self):
        self.loaded_resource = dict()
        self.load_folder = find_load_folder(os.getcwd())

    def __getitem__(self, key, reload=False):
        if key not in self.loaded_resource.keys() or reload:
            self.loaded_resource[key] = self._load(key)

        return self.loaded_resource[key]

    def __setitem__(self, key, value):
        self.loaded_resource[key] = value

    @abstractmethod
    def _load(self, name):
        """ Returns loaded resource
        """
        raise NotImplementedError

"""
LoadedDatasets Class
============================================
"""
class LoadedDatasets(LoadedResource):
    shuffle: bool = False
    def _load(self, dataset_name):
        if dataset_name == "dbpedia14":
            assert self.load_folder is not None
            dataset = load_dataset("csv", column_names=["label", "title", "sentence"],
                                    data_files={"train": os.path.join(self.load_folder, "dbpedia_csv/train.csv"),
                                                "validation": os.path.join(self.load_folder, "dbpedia_csv/test/test.csv")})
            dataset = dataset.map(self.target_offset, batched=True)
            num_labels = 14
        elif dataset_name == "ag_news":
            dataset = load_dataset("ag_news")
            num_labels = 4
        # binary sentiment classification
        elif dataset_name == "imdb":
            dataset = load_dataset("imdb", ignore_verifications=True)
            num_labels = 2
        elif dataset_name == "yelp_long":
            dataset = load_dataset("yelp_polarity")
            num_labels = 2
        elif dataset_name == "yelp":
            dataset_dict = dict()
            for split in ['train', 'val', 'test']:
                
                with open(os.path.join(self.load_folder, f"yelp/{split}/data.txt")) as file:
                    texts = [line.rstrip() for line in file]
                with open(os.path.join(self.load_folder, f"yelp/{split}/labels.txt")) as file:
                    labels = [int(line.rstrip()) for line in file]
                dataset_dict[split] = Dataset.from_dict({'text': texts, 'label':labels})
            dataset = DatasetDict(dataset_dict)
            num_labels = 2
        elif dataset_name == "sst2":
            assert self.load_folder is not None
            dataset = load_dataset("csv", column_names=["text", "label"],
                                    data_files={"train": os.path.join(self.load_folder, "sst/train.csv"),
                                        "dev": os.path.join(self.load_folder , "sst/dev.csv"),
                                        "test": os.path.join(self.load_folder, "sst/test.csv"),})
            num_labels = 2
        elif dataset_name == "mnli":
            dataset = load_dataset("glue", "mnli")
            num_labels = 3
        elif dataset_name == "mrpc":
            dataset = load_dataset('glue', 'mrpc')
            num_labels = 2
        else:
            raise Exception("Cannot find the dataset.")
        if self.shuffle:
            dataset = dataset.shuffle(seed=0)
        
        return dataset, num_labels

    # offset target by 1 if labels start from 1
    @staticmethod
    def target_offset(examples):
        examples["label"] = list(map(lambda x: x - 1, examples["label"]))
        return examples

datasets = LoadedDatasets()

"""
Available models in huggingface hub
============================================
"""
# transformers
# TODO: for consistency, use the name composed of `{PLM_name}-{dataset_name}`
hf_lm_names = {

    'bert-base-uncased' : 'bert-base-uncased',
    'roberta-base': 'roberta-base',
    'albert-base-v2': 'albert-base-v2',
    'bert' : 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert ': 'albert-base-v2'
}
hf_model_names =  {
    # SQUAD
    'bert-base-uncased-squad2': 'deepset/bert-base-uncased-squad2',
    'roberta-base-squad2': 'deepset/roberta-base-squad2',
    'albert-base-v2-squad2': 'twmkn9/albert-base-v2-squad2',

    # for SST-2
    'bert-base-uncased-SST-2': 'textattack/bert-base-uncased-SST-2',
    'roberta-base-SST-2': 'textattack/roberta-base-SST-2',
    'albert-base-v2-SST-2':'textattack/albert-base-v2-SST-2',
    # consistent with dataset_name
    'bert-base-uncased-sst2': 'textattack/bert-base-uncased-SST-2',
    'bert-sst2': 'textattack/bert-base-uncased-SST-2',
    'roberta-base-sst2': 'textattack/roberta-base-SST-2',
    'roberta-sst2': 'textattack/roberta-base-SST-2',
    'albert-base-v2-sst2':'textattack/albert-base-v2-SST-2',
    'albert-sst2':'textattack/albert-base-v2-SST-2',
    # 'textattack/distilbert-base-uncased-SST-2',
    # 'textattack/distilbert-base-cased-SST-2',
    # 'textattack/xlnet-base-cased-SST-2',
    # 'textattack/xlnet-large-cased-SST-2',
    # 'textattack/facebook-bart-large-SST-2',

    # for yelp
    'bert-base-uncased-yelp': 'textattack/bert-base-uncased-yelp-polarity',
    'roberta-base-yelp': 'VictorSanh/roberta-base-finetuned-yelp-polarity',
    'albert-base-v2-yelp':'textattack/albert-base-v2-yelp-polarity',
    'bert-yelp': 'textattack/bert-base-uncased-yelp-polarity',
    'roberta-yelp': 'VictorSanh/roberta-base-finetuned-yelp-polarity',
    'albert-yelp':'textattack/albert-base-v2-yelp-polarity',

    # for ag-news
    'bert-base-uncased-ag-news': 'textattack/bert-base-uncased-ag-news',
    'roberta-base-ag-news': 'textattack/roberta-base-ag-news', 
    'albert-base-v2-ag-news': 'textattack/albert-base-v2-ag-news',
    # consistent with dataset_name
    'bert-base-uncased-ag_news': 'textattack/bert-base-uncased-ag-news',
    'roberta-base-ag_news': 'textattack/roberta-base-ag-news', 
    'albert-ag_news': 'textattack/albert-base-v2-ag-news',
    'bert-ag_news': 'textattack/bert-base-uncased-ag-news',
    'roberta-ag_news': 'textattack/roberta-base-ag-news', 
    'albert-ag_news': 'textattack/albert-base-v2-ag-news',
    # 'textattack/distilbert-base-uncased-ag-news',

    # for MRPC
    'bert-base-uncased-MRPC': 'textattack/bert-base-uncased-MRPC',
    'roberta-base-MRPC': 'textattack/roberta-base-MRPC',
    'albert-base-v2-MRPC': 'textattack/albert-base-v2-MRPC',
    # consistent with dataset_name
    'bert-base-uncased-mrpc': 'textattack/bert-base-uncased-MRPC',
    'roberta-base-mrpc': 'textattack/roberta-base-MRPC',
    'albert-base-v2-mrpc': 'textattack/albert-base-v2-MRPC',

    # for QQP
    'bert-base-uncased-QQP': 'textattack/bert-base-uncased-QQP',
    # 'textattack/distilbert-base-uncased-QQP',
    # 'textattack/distilbert-base-cased-QQP',
    'albert-base-v2-QQP': 'textattack/albert-base-v2-QQP',
    'roberta-base-QQP': 'howey/roberta-large-qqp',
    # 'textattack/xlnet-large-cased-QQP',
    # 'textattack/xlnet-base-cased-QQP',

    # for snil
    # 'textattack/bert-base-uncased-snli',
    # 'textattack/distilbert-base-cased-snli',
    # 'textattack/albert-base-v2-snli',

    # for WNLI
    # 'textattack/bert-base-uncased-WNLI',
    # 'textattack/roberta-base-WNLI',
    # 'textattack/albert-base-v2-WNLI',

    # for MNLI
    # 'textattack/bert-base-uncased-MNLI',
    # 'textattack/distilbert-base-uncased-MNLI',
    # 'textattack/roberta-base-MNLI',
    # 'textattack/xlnet-base-cased-MNLI',
    # 'textattack/facebook-bart-large-MNLI',
    # 'facebook/bart-large-mnli',
}
    

"""
LoadedHfModels Class
============================================
"""
class LoadedHfModels(LoadedResource):

    def _load(self, name): 
        if name in hf_model_names.keys():
            model = AutoModelForSequenceClassification.from_pretrained(hf_model_names[name])
        if name in hf_lm_names.keys():
            model = AutoModelForMaskedLM.from_pretrained(hf_lm_names[name])
        return model

hf_models = LoadedHfModels()

"""
LoadedHfTokenizers Class
============================================
"""
class LoadedHfTokenizers(LoadedResource):

    def _load(self, name):
        all_names = {**hf_model_names, **hf_lm_names}
        print('Valid name:', name)
        assert name in all_names
        valid_name = all_names[name]
            
        return AutoTokenizer.from_pretrained(valid_name)
       

hf_tokenizers = LoadedHfTokenizers()

"""
Model-Tokenizer Bundle Class
============================================
"""
@dataclass
class HfModelBundle:
    model_name: str
    model: Module
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    @property
    def embedding_layer(self):
        return self.model.get_input_embeddings()
    
    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def full_model_name(self):
        all_names = {**hf_lm_names, **hf_model_names}
        if self.model_name in all_names:
            return all_names[self.model_name]
        return None

    @property
    def all_special_ids(self):
        return self.tokenizer.all_special_ids

    def tokenize_from_words(self, words1: List[str], words2: List[str]=None):
        if getattr(self, "allennlp_tokenizer", None) is None:
            self.allennlp_tokenizer = PretrainedTransformerTokenizer(self.full_model_name)
        if words2 is None:
            wordpieces, offsets = self.allennlp_tokenizer.intra_word_tokenize(words1)
        else:
            wordpieces, offsets, offsets2 = self.allennlp_tokenizer.intra_word_tokenize_sentence_pair(words1, words2)
            offsets.extend(offsets2)
        wordpieces = {
            'token_str': [t.text for t in wordpieces],
            "input_ids": [t.text_id for t in wordpieces],
            "token_type_ids": [t.type_id for t in wordpieces],
            "attention_mask": [1] * len(wordpieces), 
        }
        
        return wordpieces, offsets
    
    def get_logit(self, words):
        model_output = self.get_model_output(words, y=None)
        if isinstance(model_output, SequenceClassifierOutput):
            return model_output.logits[0]
        else: 
            return None
            
    def get_model_output(self, words, y=None):
        wordpieces = self.tokenize_from_words(words)[0]
        model_input = {
            "input_ids": torch.LongTensor(wordpieces['input_ids']).unsqueeze(0),
            "token_type_ids": torch.LongTensor(wordpieces['token_type_ids']).unsqueeze(0),
            "attention_mask": torch.LongTensor(wordpieces['attention_mask']).unsqueeze(0), 
            
        }
        if y is not None:
            model_input["labels"] = torch.LongTensor([y]).unsqueeze(0)
        model_output = self.forward(model_input)
        return model_output
        

    def forward(self, model_input, return_last_hidden_states=False):
        # to correct device
        device = self.device
        model_input = {k: v.to(device) for k, v in model_input.items()}
        self.model.to(device)

        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input, output_hidden_states=True) 

        if return_last_hidden_states:
            # TODO: include other types of tasks
            assert isinstance(model_output, (SequenceClassifierOutput, MaskedLMOutput))

            # hidden_state[-1] taking the output of the last encoding layer from 
            # tuple for all layers' the hidden state
            last_hidden_state = model_output.hidden_states[-1]

            return last_hidden_state
            
        else:
            return model_output

    def get_sentence_embedding(self, examples: List[str], use_cls=False, normalize=False):
        embeddings = []
        for batch in lazy_groups_of(examples, 2): # assume the device is capable to deal with bsz 2
            model_input = self.tokenizer.batch_encode_plus(
                batch, 
                max_length=256, 
                truncation=True, 
                padding=True, 
                return_tensors='pt'
                )
            
            last_hidden_states = self.forward(model_input, return_last_hidden_states=True)
            if use_cls:  
                # [:, 0]  taking the hidden state corresponding
                # to the first token, i.e., [CLS].           
                embeddings.append(last_hidden_states[:, 0])
            else:
                all_tokens_embeddings = last_hidden_states[:, 1:]
                embeddings.append(all_tokens_embeddings.mean(dim=1))
        result = torch.cat(embeddings, dim=0) # dim: [num_sample, hidden_size]
        if normalize:
            return (result - result.mean(dim=1))/result.std(dim=1)
        else:
            return result
    
    def get_sentence_embedding_from_words(self, words, use_cls=False, normalize=False):
        wordpieces = self.tokenize_from_words(words)[0]
        model_input = {
            "input_ids": torch.LongTensor(wordpieces['input_ids']).unsqueeze(0),
            "token_type_ids": torch.LongTensor(wordpieces['token_type_ids']).unsqueeze(0),
            "attention_mask": torch.LongTensor(wordpieces['attention_mask']).unsqueeze(0), 
        }
        last_hidden_states = self.forward(model_input, return_last_hidden_states=True)
        if use_cls:
            result = last_hidden_states[:, 0] # dim: [num_sample=1, hidden_size]
        else:
            all_tokens_embeddings = last_hidden_states[:, 1:] # dim: [num_sample=1, seq_len, hidden_size]
            result =  all_tokens_embeddings.mean(dim=1) # dim: [num_sample=1, hidden_size]
        if normalize:
            return (result - result.mean(dim=1))/result.std(dim=1)
        else:
            return result

    def get_cos_sim(self, words1, words2):
        
        cos_sim = CosineSimilarity(dim=1)
        sent_emb1 = self.get_sentence_embedding_from_words(words1)
        sent_emb2 = self.get_sentence_embedding_from_words(words2)
        return cos_sim(sent_emb1, sent_emb2)


class LoadedHfModelBundle(LoadedResource):

    def _load(self, name):
        tokenizer = hf_tokenizers[name]
        model = hf_models[name]
        return HfModelBundle(name, model, tokenizer)

hf_model_bundles = LoadedHfModelBundle()

@dataclass(frozen=True)
class AllenNLPModelBundle:
    pass

"""
Sentiment Lexicon
============================================
"""
def get_sentiment_lexicon(n=50, simple_word=False):
    tagged_neg = nltk.pos_tag(list(opinion_lexicon.negative()))
    tagged_pos = nltk.pos_tag(list(opinion_lexicon.positive()))
    pos_lexicon = [] 
    neg_lexicon = []
    for word, tag in tagged_pos:
        if tag == 'JJ': # adjective
            pos_lexicon.append(word)

    for word, tag in tagged_neg:
        if tag == 'JJ':
            neg_lexicon.append(word)
    def is_simple_word(word):
        # hf_tokenizers['bert-base-uncased'].tokenize(word)
        return True

    if simple_word:
        pos_lexicon = [w for w in pos_lexicon if is_simple_word(w) ]
        neg_lexicon = [w for w in neg_lexicon if is_simple_word(w) ]

    random.seed(20)
    pos_lexicon50 = random.sample(pos_lexicon, n)
    random.seed(20)
    neg_lexicon50 = random.sample(neg_lexicon, n)
    return pos_lexicon50, neg_lexicon50

