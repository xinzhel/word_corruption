# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import jiwer
# from datasets import load_dataset
# from textflint.generation.transformation.UT import Keyboard, Typos, SwapNamedEnt
# from textflint.input.component.sample import UTSample
# from attack_utils import get_grad
# import numpy as np
# import torch
# 
# 
# 
# # load data
# dataset = load_dataset("ag_news")
# test_data = dataset['test']
# test_data = test_data.rename_column('text', 'x')
# test_data = test_data.rename_column('label', 'y')
# 
# # load model
# tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-SST-2', use_fast=True)
# model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-SST-2')
# embedding_layer = model.get_input_embeddings()
# 
# # transform data
# keyboard = Keyboard()
# typos = Typos()
# ne = SwapNamedEnt()
# 
# 
# def transform(example, ):
#     sample = UTSample(example)
# 
#     return_dict = dict()
#     # transform text
#     return_dict['kb_transform'] = keyboard.transform(sample, field='x', n=1)[0].get_text('x')
#     return_dict['typos_transform'] = typos.transform(sample, field='x', n=1)[0].get_text('x')
#     return_dict['ne_transform'] = ne.transform(sample, field='x', n=1)[0].get_text('x')
#     
#     # calculate character editings
#     return_dict['kb_wer'] = jiwer.compute_measures(example['x'], return_dict['kb_transform'])['wer'] 
#     return_dict['typos_wer'] = jiwer.compute_measures(example['x'], return_dict['typos_transform'])['wer']
#     
#     # tokenize and calculate average lengths of tokens for orig/transformed test data
#     return_dict['x_bert_length'] = len(tokenizer.tokenize(example['x']))
#     return_dict['kb_bert_length'] = len(tokenizer.tokenize(example['kb_transform']))
#     return_dict['typos_bert_length'] = len(tokenizer.tokenize(example['typos_transform']))
#     
#     return return_dict
# 
# test_data = test_data.map(transform)


import transformers
# tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
# tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
ag_news = ["""Fears efféctiveness."""]
tokenizer.tokenize(ag_news[0])
# ['fears', 'effectiveness', '.']