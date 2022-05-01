import resource
import pickle
from torch.nn import CosineSimilarity
import pandas as pd
import argparse
from tqdm import tqdm

class Evaluator:
    def __init__(self, model_bundle, data_embeddings, debug=False):
        self.model_bundle = model_bundle
        self.data_embeddings = data_embeddings

        self.all_examples = []
        self.cos_sim = CosineSimilarity(dim=1)
        self.debug = debug # print information

    def add_to_groups(self, index, noisy_example, debug=False):
        if debug:
            self.debug = True
        noisy_example = self.calculate_wcs(noisy_example,)
        noiser_names = [k[2:] for k in noisy_example.keys() if k.startswith('x_')]
        y = noisy_example['y']
        logit_clean = self.get_logit( noisy_example['x'], y )
        if logit_clean is not None:
            logit_clean = logit_clean.detach().cpu()

        # loop over all noisy version of this sample
        result = []
        for noiser_str in noiser_names:
            if noisy_example['x_'+noiser_str] is None:
                continue
            # sent sim
            sim = self.get_cosine_sim(index, noisy_example['x_'+noiser_str],  noisy_example['x'])

            # pred
            logit = self.get_logit( noisy_example['x_'+noiser_str], y )
            if logit is not None:
                logit = logit.detach().cpu()

            result.append({'index':index, 
                 'label': y, 
                 'noise_type': noiser_str, 
                 'countM': noisy_example[f'countM_{noiser_str}'] , 
                 'countM_round': round(noisy_example[f'countM_{noiser_str}'] ), 
                 'countO': noisy_example[f'countO_{noiser_str}'],
                 'countO_round': round(noisy_example[f'countO_{noiser_str}'] ),
                 'wcr1': noisy_example[f'wcr1_{noiser_str}'],
                 'wcr2': noisy_example[f'wcr2_{noiser_str}'],
                 'cos_sim': sim.detach().item(), 
                 'logit': logit,
                 'logit_clean': logit_clean,
                 'O_sets': noisy_example[f'O_sets_{noiser_str}'],
                 'M_sets': noisy_example[f'M_sets_{noiser_str}'],
                 'N_sets': noisy_example[f'N_sets_{noiser_str}']
                 })
        
        self.all_examples.extend(
            result
        )
        return result
            
        
    def calculate_wcs(self, example):  # a dictionary [a list of words]
        """ calculate word corruption scores
        """
        
        tokens, offsets = self.model_bundle.tokenize_from_words( example['x'] )
        tokens = tokens['token_str']
        noiser_names = [k[2:] for k in example.keys() if k.startswith('x_')]
        for noiser_str in noiser_names:
            if example[f'x_{noiser_str}'] is None:
                continue
            # tokenize and calculate increased subwords after transformation
            tokens_noisy, offsets_noisy = self.model_bundle.tokenize_from_words(example[f'x_{noiser_str}'])  
            tokens_noisy = tokens_noisy['token_str']
            
            O_sets = []
            M_sets = []
            N_sets = []
            countM = 0
            countO = 0
            wcr1 = 0
            wcr2 = 0
            num_context_words = 0
            for offset, offset_noisy in zip(offsets, offsets_noisy):
                # word corruption score for a single word
                token_set = tokens[offset[0]:offset[1] + 1]
                token_set_noisy = tokens_noisy[offset_noisy[0]:offset_noisy[1] + 1]
                if self.debug:
                    print(noiser_str)
                    print(token_set)
                    print(token_set_noisy)
                
                # disagreement set
                extra_token_set = (set(token_set) | set(token_set_noisy)).difference(set(token_set))
                M_sets.append(extra_token_set )
                countM += len(extra_token_set) # count M
                wcr1 += len(extra_token_set) / len(set(token_set_noisy))

                # discard set
                discard_token_set = (set(token_set) | set(token_set_noisy)).difference(set(token_set_noisy))
                O_sets.append(discard_token_set)
                countO += len(discard_token_set)
                wcr2 += len(discard_token_set) / len(set(token_set))

                # agreement set
                intersection_set = set(token_set_noisy) & set(token_set)
                N_sets.append(intersection_set)
                
                num_context_words += 1 
            try:
                # calculate the number of subwords, which equals to character editings
                # num_context_words = jiwer.compute_measures(example['x'], example[f'x_{noiser_str}'])['wer']
                # which indicates that the number of additional subwords each edit brings on average.
                example[f'countM_{noiser_str}'] = countM / num_context_words
                example[f'countO_{noiser_str}'] = countO / num_context_words
                example[f'wcr1_{noiser_str}'] = wcr1 / num_context_words
                example[f'wcr2_{noiser_str}'] = wcr2 / num_context_words
                example[f'O_sets_{noiser_str}'] = O_sets
                example[f'M_sets_{noiser_str}'] = M_sets
                example[f'N_sets_{noiser_str}'] = N_sets

                
            except ZeroDivisionError:
                if not noiser_str == 'accent' and self.debug:
                    print(noiser_str, ' has no change.')
                    print(example['x'])
                    print(example[f'x_{noiser_str}'])
                example[f'x_{noiser_str}'] = None

        return example

    def get_cosine_sim(self, index, words, ref_words=None):
        if self.data_embeddings is not None:
            sent_emb1 = self.data_embeddings[index].unsqueeze(0)
        else:
            sent_emb1 = self.model_bundle.get_sentence_embedding_from_words(ref_words)
        sent_emb2 = self.model_bundle.get_sentence_embedding_from_words(words)
        assert sent_emb1.shape == sent_emb2.shape
        return self.cos_sim(sent_emb1, sent_emb2)

    def get_logit(self, words, y ):
        # get logits
        model_output = self.model_bundle.get_logit(words, y)
        return model_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating...")
    parser.add_argument("--dataset_name", default="sst2", type=str,)
    parser.add_argument("--model_name", default="bert-base-uncased-SST-2", type=str,)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name=args.model_name
    debug=False

    with open(f'outputs/{dataset_name}_noisy.pickle', 'rb') as file:
        noisy_data = pickle.load(file)

    model_bundle = resource.hf_model_bundles[model_name]
    model_bundle.model = model_bundle.model.to("cuda:0")

    if dataset_name == 'sentiment-lexicon':
        data_embeddings = None
    else:
        ref_text = resource.datasets[dataset_name][0]['test']['text']
        data_embeddings = model_bundle.get_sentence_embedding(ref_text)

    evaluator = Evaluator(model_bundle, data_embeddings, debug=debug)

    print('Example: \n')
    print(noisy_data[0])
    print_example = True
    for index, noisy_sample in tqdm(noisy_data):
        if dataset_name == 'sentiment-lexicon':
            noisy_sample['y'] = None
        result = evaluator.add_to_groups(index, noisy_sample)
        if print_example:
            print(result)
            print_example = False

    print('Saving result into: ', f'outputs/{dataset_name}_{model_name}.pickle')
    with open(f'outputs/{dataset_name}_{model_name}.pickle', 'wb') as file:
        pickle.dump(evaluator.all_examples , file)
