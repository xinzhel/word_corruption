from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.constraints import Constraint
from textattack.search_methods import GreedyWordSwapWIR
from textattack.goal_functions import UntargetedClassification
from textattack.goal_function_results import GoalFunctionResultStatus
from multiset import Multiset
import random
import numpy as np

class UntargetedClassificationQueryOneByOne(UntargetedClassification):

    def get_results(self, attacked_text_list, check_skip=False):
        results = []
        for attaked_text in attacked_text_list:
            # query for result and add self.num_queries by 1
            res, search_over = super().get_results([attaked_text], check_skip=check_skip)
            if len(res):
                results.append(res[0])
            
                if res[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    search_over = True
                    break

        return results, search_over


class GreedyWordSwapWIRWithCorruption(GreedyWordSwapWIR):
    def __init__(self, model_bundle, wir_method="unk", use_corrupt='worst'):
        super().__init__(wir_method)
        self.model_bundle = model_bundle
        self.use_corrupt = use_corrupt

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)

        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            ### Corruption informed
            
            if self.use_corrupt == "worst":
                transformed_text_candidates = self.rank_by_corruption(attacked_text, transformed_text_candidates)
                # mid_idx = len(transformed_text_candidates)//2
                # assert mid_idx <len(transformed_text_candidates)
                transformed_text_candidates = [transformed_text_candidates[0]]
            elif self.use_corrupt == "rank":
                transformed_text_candidates = self.rank_by_corruption(attacked_text, transformed_text_candidates)
            else:
                transformed_text_candidates = [random.choice(transformed_text_candidates)]
            ###
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        return cur_result

    def rank_by_corruption(self, init_text, noisy_texts):
        # words_offsets = self.model_bundle.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(init_text._text_input['text'])
        # words, _ = zip(*words_offsets)
        words = init_text.words
        # scores = [self.calculate_corruption(words, t.words) for t in noisy_texts] 
        scores = []
        num_complete = 0
        num_intact = 0
        for t in noisy_texts:
            noisy_words = t.words
            score, t = self.calculate_corruption(words, noisy_words)
            if t == "complete":
                num_complete += 1
            if t == "intact":
                num_intact += 1
            scores.append(score)
        
        sort_idx = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1], reverse=True)]
        print('~~~~~~~~Total number of perturbed examples: ', len(scores))
        print('~~~~~~~~Percentage of complete: ', num_complete/len(scores))
        print('~~~~~~~~Percentage of intact: ', num_intact/len(scores))
        print('~~~~~~~~Statistics for corruption scores: ', np.min(scores), np.mean(scores), np.max(scores), np.std(scores))
    
        return [noisy_texts[idx] for idx in sort_idx]
    
    def calculate_corruption(self, words, noisy_words):
        words = [w.translate(w.maketrans('', '', '▁#Ġ')) for w in words] # remove substring indicators
        noisy_words = [w.translate(w.maketrans('', '', '▁#Ġ')) for w in noisy_words] # remove substring indicators
        if words == noisy_words:
            print('words == noisy words.')
            return 0
        # print('origin: ', words)
        # print('noisy: ', noisy_words)
        tokens, offsets = self.model_bundle.tokenize_from_words( words )
        tokens = tokens['token_str']
        tokens = [token.translate(token.maketrans('', '', '▁#Ġ')) for token in tokens] # remove substring indicators

        tokens_noisy, offsets_noisy = self.model_bundle.tokenize_from_words(noisy_words)  
        tokens_noisy = tokens_noisy['token_str']
        tokens_noisy = [token.translate(token.maketrans('', '', '▁#Ġ')) for token in tokens_noisy]
        
        score_A = 0
        for i, (offset, offset_noisy) in enumerate(zip(offsets, offsets_noisy)): # for each word 
            
            # word corruption score for a single word
            token_set = tokens[offset[0]:offset[1] + 1]
            token_set_noisy = tokens_noisy[offset_noisy[0]:offset_noisy[1] + 1]

            if token_set == token_set_noisy: # Just in case. TMBOK, NO tokenizer would tokenizes different words into same tokens
                continue

            overlap_set = Multiset(token_set) & Multiset(token_set_noisy) 
            missing_set = Multiset(token_set).difference(overlap_set)
            additive_set = Multiset(token_set_noisy).difference(overlap_set) #(set(token_set) | set(token_set_noisy)).difference(set(token_set))
            missing_rate = len(missing_set) / (len(missing_set) + len(overlap_set))
            
            score_A +=  len(additive_set) #+ 30**missing_rate  #len(additive_set) + ((len(additive_set)+1)**missing_rate)

        if score_A == 0:
            print("!!!!Wrong")
        
        if len(overlap_set) == 0:
            if len(missing_set) == 1:
                t = 'intact'
            else:
                t = 'complete'

        elif len(missing_set) > 0 and len(additive_set) > 0 and len(overlap_set) > 0:
            t = 'partial'
        elif len(missing_set) == 0:
            assert len(additive_set) > 0 and len(overlap_set) > 0
            t = 'additive'
        elif len(additive_set) == 0:
            assert len(missing_set) > 0 and len(overlap_set) > 0
            t = 'missing'
        else: 
            print(tokens)
            print(tokens_noisy)
            raise Exception('There is an unexpected corruption type.')
    
        return score_A, t

