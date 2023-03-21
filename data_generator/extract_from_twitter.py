

import json
import re 
from utils import get_sentiment_lexicon

with open("data/twitter_sentiment140.csv", 'r') as f:
    text_to_search = f.readlines()
text_to_search = [text_to_search[i][:-3] for i in range(len(text_to_search))]
text_to_search_str = ''.join(text_to_search)
def search_reduplications(text_to_search_str, words):
    result = {}
    for clean_word in words:
        pattern = '\\b'
        for c in clean_word:
            pattern += c+'+'
        # print(r'{}'.format(pattern))
        try:
            pattern = re.compile(r'{}'.format(pattern))
        except re.error:
            print(pattern)
            continue
        matches = pattern.finditer(text_to_search_str)
        for match in matches:
            if match.group(0)!=clean_word:
                try:
                    result[clean_word].add(match.group(0))
                except KeyError:
                    result[clean_word] = set([match.group(0)])
    return result

def search_abbreviation(text_to_search_str, words):
    result = {}
    
    for clean_word in words:
        consonants = [clean_word[i] for i in range(1, len(clean_word)) if clean_word[i] not in ['a', 'e', 'i', 'o', 'u'] and clean_word[i] != clean_word[i-1]]
        if len(consonants) < 3:
            continue
        pattern = '\\b'+clean_word[0] # keep start char

        for c in consonants:
            pattern += c + '?' # 0 or 1 `c`
        # print(r'{}'.format(pattern))
        try:
            pattern = re.compile(r'{}'.format(pattern))
        except re.error:
            print(pattern)
            continue
        matches = pattern.finditer(text_to_search_str)
        for match in matches:
            if match.group(0)!=clean_word and len(match.group(0))>3:
                try:
                    result[clean_word].add(match.group(0))
                except KeyError:
                    result[clean_word] = set([match.group(0)])
    return result

# intact-reduplication, intact-abbreviation
# pos_lexicon, neg_lexicon = get_sentiment_lexicon(simple_word=True)

# additive-reduplication
# pos_lexicon, neg_lexicon = get_sentiment_lexicon(simple_word=False)

# all
pos_lexicon, neg_lexicon = get_sentiment_lexicon(simple_word=None)
print(len(pos_lexicon), len(neg_lexicon)) # 710 1441

# pos_reduplications = search_reduplications(text_to_search_str, pos_lexicon)
# print('pos_reduplications:', len(pos_reduplications), ) # 179
# new_results = {clean: list(noisy)  for clean, noisy in pos_reduplications.items()}
# json.dump(new_results, open('data/pos-reduplications.json', 'w'), sort_keys=True, indent=4)
# 
pos_abbreviations = search_abbreviation(text_to_search_str, pos_lexicon)
print('pos_abbreviations:', len(pos_abbreviations)) # 126
new_results = {clean: list(noisy)  for clean, noisy in pos_abbreviations.items()}
json.dump(new_results, open('data/pos-abbreviations.json', 'w'), sort_keys=True, indent=4)

# neg_reduplications = search_reduplications(text_to_search_str, neg_lexicon)
# print('neg_reduplications:', len(neg_reduplications), ) # 286
# new_results = {clean: list(noisy)  for clean, noisy in neg_reduplications.items()}
# json.dump(new_results, open('neg_reduplications.json', 'w'), sort_keys=True, indent=4)

neg_abbreviations = search_abbreviation(text_to_search_str, neg_lexicon)
print('neg_abbreviations:', len(neg_abbreviations)) # 271
new_results = {clean: list(noisy)  for clean, noisy in neg_abbreviations.items()}
json.dump(new_results, open('data/neg-abbreviations.json', 'w'), sort_keys=True, indent=4)




