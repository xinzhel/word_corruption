from matplotlib import use
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import TextBuggerLi2018
from textattack.attacker import Attacker
from textattack.attack_args import AttackArgs
from textattack.models.wrappers import HuggingFaceModelWrapper
from attack_utils import GreedyWordSwapWIRWithCorruption, UntargetedClassificationQueryOneByOne
import resource

def build(model_bundle, corruption_inf=None):
    from textattack import Attack
    from textattack.constraints.pre_transformation import (
        RepeatModification,
        StopwordModification,
    )
    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
    from textattack.goal_functions import UntargetedClassification
    from textattack.search_methods import GreedyWordSwapWIR
    from textattack.transformations import (
        CompositeTransformation,
        WordSwapEmbedding,
        WordSwapHomoglyphSwap,
        WordSwapNeighboringCharacterSwap,
        WordSwapRandomCharacterDeletion,
        WordSwapRandomCharacterInsertion,
    )
    #
    #  we propose five bug generation methods for TEXTBUGGER:
    #
    transformation = CompositeTransformation(
        [
            # (1) Insert: Insert a space into the word.
            # Generally, words are segmented by spaces in English. Therefore,
            # we can deceive classifiers by inserting spaces into words.
            WordSwapRandomCharacterInsertion(
                # random_one=True,
                # letters_to_insert=" ",
                skip_first_char=True,
                skip_last_char=True,
            ),
            # (2) Delete: Delete a random character of the word except for the first
            # and the last character.
            WordSwapRandomCharacterDeletion(
                # random_one=True, 
                skip_first_char=True, skip_last_char=True
            ),
            # (3) Swap: Swap random two adjacent letters in the word but do not
            # alter the first or last letter. This is a common occurrence when
            # typing quickly and is easy to implement.
            WordSwapNeighboringCharacterSwap(
                # random_one=True, 
                skip_first_char=True, skip_last_char=True
            ),
            # (4) Substitute-C (Sub-C): Replace characters with visually similar
            # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
            # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
            WordSwapHomoglyphSwap(),
            # (5) Substitute-W
            # (Sub-W): Replace a word with its topk nearest neighbors in a
            # context-aware word vector space. Specifically, we use the pre-trained
            # GloVe model [30] provided by Stanford for word embedding and set
            # topk = 5 in the experiment.
            # WordSwapEmbedding(max_candidates=5),
        ]
    )


    constraints = [RepeatModification(), StopwordModification()]
    # In our experiment, we first use the Universal Sentence
    # Encoder [7], a model trained on a number of natural language
    # prediction tasks that require modeling the meaning of word
    # sequences, to encode sentences into high dimensional vectors.
    # Then, we use the cosine similarity to measure the semantic
    # similarity between original texts and adversarial texts.
    # ... "Furthermore, the semantic similarity threshold \eps is set
    # as 0.8 to guarantee a good trade-off between quality and
    # strength of the generated adversarial text."
    constraints.append(UniversalSentenceEncoder(threshold=0.8))

    #
    # Goal is untargeted classification
    #
    model_wrapper = HuggingFaceModelWrapper( model_bundle.model, model_bundle.tokenizer, )
    goal_function = UntargetedClassification( model_wrapper )
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    # search_method = GreedyWordSwapWIR(wir_method="delete")
    search_method = GreedyWordSwapWIRWithCorruption(model_bundle, wir_method="delete", use_corrupt="worst")

    return Attack(goal_function, constraints, transformation, search_method)


model_bundle = resource.hf_model_bundles['roberta-base-QQP']

attack = build(model_bundle,)

success_fail_dict = {}
dataset = resource.datasets['qqp'][0]['validation']
dataset_wrapper = HuggingFaceDataset(dataset)
attack_args = AttackArgs(num_examples=100)
attacker = Attacker(attack, dataset_wrapper, attack_args=attack_args)
attacker.attack_dataset()