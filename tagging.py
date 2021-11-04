from typing import Dict

import nltk
from nltk.corpus import brown

from collections import defaultdict

brown_corpus = brown.tagged_words()
tag_freq_data = nltk.ConditionalFreqDist((tag, word.lower()) for (word, tag) in brown_corpus)
tag_bigram_data = defaultdict(lambda: defaultdict(int))
for ((w1, t1), (w2, t2)) in nltk.bigrams(brown_corpus):
    tag_bigram_data[t1][t2] += 1
class HMM:
    """This class represents a Hidden Markhov Model for tagging
    
    This class is a Hidden Markhov Model to disambiguate the pos tags
    of new sentences
    
    Attributes
    ----------
    Methods
    ---------
    """
    def __init__(self, tag_word_count_dict: Dict[str, Dict[str, int]], tag_bigram_count_dict: Dict[str, Dict[str, int]]):
        """Initialize HMM object"""
        self.tag_word_count_dict = tag_word_count_dict
        self.tag_bigram_count_dict = tag_bigram_count_dict

    def get_word_tag_cond_prob(self, word: str, tag: str) -> float:
        """Return the conditional probability of a word given a tag"""

        tag_word_count = self.tag_word_count_dict[tag][word]
        tag_count = sum(self.tag_word_count_dict[tag].values())

        return tag_word_count/tag_count

    def get_tag_bigram_cond_prob(self, tag0: str, tag1: str) -> float:
        """Return the conditional probability of a tag given the previous tag"""

        tag_bigram_count = self.tag_bigram_count_dict[tag0][tag1]
        tag_count = sum(self.tag_bigram_count_dict[tag0].values())

        return tag_bigram_count/tag_count
    

if __name__ == '__main__':
    hmm = HMM(tag_freq_data, tag_bigram_data)
    print(hmm.get_tag_bigram_cond_prob("NN", "VB"))