import time
import math
from functools import reduce
from collections import Counter, defaultdict
from typing import Set, Any, List, Tuple, Dict, Union

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import brown

from language_models import FSA, HiddenMarkovModel

class NGram(FSA):
    """Create an object to represent an ngram language model
    
    This object will be able to generate symbols based on previous symbols.
    Given a corpus of sentences"""

    def __init__(self, ngram=2, smooth=False):

        self.ngram = ngram
        self.smooth = smooth

    def train(self, corpus: List[List[str]]):
        """Train the NGram model on valid corpus"""
        self.get_ngram_table(corpus)

    def get_ngram_table(self, corpus: List[List[str]]) -> pd.DataFrame:
        """Get the cooccurence of words in corpus"""

        for sent in corpus:
            for i in range(len(sent)):
                if i == 0:
                    pass

class HMMTagger(HiddenMarkovModel):
    """To tag input sentences using hidden markov models
    
    This tagger takes as an input a corpus of sentences that are tagged in order
    to tag future input sentences.
    
    Attributes
    ----------
    Methods
    ----------
    """

    def __init__(self, corpus: List[List[Tuple[str, str]]]):
        """Initialize the HMMTagger object"""

        self.corpus = corpus
        
        states = self.get_tag_set()
        start, transitions, final = self.get_transitions()
        emissions = self.get_emissions()

        super().__init__(states, start, final, transitions, emissions)
    
    def get_tag_set(self) -> Set[str]:
        """Return a set of all tags in corpus"""

        tag_set = set()
        for sent in self.corpus:
            for word, tag in sent:
                if tag not in tag_set:
                    tag_set.add(tag)

        return tag_set

    def get_transitions(self) -> pd.DataFrame:
        """Return a dataframe of probabilities of a state going to state"""

        tag_tag_counts = defaultdict(Counter)
        for sent in self.corpus:
            for i in range(len(sent)):
                if i == 0:
                    tag_tag_counts["<s>"][sent[i][1]] += 1
                else:
                    tag_tag_counts[sent[i-1][1]][sent[i][1]] += 1
                    if i == len(sent)-1:
                        tag_tag_counts[sent[i][1]]["</s>"] += 1

        transitions = pd.DataFrame(tag_tag_counts).fillna(0)
        transitions = transitions/transitions.sum(axis=0)

        start = transitions["<s>"]
        final = transitions.loc["</s>"]

        start.drop(["</s>"], inplace=True)
        final.drop(["<s>"], inplace=True)
        transitions.drop(["<s>"], axis=1, inplace=True)
        transitions.drop(["</s>"], axis=0, inplace=True)



        return start, transitions, final

    def get_emissions(self) -> pd.DataFrame:
        """Return a dataframe of probabilities of a word coming from a state"""

        tag_word_counts = defaultdict(Counter)
        for sent in self.corpus:
            for word, tag in sent:
                tag_word_counts[tag][word.lower()]+=1

        emissions = pd.DataFrame(tag_word_counts).fillna(0)
        emissions = emissions/emissions.sum(axis=0)

        return emissions

if __name__ == '__main__':
    start_time = time.time()
    brown_sents = brown.tagged_sents()
    print("Brown corpus Initialization: {}s".format(time.time() - start_time))
    start_time = time.time()
    hmmtagger = HMMTagger(brown_sents)
    print("HMMTagger Initialization: {}s".format(time.time() - start_time))
    start_time = time.time()
    print("HMMTagger forward results: ", hmmtagger.forward(["my", "brother", "needs", "a", "new", "tire"]))
    print("HMMTagger forward: {}s".format(time.time() - start_time))
    start_time = time.time()
    print("HMMTagger viterbi results: ", hmmtagger.viterbi(["my", "old", "man", "needs", "food"]))
    print("HMMTagger viterbi: {}s".format(time.time() - start_time))
    start_time = time.time()
    print("HMMTagger forward_backward results: ", hmmtagger.forward_backward(hmmtagger.transitions, hmmtagger.emissions, ["my", "old", "man", "needs", "food"]))
    print("HMMTagger forward_backward: {}s".format(time.time() - start_time))
    start_time = time.time()
    print("HMMTagger baum_welch results: ", hmmtagger.baum_welch(["my", "old", "man", "needs", "food"], 100))
    print("HMMTagger baum_welch: {}s".format(time.time() - start_time))
