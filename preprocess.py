import time
import numpy as np
import pandas as pd

from nltk.util import ngrams
from nltk.corpus import brown

from collections import defaultdict, Counter

class Corpus:
    """Preprocess and extract data from natural language corpuses
    
    This object will contain lists of language data and perform functions on them.
    These functions will be able to return useful features of the dataset for model training.py
    
    Attributes
    ----------
    Methods
    ----------
    """

    def __init__(self, corpus):
        """Initialize corpus object"""

        self.corpus = corpus

    def get_ngrams(self, n=2):
        """Return the ngram frequencies of corpus"""

        co_counts = defaultdict(Counter)
        for sent in self.corpus:
            grams = ngrams(sent, n)
            for i, gram in enumerate(grams):

                if i == 0:
                    co_counts["<s>"][gram[0]] += 1

                co_counts[gram[0]][gram[1]] += 1

                if i == len(list(grams))-1:
                    co_counts[gram[1]]["</s>"] += 1

        co_df = pd.DataFrame(co_counts).fillna(0)
        co_df = co_df/co_df.sum(axis=0)

        start = co_df["<s>"]
        final = co_df.loc["</s>"]

        start.drop(["</s>"], inplace=True)
        final.drop(["<s>"], inplace=True)
        co_df.drop(["<s>"], axis=1, inplace=True)
        co_df.drop(["</s>"], axis=0, inplace=True)



        return start, co_df, final

if __name__ == '__main__':
    start_time = time.time()
    sents = brown.sents()
    print("Brown corpus Initialization: {}s".format(time.time() - start_time))
    start_time = time.time()
    corpus = Corpus(sents)
    print("Corpus Initialization: {}s".format(time.time() - start_time))
    start_time = time.time()
    print("Corpus get_ngrams results: {}".format(corpus.get_ngrams()))
    print("Corpus get_ngrams: {}s".format(time.time() - start_time))