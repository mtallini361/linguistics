import time
from collections import Counter, defaultdict
from typing import Set, Any, List, Tuple, Dict, Union

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import brown

from fsa import FSA

class MarkovChain(FSA):
    """Class to represent a Markov Chain
    
    This object is used for statistical traversal of fsa
    
    Attributes
    ----------
    Methods
    ----------
    """

    def __init__(self, states: Set[str], start: pd.Series, final: pd.Series, transitions: pd.DataFrame):
        """Initialize markov chain model"""

        self.states = states

        self.check_start(start)
        self.start = start

        self.check_final(final)
        self.final = final
        
        self.check_transitions(transitions)
        self.transitions = transitions

    def check_values(self, table: Union[pd.Series, pd.DataFrame]) -> bool:
        """Check if the values in transitions are valid"""

        if table.to_numpy().dtype != np.float:
            return False
        else:
            return True

    def check_start(self, start: pd.Series):
        """Check if the start series is valid"""

        if set(start.index) != self.states:
            raise ValueError("Index of start series must include all states")
        if not self.check_values(start):
            raise ValueError("Values of start series must be type float")
        if not np.isclose(start.to_numpy().sum(), [1.0]):
            raise ValueError("Start series values must add to 1")

    def check_final(self, final: pd.Series):
        """Check if the final series is valid"""

        if set(final.index) != self.states:
            raise ValueError("Index of final series must include all states")
        if not self.check_values(final):
            raise ValueError("Values of final series must be float")

    def check_transitions(self, transitions: pd.DataFrame):
        """Check if this is a valid transitions dataframe"""

        if not all(elem in self.states for elem in transitions.columns):
            raise ValueError("Transition columns must have every state in states")
        if not all(elem in self.states for elem in transitions.index):
            raise ValueError("Transition index must have every state in states")
        if not self.check_values(transitions):
            raise ValueError("Transition values must be floats")
            
if __name__ == '__main__':
    states = {'1', '2', '3'}
    start = pd.Series({'1': float(1/3), '2': float(1/3), '3': float(1/3)})
    final = pd.Series({'1': .5, '2': .34, '3': .62})
    index = list(states)
    index.sort()
    trans = [{'1': .2, '2': .4, '3': .4},
    {'1': .4, '2': .2, '3': .4},
    {'1': .4, '2': .4, '3': .2}]
    transitions = pd.DataFrame(trans, index=index)
    start_time = time.time()
    mc = MarkovChain(states, start, final, transitions)
    print("MarkovChain Initialization: {}s".format(time.time()-start_time))

class HiddenMarkovModel(MarkovChain):
    """This object will represent a hidden markov model
    
    This object is responsible for adding observations to a markov chain
    and having the ability to traverse the model to get the sequence of
    tags with the highest probability
    
    Attributes
    ----------
    Methods
    ----------
    """

    def __init__(self, states: Set[str], start: pd.Series, final: pd.Series, transitions: pd.DataFrame, emissions: pd.DataFrame):
        """Initialize the hidden markov model"""

        super().__init__(states, start, final, transitions)

        self.check_emissions(emissions)
        self.emissions = emissions

    def check_emissions(self, emissions: pd.DataFrame):
        """Check if observations is valid"""

        if not all(elem in self.states for elem in emissions.columns):
            raise ValueError("Observation columns must have every state in states")
        if not self.check_values(emissions):
            raise ValueError("Observation values must be floats that add to 1")

    def forward(self, observed: List[Any]) -> float:
        """Compute the liklihood of an observed sequence"""

        forward = pd.DataFrame(self.start * self.emissions.loc[observed[0]])

        for t in range(1, len(observed)):
            forward[t] =  self.transitions @ forward[t-1] @ self.emissions.loc[observed[t]]
        
        return  forward[len(observed)-1] @ self.final.T

    def viterbi(self, observed: List[Any]) -> List[str]:
        """Compute the most likely sequence hidden states"""

        viterbi = pd.DataFrame(self.start * self.emissions.loc[observed[0]])
        backpointer = pd.DataFrame(np.empty((self.start.shape[0], 1)), index=self.start.index)
        
        for t in range(1, len(observed)):
           prod_matrix = self.transitions.multiply(viterbi[t-1], axis="columns").multiply(self.emissions.loc[observed[t]], axis='index') 
           viterbi[t] = prod_matrix.max(axis=1)
           backpointer[t] = self.transitions.multiply(viterbi[t-1], axis="columns").idxmax()

        final_viterbi = (viterbi[len(observed)-1] * self.final).max()
        final_backpointer = (viterbi[len(observed)-1] * self.final).idxmax()
        previous = final_backpointer
        
        best_path = [previous]
        for t in range(len(observed) - 2, -1, -1):
            best_path.insert(0, backpointer[t+1][previous])
            previous = backpointer[t+1][previous]

        return best_path


if __name__ == '__main__':
    states = {'rainy', 'sunny'}
    index = list(states)
    index.sort()
    start = pd.Series({'rainy': .25, 'sunny': .75})
    final = pd.Series({'rainy': .75, 'sunny': .25})
    transitions = pd.DataFrame(
        [
            {'rainy': .2, 'sunny': .5},
            {'rainy': .8, 'sunny': .5}
        ], index=index
    )
    observations = pd.DataFrame(
        [
            {'rainy': .1, 'sunny': .4},
            {'rainy': .9, 'sunny': .6}
        ], index=['sad', 'happy']
    )
    start_time = time.time()
    hmm = HiddenMarkovModel(states, start, final, transitions, observations)
    print("HiddenMarkovModel Initialization: {}s".format(time.time() - start_time))
    start_time = time.time()
    print(hmm.forward(["happy", "sad", "sad", "happy"]))
    print("HiddenMarkovModel forward: {}s".format(time.time() - start_time))
    start_time = time.time()
    print(hmm.viterbi(["happy", "happy", "happy", "happy"]))
    print("HiddenMarkovModel viterbi: {}s".format(time.time() - start_time))

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
    print(hmmtagger.forward(["my", "brother", "needs", "a", "new", "tire"]))
    print("HMMTagger forward: {}s".format(time.time() - start_time))
    start_time = time.time()
    print(hmmtagger.viterbi(["i", "need", "a", "new", "tire"]))
    print("HMMTagger viterbi: {}s".format(time.time() - start_time))