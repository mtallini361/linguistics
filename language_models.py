import math
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Union, Type, Any

class FSA:
    """Create a class to represent a finite state automaton
    
    This class will contain all the functions of a finite state automaton
    and its attributes. It will hold the states alphabet and transition matrix
    required to recognize sequences as part of the language or as not
    
    Attributes
    ----------
    Methods
    ----------
    """

    def __init__(self, states: Set[str], alphabet: Set[Any], start: str, final: Set[str], transitions: pd.DataFrame):
        """Initialize the fsa object"""

        self.states = states
        self.alphabet = alphabet

        self.check_states_membership(start)
        self.start = start

        self.check_states_subset(final)
        self.final = final
        
        self.check_transitions(transitions)
        self.transitions = transitions

    def check_states_membership(self, state: str):
        """Check if this is a valid state"""

        if not state in self.states:
            raise ValueError("State must be in states")

    def check_states_subset(self, sub_states: Set[str]):
        """Check if this is a valid set of final states"""

        if not sub_states.issubset(self.states):
            raise ValueError("Final states must be a subset of states")

    def check_values(self, table: pd.DataFrame) -> bool:
        """Check if the values of transitions are valid"""

        valid_values = list(self.states)
        valid_values.append(None)

        return np.isin(table.to_numpy(), valid_values).all()

    def check_transitions(self, transitions: pd.DataFrame):
        """Check if this is a valid transitions dataframe"""

        if not all(elem in self.states for elem in transitions.columns):
            raise ValueError("Transition columns must have every state in states")
        if not all(elem in self.alphabet for elem in transitions.index):
            raise ValueError("Transition index must have every symbol in alphabet")
        if not self.check_values(transitions):
            raise ValueError("Transition values must be in states or None")

    def recognize(self, tape: List[Any]) -> bool:
        """Check if the sequence of symbols is recognized by the fsa"""

        index = 0
        current_state = self.start

        for x in tape:
            new_state = self.transitions[current_state][tape[index]]
            if new_state is None:
                return False
            else:
                current_state = new_state
                index += 1

        if current_state in self.final:
            return True
        else:
            return False

if __name__ == '__main__':
    states = {'0', '1'}
    alphabet = {'a', 'b'}
    start = '0'
    final = {'1'}
    index = list(alphabet)
    index.sort()
    transitions = pd.DataFrame([{'0': '0', '1': '1'}, {'0': '1', '1': None}], index=index)
    start_time = time.time()
    fsa = FSA(states, alphabet, start, final, transitions)
    print("FSA Initalization: {}s".format(time.time() - start_time))
    start_time = time.time()
    fsa.recognize("babababababababa")
    print("FSA recognize: {}s".format(time.time() - start_time))


class NFSA(FSA):
    """Create a class that represents a nondeterministic fsa
    
    This object will be able to do everything a fsa can but instead of
    transitioning to one state at any given state x symbol combination,
    it will transition to a set of states for any state x symbol combo
    
    Attributes
    ----------
    Methods
    ----------
    """

    def __init__(self, states: Set[str], alphabet: Set[Any], start: str, final: Set[str], transitions: pd.DataFrame):
        """Initialize the fsa object"""

        self.states = states
        self.alphabet = alphabet

        self.check_states_membership(start)
        self.start = start

        self.check_states_subset(final)
        self.final = final
        
        self.check_transitions(transitions)
        self.transitions = transitions
        
    
    def power_set(self, seq: Set[str]) -> Set[Set[str]]:
        """Return the powerset of a sequence"""

        length = len(seq)
        return {
            frozenset({e for e, b in zip(seq, f'{i:{length}b}') if b == '1'})
            for i in range(2 ** length)
        }

    def check_values(self, table: pd.DataFrame) -> bool:
        """Check if the values of transitions are valid"""

        ps = self.power_set(self.states)

        return np.isin(table.to_numpy(), list(ps)).all()

    def recognize(self, tape: List[Any]) -> bool:
        """Check if tape is accepted by nfsa"""

        agenda = [(self.start, 0, tape)]
        current_search_state = agenda.pop()

        while True:
            if self.accept_state(current_search_state):
                return True
            else:
                agenda += self.generate_new_states(current_search_state)
            if len(agenda) == 0:
                return False
            else:
                current_search_state = agenda.pop()

    def generate_new_states(self, current_state: Tuple[str, int, List[Any]]) -> List[Tuple[str, int, List[Any]]]:
        """Return a list of new search states to traverse"""

        current_node = current_state[0]
        index = current_state[1]
        tape = current_state[2]

        new_search_states = []
        if '' in self.transitions.index:
            new_search_states += [(x, index, tape) for x in self.transitions[current_node]['']]
        if index < len(tape):
            new_search_states += [(x, index+1, tape) for x in self.transitions[current_node][tape[index]]]

        return new_search_states

    def accept_state(self, current_state: Tuple[str, int, List[Any]]) -> bool:
        """Check if current search state is in a final state and end of tape"""
        print(current_state)
        current_node = current_state[0]
        index = current_state[1]
        tape = current_state[2]

        if index == len(tape) and current_node in self.final:
            return True
        else:
            return False

if __name__ == '__main__':
    states = {'0', '1'}
    alphabet = {'', 'a', 'b'}
    start = '0'
    final = {'1'}
    index = list(alphabet)
    index.sort()
    transitions = pd.DataFrame([{'0': {'0'}, '1': {'1'}}, {'0': {'1'}, '1': {'0'}}, {'0': {'0'}, '1': {'0'}}], index=index)
    start_time = time.time()
    nfsa = NFSA(states, alphabet, start, final, transitions)
    print("NFSA Initalization: {}s".format(time.time() - start_time))
    start_time = time.time()
    nfsa.recognize("ba")
    print("NFSA recognize: {}s".format(time.time() - start_time))


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
            forward[t] =  self.transitions.multiply(forward[t-1], axis="columns").multiply(self.emissions.loc[observed[t]], axis="index").sum(axis=1)
        
        return  forward[len(observed)-1] @ self.final.T

    def viterbi(self, observed: List[Any]) -> List[str]:
        """Compute the most likely sequence hidden states"""

        viterbi = pd.DataFrame(self.start * self.emissions.loc[observed[0]])
        backpointer = pd.DataFrame(np.empty((self.start.shape[0], 1)), index=self.start.index)
        
        for t in range(1, len(observed)):
           prod_matrix = self.transitions.multiply(viterbi[t-1], axis="columns").multiply(self.emissions.loc[observed[t]], axis='index')
           viterbi[t] = prod_matrix.max(axis=1)
           backpointer[t] = self.transitions.multiply(viterbi[t-1], axis="columns").idxmax(axis=1)

        final_viterbi = (viterbi[len(observed)-1] * self.final).max()
        final_backpointer = (viterbi[len(observed)-1] * self.final).idxmax(axis=1)
        previous = final_backpointer
        
        best_path = [previous]
        for t in range(len(observed) - 2, -1, -1):
            best_path.insert(0, backpointer[t+1][previous])
            previous = backpointer[t+1][previous]

        return best_path

    def forward_backward(self, transition_df: pd.DataFrame, emission_df: pd.DataFrame, observed: List[Any]) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Return the forward and backward liklihoods of the observed"""

        forward = pd.DataFrame(self.start * emission_df.loc[observed[0]])
        for t in range(1, len(observed)):
            forward[t] = transition_df.multiply(forward[t-1], axis="columns").sum(axis=1)
            forward[t] = forward[t] * emission_df.loc[observed[t]]

        p_fwd = forward[len(observed)-1] @ self.final

        col = self.final.copy()
        col.name = len(observed) - 1
        backward = pd.DataFrame(col)
        for t in range(len(observed)-2, -1, -1):
            backward[t] = transition_df.multiply(emission_df.loc[observed[t+1]], axis="index").multiply(backward[t+1], axis="index").sum(axis=0)

        p_bkw = (self.start * emission_df.loc[observed[0]] * backward[0]).sum()

        if not math.isclose(p_fwd, p_bkw):
            print(p_fwd)
            print(p_bkw)
        assert math.isclose(p_fwd, p_bkw)
        return forward, backward, p_fwd

    def baum_welch(self, observed: List[Any], n_iter: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return optimized transition and emission dataframes"""

        transitions = self.transitions
        emissions = self.emissions
        for _ in range(n_iter):
            old_transitions = transitions
            old_emissions = emissions
            transitions = self.transitions
            emissions = self.emissions
            # expectation step, forward and backward probs
            alpha, beta, posterior = self.forward_backward(old_transitions, old_emissions, observed)
            
            gamma = (alpha * beta).div((alpha * beta).sum(axis=0), axis="columns")
            xi = []
            for t in range(len(observed) - 1):
                num = old_transitions.multiply(alpha[t], axis="columns").multiply(beta[t+1], axis="index").multiply(old_emissions.loc[observed[t+1]], axis="index")
                xi_t = num / posterior
                xi.append(xi_t)
            
            xi = reduce(lambda x, y: x.add(y), xi).div(reduce(lambda x, y: x.add(y), [z.sum(axis=0) for z in xi]), axis="columns")
            transitions = xi.combine_first(transitions)
            transitions = transitions.div(transitions.sum(axis=0) + self.final, axis="columns")

            observed_vocab = list(set(observed))
            for v in observed_vocab:
                v_sum = []
                for t in range(len(observed)):
                    if observed[t] == v:
                        v_sum.append(gamma[t])
                emissions.loc[v] = (reduce(lambda x, y: x.add(y), v_sum) / gamma.sum(axis=1)).combine_first(emissions.loc[v])
            emissions = emissions.div(emissions.sum(axis=0), axis="columns")

        # get out   
        return transitions, emissions

if __name__ == '__main__':
    states = {'rainy', 'sunny'}
    index = list(states)
    index.sort()
    start = pd.Series({'rainy': .25, 'sunny': .75})
    final = pd.Series({'rainy': .1, 'sunny': .2})
    transitions = pd.DataFrame(
        [
            {'rainy': .1, 'sunny': .4},
            {'rainy': .8, 'sunny': .4}
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
    print("HiddenMarkovModel forward results: ", hmm.forward(["happy", "sad", "sad", "happy"]))
    print("HiddenMarkovModel forward: {}s".format(time.time() - start_time))
    start_time = time.time()
    print("HiddenMarkovModel viterbi results: ", hmm.viterbi(["happy", "sad", "happy", "happy", "sad"]))
    print("HiddenMarkovModel viterbi: {}s".format(time.time() - start_time))
    start_time = time.time()
    print("HiddenMarkovModel forward_backward results: ", hmm.forward_backward(hmm.transitions, hmm.emissions, ["happy", "happy", "happy", "happy"]))
    print("HiddenMarkovModel forward_backward: {}s".format(time.time() - start_time))
    start_time = time.time()
    print("HiddenMarkovModel baum_welch results: ", hmm.baum_welch(["happy", "sad", "happy", "happy", "sad"], 100))
    print("HiddenMarkovModel baum_welch: {}s".format(time.time() - start_time))