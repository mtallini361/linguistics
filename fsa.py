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