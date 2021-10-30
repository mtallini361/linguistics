import math
from typing import List, Dict, Tuple, Set, Union, Type

class FSA:
    """A class object representing a finite state automata
    
    This class will be used to contain the states alphabet and transition function
    required to represent a finite state automata and be used to recognize strings.
    
    Attributes
    ----------
    states: Set[str]
    input_alphabet: Set[str]
    start_state: str
    final_states: Set[str]
    transition_dict: Dict[Tuple[str, str], str]

    Methods
    ----------
    check_membership(element: str, elements: Set[str])
        Check if element is in elements aand raise error if not
    check_subset(F: Set[str], Q: Set[str])
        Check if F is a subset of Q and raise error if not
    check_transition_dict(Q: Set[str], Sigma: Set[str], delta: Dict[Tuple[str, str], str])
        Check if all keys in delta are elements of Q x Sigma and all values are elements in Q
    """

    def __init__(self, states: Set[str], input_alphabet: Set[str], start_state: str, final_states: Set[str], transition_dict: Dict[Tuple[str, str], str]):
        """Initialize the fsa object"""

        #Check whether the inputs are valid
        self.check_membership(start_state, states)
        self.check_subset(final_states, states)
        self.check_transition_dict(states, input_alphabet, transition_dict)

        #Set the machine attributes
        self.states = states
        self.input_alphabet = input_alphabet
        self.start_state = start_state
        self.final_states = final_states
        self.transition_dict = transition_dict

    def check_membership(self, element: str, elements: Set[str]):
        """Check if element is in elements"""

        if element in elements:
            pass
        else:
            raise ValueError('Variables need to be members of states and input_alphabet')

    def check_subset(self, F: Set[str], Q: Set[str]):
        """Check if subset is the subset of superset"""

        if F.issubset(Q):
            pass
        else:
            raise ValueError('Final states must be a subset of states')

    def check_transition_dict(self, Q: Set[str], Sigma: Set[str], delta: Dict[Tuple[str, str], str]):
        """Check if all keys in delta are elements of q and sigma and values are elements of q"""

        for key, value in delta.items():
            self.check_membership(key[0], Q)
            self.check_membership(key[1], Sigma)
            self.check_membership(value, Q)

    def recognize(self, tape: List[str]) -> bool:
        """Check if the sequence of strings is recognized by the fsa"""

        #Start read from the start state
        current_state = self.start_state

        #Iterate over all strings in list
        for x in tape:

            #Check if current state and current str are in dictionary
            #If so return new state
            if (current_state, x) in self.transition_dict.keys():
                current_state = self.transition_dict[(current_state, x)]

            #If not return False
            else:
                return False
        
        #If the end of the list is in a final state return True
        if current_state in self.final_states:
            return True
        else:
            return False

class NFSA(FSA):
    """A class object representing a non deterministic finite state automata
    
    This class Inherits from the FSA class with added functions to allow for
    non deterministic state paths.
    
    Attributes
    ----------
    Methods
    ----------
    """

    def __init__(self, states: Set[str], input_alphabet: Set[str], start_state: str, final_states: Set[str], transition_dict: Dict[Tuple[str, str], str]):
        input_alphabet.add('')
        super().__init__(states, input_alphabet, start_state, final_states, transition_dict)

    def recognize(self, tape: List[str]) -> bool:
        """Check if the sequence of strings is recognized by the nfsa"""

        agenda = [(self.start_state, 0)]
        current_search_state = agenda.pop()

        for _ in range(math.factorial(len(tape))):
            if self.accept_state(tape, current_search_state):
                return True
            else:
                agenda += self.generate_new_states(tape, current_search_state)
            if len(agenda) == 0:
                return False
            else:
                current_search_state = agenda.pop()

        return False

    def generate_new_states(self, tape: List[str], current_state: Tuple[str, int]) -> List[Tuple[str, int]]:
        """Return a list of all the possible state paths at current_state with tape"""

        current_node = current_state[0]
        index = current_state[1]

        search_states = []
        if (current_node, '') in self.transition_dict.keys():
            search_states.append((self.transition_dict[(current_node, '')], index))
        if index < len(tape):
            search_states.append((self.transition_dict[(current_node, tape[index])], index+1))

        return search_states

    def accept_state(self, tape: List[str], search_state: Tuple[str, int]) -> bool:
        """Test if in a final state at end of tape"""

        current_node = search_state[0]
        index = search_state[1]
        if index == len(tape) and current_node in self.final_states:
            return True
        else:
            return False

def concatenate(fsa1: Union[FSA, NFSA], fsa2: Union[FSA, NFSA]) -> NFSA:
    """Concatenate two fsa objects together"""

    new_states = fsa1.states.union(fsa2.states)
    new_alphabet = fsa1.states.union(fsa2.states)
    new_start = fsa1.start_state
    new_finals = fsa1.final_states.union(fsa2.final_states)
    new_transitions = fsa1.transition_dict.update(fsa2.transition_dict)
    for final_state in fsa1.final_states:
        new_transitions[(final_state, '')] = fsa2.start_state

    return NFSA(new_states, new_alphabet, new_start, new_finals, new_transitions)

states = {'0', '1'}
input_alpha = {'a', 'b'}
start = '0'
final_states = {'1'}
transition_dict = {('0', ''): '0', ('0', 'a'): '0', ('0', 'b'): '1', ('1', ''): '1', ('1', 'a'): '1', ('1', 'b'): '0'}
nfsa = NFSA(states, input_alpha, start, final_states, transition_dict)
print(nfsa.recognize('aaaaaaaaaaab'))