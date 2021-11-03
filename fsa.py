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

    def __init__(self, states: Set[str], alphabet: Set[str],  start_state: str, final_states: Set[str], transition_dict: Dict[Tuple[str, str], str]):
        """Initialize the fsa object"""

        #Set alphabet and state sets
        self.states = states
        self.alphabet = alphabet

        #Check and set if start_state and final_states are in states
        self.check_state(start_state)
        self.start_state = start_state
        self.check_final_states(final_states)
        self.final_states = final_states
        
        #Check and set transtion table in form of dictionary
        self.check_transition_dict(transition_dict)
        self.transition_dict = transition_dict

    def check_membership(self, element: str, elements: Set[str]) -> bool:
        """Check if element is in elements"""

        return element in elements

    def check_subset(self, subset: Set[str], superset: Set[str]) -> bool:
        """Check if subset is the subset of superset"""

        return subset.issubset(superset)

    def check_intersection(self, set1: Set[str], set2: Set[str]) -> bool:
        """Check if two sets have shared elements"""

        return len(set1.intersection(set2)) > 0

    def check_state(self, state: str):
        """Check if a state is an element of states"""

        if not self.check_membership(state, self.states):
            raise ValueError("start state must be in states")

    def check_not_state(self, state: str):
        """Check if a state is not an element of states"""

        if self.check_membership(state, self.states):
            raise ValueError("Input state can't be in states")

    def check_final_states(self, final_states: Set[str]):
        """Check if final_states is a subset of states"""

        if not self.check_subset(final_states, self.states):
            raise ValueError("final states must be subset of states")

    def check_string(self, string: str):
        """Check if a string is in the alphabet"""

        if not self.check_membership(string, self.alphabet):
            raise ValueError("String must be in alphabet")

    def check_transition_dict(self, delta: Dict[Tuple[str, str], str]):
        """Check if all keys in delta are elements of states and alpahbet sets"""

        for key, value in delta.items():
            self.check_state(key[0])
            self.check_string(key[1])
            self.check_state(value)

    
    def check_other_fsa(self, other_fsa: Union['FSA', 'NFSA']):
        """Check to make sure fsas dont overlap"""

        if self.check_intersection(self.states, other_fsa.states):
            raise ValueError("Operations on other FSAs must have unique states")



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

    def concatenate_transitions(self, trans_dict1: Dict[Tuple[str, str], Union[Set[str], str]], trans_dict2: Dict[Tuple[str, str], Union[Set[str], str]]) -> Dict[Tuple[str, str], Set[str]]:
        """Take two transition dicts and make a combination of them"""

        #Combine the dictionaries
        new_trans = {**trans_dict1, **trans_dict2}

        #Iterate over the new_dict and if the value is not a set it becomes one
        for key, value in new_trans.items():
            if not isinstance(value, set):
                new_trans[key] = {value}

        return new_trans

    def concatenate(self, other_fsa: Union['FSA', 'NFSA']) -> 'NFSA':
        """Concatenate two fsa objects together"""

        #Check to make sure other_fsa has unique states
        self.check_other_fsa(other_fsa)

        #Test this function
        new_states = self.states.union(other_fsa.states)
        new_alphabet = self.states.union(other_fsa.states)
        new_start = self.start_state
        new_finals = self.final_states.union(other_fsa.final_states)
        new_transitions = self.concatenate_transitions(self.transition_dict, other_fsa.transition_dict)
        for final_state in self.final_states:
            new_transitions[(final_state, '')] = {other_fsa.start_state}

        return NFSA(new_states, new_alphabet, new_start, new_finals, new_transitions)

    def close(self, new_start: str, new_final: str) -> 'NFSA':
        """Take a fsa and apply the Kleene closure to it"""

        #Check if these state names already exist in states
        self.check_not_state(new_start)
        self.check_not_state(new_final)

        #Get variables for closed nfsa
        new_states = self.states.union({new_start}).union({new_final})
        new_alphabet = self.alphabet
        new_finals = self.final_states.union({new_final})
        new_transitions = self.transition_dict

        #Make existing transitions nondeterministic
        for key, value in new_transitions.items():
            new_transitions[key] = {value}

        #Add empty string paths from new start state to previos start state and new final state
        new_transitions.update({(new_start, ''): {self.start_state, new_final}})

        #Add empty transitions from all the previous final states to the new one and previous start state
        for final_state in self.final_states:
            new_transitions.update({(final_state, ''): {self.start_state, new_final}})

        return NFSA(new_states, new_alphabet, new_start, new_finals, new_transitions)

    def union(self, other_fsa: Union['FSA', 'NFSA'], new_start: str, new_final: str) -> 'NFSA':
        """Return the Union of two FSAs"""

        #Check to make sure fsas dont share states
        self.check_other_fsa(other_fsa)

        #Check if these state names already exist in states
        self.check_not_state(new_start)
        self.check_not_state(new_final)
        other_fsa.check_not_state(new_start)
        other_fsa.check_not_state(new_final)

        #Get variables for closed nfsa
        new_states = self.states.union(other_fsa.states).union({new_start}).union({new_final})
        new_alphabet = self.alphabet.union(other_fsa.alphabet)
        new_finals = self.final_states.union(other_fsa.final_states)
        new_transitions = self.concatenate_transitions(self.transition_dict, other_fsa.transition_dict)
        new_transitions[(new_start, '')] = {self.start_state, other_fsa.start_state}
        for final_state in new_finals:
            new_transitions[(final_state, '')] = {new_final}
        new_finals = new_finals.union({new_final})

        return NFSA(new_states, new_alphabet, new_start, new_finals, new_transitions)

class NFSA(FSA):
    """A class object representing a non deterministic finite state automata
    
    This class Inherits from the FSA class with added functions to allow for
    non deterministic state paths.
    
    Attributes
    ----------
    Methods
    ----------
    """

    def __init__(self, states: Set[str], alphabet: Set[str], start_state: str, final_states: Set[str], transition_dict: Dict[Tuple[str, str], Set[str]]):
        
        #Add empty string to alphabet
        alphabet.add('')

        #Set alphabet and state sets
        self.states = states
        self.alphabet = alphabet

        #Check whether the inputs are valid
        self.check_state(start_state)
        self.start_state = start_state
        self.check_final_states(final_states)
        self.final_states = final_states
        
        self.check_transition_dict(transition_dict)
        self.transition_dict = transition_dict

    def check_dest_states(self, states: Set[str]):
        """Check if destination states are subsets of states"""
        if not self.check_subset(states, self.states):
            raise ValueError("The states transitioned to must be a subset of states")

    def check_transition_dict(self, delta: Dict[Tuple[str, str], Set[str]]):
        """Check if all keys in delta are elements of q and sigma and values are a subset of q"""

        for key, value in delta.items():
            self.check_state(key[0])
            self.check_string(key[1])
            self.check_dest_states(value)

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
            search_states += [(x, index) for x in self.transition_dict[(current_node, '')]]
        if index < len(tape):
            search_states += [(x, index+1) for x in self.transition_dict[(current_node, tape[index])]]

        return search_states

    def accept_state(self, tape: List[str], search_state: Tuple[str, int]) -> bool:
        """Test if in a final state at end of tape"""

        current_node = search_state[0]
        index = search_state[1]
        if index == len(tape) and current_node in self.final_states:
            return True
        else:
            return False

class SequentialFST(FSA):
    """A class to represent a finite state transducer
    
    This class object defines relations of strings that are accepted
    or rejected like a finite state automata.
    
    Attributes
    ----------
    Methods
    ----------
    """
    def __init__(self, states: Set[str], input_alpha: Set[str], output_alpha: Set[str], start_state: str, final_states: Set[str], trans_dict: Dict[Tuple[str, str], str], output_dict: Dict[Tuple[str, str], str]):
        """Initialize the SequentialFST object"""

        super().__init__(states, input_alpha, start_state, final_states, trans_dict)
        output_alpha.add('')
        self.output_alphabet = output_alpha

        self.check_output_dict(output_dict)
        self.output_dict = output_dict

    def check_output_string(self, string: str):
        """Check if string is in output alpha"""

        if not self.check_membership(string, self.output_alphabet):
            raise ValueError("Output string must be in output_alphabet")

    def check_output_dict(self, output_dict: Dict[Tuple[str, str], str]):
        """Check if a valid output dict"""

        for key, value in output_dict.items():
            self.check_state(key[0])
            self.check_string(key[1])
            self.check_output_string(value)

        if not self.transition_dict.keys() == output_dict.keys():
            raise ValueError("All transitions in transition_dict must be in output_dict")

    def transduce(self, tape: str) -> str:
        """Pass a string into the transducer and if accepted return the output string"""

        #Start read from the start state
        output_string = ''
        current_state = self.start_state

        #Iterate over all strings in list
        for x in tape:

            #Check if current state and current str are in dictionary
            #If so return new state
            if (current_state, x) in self.transition_dict.keys():
                output_string += self.output_dict[(current_state, x)]
                current_state = self.transition_dict[(current_state, x)]

            #If not return False
            else:
                return None
        
        #If the end of the list is in a final state return True
        if current_state in self.final_states:
            return output_string
        else:
            return None

    def invert_dicts(self) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], str]]:
        """Invert trans_dict and output_dict to switch key[1] with value"""

        new_trans_dict = {}
        new_output_dict = {}
        for key, value in self.output_dict.items():
            new_output_dict[(key[0], value)] = key[1]
            new_trans_dict[(key[0], value)] = self.transition_dict[key]

        return new_trans_dict, new_output_dict

    def invert(self):
        """Invert the fsts output and input alphabets along with transitions"""

        #Place alphabets into temp variables
        input_alpha = self.alphabet
        output_alpha = self.output_alphabet

        #Set the opposite alphabet as the new one
        self.alphabet = output_alpha
        self.output_alphabet = input_alpha

        #Invert dictionaries
        new_trans_dict, new_output_dict = self.invert_dicts()

        #Check new_dicts viability set transition and output dicts
        self.check_transition_dict(new_trans_dict)
        self.transition_dict = new_trans_dict
        self.check_output_dict(new_output_dict)
        self.output_dict = new_output_dict

    def compose(self, other_sfst: 'SequentialFST') -> 'SequentialFST':
        """"""
        pass


        
        
        

if __name__ == '__main__':
    states = {'0', '1'}
    alpha = {'a', 'b'}
    output_alpha = {'a', 'y', 'z'}
    start_state = '0'
    final_states = {'1'}
    transition_dict = {('0', 'a'): '0', ('0', 'b'): '1', ('1', 'a'): '0', ('1', 'b'): '1'}
    output_dict = {('0', 'a'): 'y', ('0', 'b'): 'z', ('1', 'a'): 'a', ('1', 'b'): 'z'}
    sfst = SequentialFST(states, alpha, output_alpha, start_state, final_states, transition_dict, output_dict)
    print(sfst.invert())
    print(sfst.alphabet)
    print(sfst.output_alphabet)
    print(sfst.transition_dict)
    print(sfst.output_dict)