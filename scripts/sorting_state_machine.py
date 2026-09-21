"""Small YAML-defined state transition table for the six-object task."""
from pathlib import Path


class SortingStateMachine:
    def __init__(self, definition):
        if not isinstance(definition, dict):
            raise ValueError('state machine must be a mapping')
        self.initial = definition.get('initial')
        self.terminal = definition.get('terminal')
        self.states = definition.get('states')
        if (not isinstance(self.initial, str) or not isinstance(self.terminal, list) or
                not self.terminal or not all(isinstance(item, str) for item in self.terminal) or
                not isinstance(self.states, dict) or self.initial not in self.states):
            raise ValueError('state machine needs an initial state, terminal states and transitions')
        names = set(self.states) | set(self.terminal)
        if set(self.states) & set(self.terminal):
            raise ValueError('terminal states cannot have outgoing transitions')
        for source, transitions in self.states.items():
            if not isinstance(source, str) or not isinstance(transitions, dict) or not transitions:
                raise ValueError('each nonterminal state needs event transitions')
            if 'error' not in transitions or transitions['error'] != 'failed':
                raise ValueError('each state must transition to failed on error')
            if any(not isinstance(event, str) or target not in names
                   for event, target in transitions.items()):
                raise ValueError('state machine contains an unknown target state')
        self.current = self.initial

    @classmethod
    def from_file(cls, path):
        import yaml
        with Path(path).open(encoding='utf-8') as stream:
            return cls(yaml.safe_load(stream))

    def advance(self, event):
        if self.current in self.terminal:
            raise RuntimeError('cannot leave terminal state ' + self.current)
        target = self.states[self.current].get(event)
        if target is None:
            raise RuntimeError('invalid state transition: {} + {}'.format(self.current, event))
        source = self.current
        self.current = target
        return source, event, target
