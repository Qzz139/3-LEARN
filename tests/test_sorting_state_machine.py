"""Exercise the actual YAML transitions used by the six-object controller."""
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from sorting_state_machine import SortingStateMachine  # noqa: E402


class StateMachineTests(unittest.TestCase):
    def test_successful_object_then_finish(self):
        machine = SortingStateMachine.from_file(ROOT / 'config/sorting_state_machine.yaml')
        for event in ('ready', 'target', 'found', 'ready', 'held', 'centered', 'released',
                      'complete', 'done'):
            machine.advance(event)
        self.assertEqual(machine.current, 'finished')
        with self.assertRaises(RuntimeError):
            machine.advance('ready')

    def test_empty_sector_and_error(self):
        machine = SortingStateMachine.from_file(ROOT / 'config/sorting_state_machine.yaml')
        for event in ('ready', 'target', 'empty'):
            machine.advance(event)
        self.assertEqual(machine.current, 'select_direction')
        machine.advance('error')
        self.assertEqual(machine.current, 'failed')

    def test_unknown_transition_rejected(self):
        machine = SortingStateMachine.from_file(ROOT / 'config/sorting_state_machine.yaml')
        with self.assertRaises(RuntimeError):
            machine.advance('held')


if __name__ == '__main__':
    unittest.main()
