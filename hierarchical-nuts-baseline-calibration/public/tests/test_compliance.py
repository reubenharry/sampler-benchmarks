"""Static compliance screening."""
import os

from evaluator.compliance import check_file, check_source

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD = '''
import jax.numpy as jnp
def phase_fractions(config): return (1.0,)
def init(x0, key, config): return {}
def step_phase(phase, carry, obs): return carry, {}
def transition(carry, history, info): return carry
def finalize(carry, history, config): return None
'''


def test_minimal_example_and_baseline_are_compliant():
    assert check_file(os.path.join(ROOT, "examples", "submission.py")) == []


def test_good_source_passes():
    assert check_source(GOOD) == []


def _messages(extra):
    return [m for _, m in check_source(GOOD + extra)]


def test_forbidden_constructs_are_flagged():
    assert any("import of 'os'" in m for m in _messages("\nimport os\n"))
    assert any("'open'" in m for m in _messages("\ndef f():\n    return open('x')\n"))
    assert any("jax.experimental" in m for m in _messages("\nfrom jax.experimental import io_callback\n"))
    assert any("__globals__" in m for m in _messages("\ndef f():\n    return f.__globals__\n"))
    assert any("import time" in m for m in _messages("\nprint('side effect')\n"))
    assert any("not allowed" in m for m in _messages("\nimport jax._src.core\n"))
    assert any("global/nonlocal" in m for m in _messages("\nSTATE = 0\ndef f():\n    global STATE\n    STATE = 1\n"))


def test_missing_function_is_flagged():
    assert any("finalize" in m for _, m in check_source(GOOD.replace("def finalize", "def other")))
