"""Static compliance screening of a ``submission.py``.

Checks the import list, forbidden builtins, host callbacks, frame/introspection
tricks and import-time side effects. This is a review aid, not a security
boundary; the evaluation additionally runs in an OS-level sandbox (see
``evaluator/sandbox.py``).

Usage: ``python -m evaluator.compliance path/to/submission.py``
"""
import ast
import sys

ALLOWED_IMPORT_ROOTS = {
    "jax", "numpy", "math", "functools", "typing", "collections", "dataclasses", "enum", "itertools",
    "blackjax", "evaluator", "scipy", "__future__",
}
FORBIDDEN_IMPORT_PREFIXES = (
    "jax.experimental", "jax.debug", "jax.profiler", "jax.pure_callback", "jax._src", "jax.interpreters",
    "evaluator.runner", "evaluator.evaluate", "evaluator.metrics", "models", "tools", "private",
)
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "breakpoint", "memoryview"}
FORBIDDEN_ATTRIBUTES = {"__globals__", "__code__", "__closure__", "__class__", "__dict__", "__subclasses__",
                        "__builtins__", "__bases__", "__mro__", "f_back", "f_globals", "f_locals", "gi_frame",
                        "tb_frame", "_getframe", "io_callback", "pure_callback", "host_callback", "callback"}
REQUIRED_FUNCTIONS = ("phase_fractions", "init", "step_phase", "transition", "finalize")


def _import_names(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return [node.module or ""] if node.level == 0 else ["<relative import>"]


def check_source(source, filename="submission.py"):
    """Return a list of ``(line, message)`` violations found in ``source``."""
    tree = ast.parse(source, filename)
    problems = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in _import_names(node):
                root = name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS or name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    problems.append((node.lineno, f"import of '{name}' is not allowed"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            problems.append((node.lineno, f"call of '{node.func.id}' is not allowed"))
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            problems.append((node.lineno, f"attribute '{node.attr}' is not allowed"))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append((node.lineno, "global/nonlocal state is not allowed"))
        elif isinstance(node, ast.Name) and node.id in {"__builtins__", "__loader__", "__spec__"}:
            problems.append((node.lineno, f"name '{node.id}' is not allowed"))

    for node in tree.body:  # import-time code: definitions, imports and constant assignments only
        allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
        is_doc = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not isinstance(node, allowed) and not is_doc:
            problems.append((node.lineno, f"statement of type {type(node).__name__} at import time"))

    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in REQUIRED_FUNCTIONS:
        if name not in defined:
            problems.append((1, f"required function '{name}' is not defined at module level"))
    return sorted(problems)


def check_file(path):
    with open(path) as handle:
        return check_source(handle.read(), path)


if __name__ == "__main__":
    found = check_file(sys.argv[1])
    for line, message in found:
        print(f"{sys.argv[1]}:{line}: {message}")
    print("compliant" if not found else f"{len(found)} violation(s)")
    sys.exit(1 if found else 0)
