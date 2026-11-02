"""OS-level sandbox for running a submission evaluation (macOS ``sandbox-exec``).

The profile denies all network access, denies writes outside a scratch
directory and denies reads under the home directory except for the listed
allow paths. Data the process must see but candidates must not be able to open
by path (private references, target data) is passed as an already-open file
descriptor. This reduces the attack surface; candidate code can still read
whatever the process holds in memory, which the static screening in
``evaluator/compliance.py`` is meant to prevent.
"""
import os
import shutil
import subprocess
import sys


def macos_profile(home, allow_read, scratch):
    """Return a sandbox-exec profile string."""
    reads = " ".join(f'(subpath "{p}")' for p in allow_read)
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{scratch}") (literal "/dev/null") (literal "/dev/dtracehelper") (subpath "/dev/fd"))
(deny file-read* (subpath "{home}"))
(allow file-read-metadata)
(allow file-read* {reads} (subpath "{scratch}"))
"""


def run_sandboxed(argv, home, allow_read, scratch, pass_fds=(), timeout=None, env=None, cwd=None):
    """Run ``argv`` under the sandbox; returns the ``CompletedProcess``.

    Parameters
    ----------
    argv : list of str
        Command to run.
    home : str
        Directory tree whose reads are denied by default (e.g. the user's home).
    allow_read : list of str
        Subtrees under ``home`` that remain readable (virtual environment, public repository).
    scratch : str
        Only writable directory; also used as ``TMPDIR``.
    pass_fds : sequence of int
        Inherited descriptors (e.g. an open private data file).
    """
    if sys.platform != "darwin" or shutil.which("sandbox-exec") is None:
        raise RuntimeError("sandbox-exec is only available on macOS; use bubblewrap or a container on Linux")
    os.makedirs(scratch, exist_ok=True)
    profile = macos_profile(home, allow_read, scratch)
    environment = dict(os.environ if env is None else env, TMPDIR=scratch, HOME=scratch, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(["sandbox-exec", "-p", profile, *argv], capture_output=True, text=True,
                          pass_fds=tuple(pass_fds), timeout=timeout, env=environment, cwd=cwd or scratch)
