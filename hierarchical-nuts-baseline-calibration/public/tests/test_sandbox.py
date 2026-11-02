"""The sandbox blocks path reads of private files, network access and writes outside scratch."""
import json
import os
import shutil
import sys
import textwrap

import pytest

from evaluator.sandbox import run_sandboxed

PROBE = textwrap.dedent('''
    import json, os, socket, sys
    out = {}
    def attempt(name, fn):
        try:
            out[name] = fn()
        except Exception as error:
            out[name] = "BLOCKED"
    attempt("private_path", lambda: open(sys.argv[1]).read())
    attempt("public_path", lambda: open(sys.argv[2]).read()[:5])
    attempt("inherited_fd", lambda: os.fdopen(int(sys.argv[3]), closefd=False).read())
    attempt("network", lambda: bool(socket.create_connection(("1.1.1.1", 53), timeout=3)))
    attempt("write_outside", lambda: open(sys.argv[4], "w").write("x"))
    attempt("write_scratch", lambda: open(os.path.join(os.environ["TMPDIR"], "ok"), "w").write("x"))
    print(json.dumps(out))
''')


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec profile is macOS only")
def test_sandbox_blocks_private_reads_network_and_outside_writes():
    home = os.path.expanduser("~")
    work = os.path.join(home, ".aop_sandbox_test")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        for name, text in {"private.txt": "SECRET", "public.txt": "PUBLIC-DATA", "fd.txt": "FD-DATA"}.items():
            with open(os.path.join(work, name), "w") as handle:
                handle.write(text)
        probe = os.path.join(work, "probe.py")
        with open(probe, "w") as handle:
            handle.write(PROBE)
        fd = os.open(os.path.join(work, "fd.txt"), os.O_RDONLY)
        result = run_sandboxed(
            [sys.executable, "-I", probe, os.path.join(work, "private.txt"), os.path.join(work, "public.txt"),
             str(fd), os.path.join(work, "outside.txt")],
            home=home, allow_read=[os.path.join(work, "public.txt"), os.path.join(work, "probe.py"),
                                   os.path.dirname(os.path.dirname(sys.executable))],
            scratch=os.path.join(work, "scratch"), pass_fds=[fd], timeout=120)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        assert report["private_path"] == "BLOCKED"
        assert report["public_path"] == "PUBLI"
        assert report["inherited_fd"] == "FD-DATA"
        assert report["network"] == "BLOCKED"
        assert report["write_outside"] == "BLOCKED"
        assert report["write_scratch"] == 1
        assert not os.path.exists(os.path.join(work, "outside.txt"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
