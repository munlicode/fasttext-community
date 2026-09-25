# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""Regression test: full initialization of the input matrix.

With thread < 10, DenseMatrix::uniform used to fill only thread/10 of the
input matrix and leave the rest uninitialized, so training could raise
"Encountered NaN" or give results that varied between processes.
"""

import json
import os
import subprocess
import sys

import fasttext

_CHILD = r"""
import hashlib, json, os, random, sys, tempfile
import numpy as np
import fasttext

rnd = random.Random(0)
words = ["w%d" % i for i in range(100)]
f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
for _ in range(100):
    line = [rnd.choice(words) for _ in range(rnd.randint(1, 20))]
    f.write("__label__%s %s\n" % (line[0], " ".join(line)))
f.close()
result = {}
try:
    dim = 50
    # lr=0 leaves the input matrix as initialized.
    m = fasttext.train_supervised(
        f.name, thread=1, epoch=1, lr=0.0, dim=dim, minCount=1, verbose=0
    )
    M = m.get_input_matrix()
    # Check 1: uninitialized memory shows up as NaN/inf, zero or out of range.
    bad = ~np.isfinite(M) | (np.abs(M) > 1.0 / dim) | (M == 0)
    result["bad_init"] = int(bad.sum())
    # Check 2: any RuntimeError, from either call, is recorded by the except.
    m = fasttext.train_supervised(
        f.name, thread=1, epoch=5, minCount=1, verbose=0
    )
    # Check 3: hash of the trained matrix.
    result["hash"] = hashlib.sha256(m.get_input_matrix().tobytes()).hexdigest()
except RuntimeError as e:
    result["error"] = str(e)
finally:
    os.unlink(f.name)
print(json.dumps(result))
"""


def _run_child():
    env = dict(os.environ)
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(fasttext.__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (pkg_root, env.get("PYTHONPATH")) if p
    )
    out = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_single_thread_init_is_complete_and_deterministic():
    """Single-thread training initializes the whole input matrix.

    Trains with thread=1 in several fresh processes and checks that:
    1. every initial value is finite, non-zero and in [-1/dim, 1/dim]
    2. training raises no RuntimeError, such as "Encountered NaN"
    3. the trained input matrix is identical across processes
    """
    # Uninitialized memory contents vary per process, so use fresh ones.
    results = [_run_child() for _ in range(5)]
    for r in results:
        # Check 2 first: an errored run has no "bad_init".
        assert "error" not in r, r["error"]
        # Check 1
        assert r["bad_init"] == 0
    # Check 3
    assert len({r["hash"] for r in results}) == 1
