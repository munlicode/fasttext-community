# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""Regression test: full initialization of the input matrix.

With thread <= 10, DenseMatrix::uniform used to leave part of the input
matrix uninitialized (for thread=10, the remainder after 10 equal blocks), so
training could raise "Encountered NaN" or give results that varied between
processes.
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
dim = 13  # 101 rows (100 words + </s>) x 13 = 1313: leaves a tail block
# Pretrained vector for one word; in range, so Check 1 still applies.
vec = tempfile.NamedTemporaryFile("w", suffix=".vec", delete=False)
vec.write("1 %d\nw0%s\n" % (dim, " 0.01" * dim))
vec.close()
result = {"bad_init": {}, "init_hash": {}}
try:
    bound = np.float32(1.0 / dim)  # values are float32
    # 11 blocks (10 + tail). thread=1: sequential; 2, 10: striding;
    # 11: one block per worker; 12: more threads than blocks.
    # Keep thread=1 first: freed matrix memory is reused by the next run.
    # Random init, then pretrained init (pv): both call uniform().
    runs = [(t, pv) for pv in ("", vec.name) for t in (1, 2, 10, 11, 12)]
    for thread, pv in runs:
        # lr=0 leaves the input matrix as initialized.
        m = fasttext.train_supervised(
            f.name, thread=thread, epoch=1, lr=0.0, dim=dim, minCount=1, verbose=0,
            pretrainedVectors=pv,
        )
        key = ("p%d" if pv else "%d") % thread
        M = m.get_input_matrix()
        result["tail"] = int(M.size % 10)
        # Check 1: uninitialized memory shows up as NaN/inf, zero or out of range.
        bad = ~np.isfinite(M) | (np.abs(M) > bound) | (M == 0)
        result["bad_init"][key] = int(bad.sum())
        # Check 2: same matrix for every thread count (and every process).
        result["init_hash"][key] = hashlib.sha256(M.tobytes()).hexdigest()
except RuntimeError as e:  # e.g. "Encountered NaN" from uninitialized values
    result["error"] = str(e)
finally:
    os.unlink(f.name)
    os.unlink(vec.name)
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


def test_init_is_complete_and_deterministic():
    """Training initializes the whole input matrix, for any thread count.

    Runs fresh processes and checks that:
    1. every initial value is finite, non-zero and in [-1/dim, 1/dim]
    2. the initial matrix is identical for every thread count and process
    """
    # Uninitialized memory contents vary per process, so use fresh ones.
    results = [_run_child() for _ in range(5)]
    for r in results:
        assert "error" not in r, r["error"]
        assert r["tail"] != 0  # the tail block is exercised
        # Check 1
        assert set(r["bad_init"].values()) == {0}, r["bad_init"]
    # Check 2: per init path (random, pretrained), one hash for all runs
    for p in (False, True):
        runs = [h for r in results for k, h in r["init_hash"].items()]
        hashes = {h for r in results for k, h in r["init_hash"].items()
                  if k.startswith("p") == p}
        assert len(hashes) == 1, [r["init_hash"] for r in results]
