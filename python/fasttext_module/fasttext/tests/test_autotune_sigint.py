# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""Autotune must restore the SIGINT handler it replaced."""

import signal
import subprocess
import sys

_SCRIPT = """
import os, signal, tempfile, threading, time, fasttext
from fasttext.tests.helpers import get_random_data
data = get_random_data(3000, max_vocab_size=600)
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("".join(f"__label__{line}\\n" for line in data))
fasttext.train_supervised(
    f.name, autotuneValidationFile=f.name, autotuneDuration=2, verbose=0
)

def later(delay, func, *args):
    timer = threading.Timer(delay, func, args)
    timer.daemon = True
    timer.start()

try:
    signal.raise_signal(signal.SIGINT)
    time.sleep(1)  # KeyboardInterrupt is raised between bytecodes
except KeyboardInterrupt:
    print("KeyboardInterrupt")

if hasattr(signal, "pthread_kill"):
    # Ctrl-C must also interrupt a blocking call (no SA_RESTART).
    r, w = os.pipe()
    later(0.5, signal.pthread_kill, threading.get_ident(), signal.SIGINT)
    later(10, os.write, w, b"x")  # unblocks the read if SIGINT did not
    start = time.monotonic()
    try:
        os.read(r, 1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt" if time.monotonic() - start < 5 else "late")
"""


def test_autotune_restores_sigint_handler():
    """Used to leave a handler pointing at the destroyed Autotune."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    expected = ["KeyboardInterrupt"] * (2 if hasattr(signal, "pthread_kill") else 1)
    assert result.stdout.split() == expected, result.stderr[-2000:]
