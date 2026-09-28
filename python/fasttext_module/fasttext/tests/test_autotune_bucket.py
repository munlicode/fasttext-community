# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""Autotune must keep a manually set bucket for n-gram trials."""

import fasttext

from .helpers import get_random_data


def test_autotune_with_manual_bucket(tmp_path):
    """Used to crash: bucket was zeroed, then n-gram trials hashed mod 0."""
    path = tmp_path / "train.txt"
    path.write_text("".join(f"__label__{line}\n" for line in get_random_data(3000)))
    # thread=12: thread <= 10 leaves the input matrix partly uninitialized.
    fasttext.train_supervised(
        str(path),
        autotuneValidationFile=str(path),
        autotuneDuration=3,  # the 3rd trial (fixed seed) samples wordNgrams=3
        bucket=1000,
        thread=12,
        verbose=0,
    )
