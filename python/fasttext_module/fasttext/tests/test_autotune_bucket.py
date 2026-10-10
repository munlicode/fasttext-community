# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""Autotune must respect a manually set bucket."""

import fasttext
import fasttext_pybind
from fasttext import FastText

from .helpers import get_random_data


def test_build_args_keeps_manual_bucket_for_autotune():
    """Used to zero it, so autotune's n-gram trials hashed modulo 0."""
    # unsupervised_default lists every arg; the overrides make it supervised.
    base = dict(
        FastText.unsupervised_default,
        input="x",
        model="supervised",
        minn=0,
        maxn=0,
        autotuneValidationFile="valid.txt",
    )
    assert FastText._build_args(dict(base, bucket=1000), {"bucket"}).bucket == 1000
    assert FastText._build_args(dict(base), set()).bucket == 0  # default: unchanged


def test_autotune_with_manual_bucket_zero(tmp_path):
    """Used to terminate: trials sampled n-grams that bucket=0 can't hold."""
    path = tmp_path / "train.txt"
    path.write_text("".join(f"__label__{line}\n" for line in get_random_data(3000)))
    # thread=12: thread <= 10 leaves the input matrix partly uninitialized.
    model = fasttext.train_supervised(
        str(path),
        autotuneValidationFile=str(path),
        autotuneDuration=3,  # enough for several trials
        bucket=0,
        minn=2,  # a manual minn must not turn on subwords either
        lr=0.1,  # a sampled lr can diverge (NaN) in the final retrain
        thread=12,
        verbose=0,
    )
    args = model.f.getArgs()
    assert (args.bucket, args.wordNgrams, args.maxn) == (0, 1, 0)


def test_unsupervised_autotune_with_manual_bucket_zero(tmp_path):
    """Used to terminate: trial 1 kept the default subwords (CLI-only path)."""
    path = tmp_path / "train.txt"
    path.write_text("".join(f"{line}\n" for line in get_random_data(3000)))
    # train_unsupervised() takes no autotune args; the CLI reaches this.
    args = dict(
        FastText.unsupervised_default,
        input=str(path),
        autotuneValidationFile=str(path),
        autotuneDuration=3,
        bucket=0,
        lr=0.05,  # a sampled lr can diverge (NaN) in the final retrain
        thread=12,  # thread <= 10 leaves the input matrix partly uninitialized
        verbose=0,
    )
    a = FastText._build_args(args, {"bucket", "lr"})
    model = FastText._FastText(args=a)
    fasttext_pybind.train(model.f, a)
    got = model.f.getArgs()
    assert (got.bucket, got.wordNgrams, got.maxn) == (0, 1, 0)
