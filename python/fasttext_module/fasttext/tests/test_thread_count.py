# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""thread < 1 must raise ValueError, not crash (thread=0 divided by zero)."""

import pytest

from fasttext.FastText import unsupervised_default

from .helpers import build_supervised_model, build_unsupervised_model, get_random_data

CASES = {
    "supervised": (build_supervised_model, {"thread": 0}),
    "unsupervised": (build_unsupervised_model, {"thread": -2}),
    "autotune": (build_supervised_model, {"thread": 0, "autotuneDuration": 5}),
}


def _write(path, data):
    path.write_text("".join(f"__label__{line}\n" for line in data))
    return str(path)


@pytest.mark.parametrize("case", CASES)
def test_train_rejects_thread_below_one(tmp_path, case):
    build, kwargs = CASES[case]
    data = get_random_data(100)
    if case == "autotune":
        kwargs = {**kwargs, "autotuneValidationFile": _write(tmp_path / "v.txt", data)}
    with pytest.raises(ValueError, match="thread must be >= 1"):
        build(data, {"verbose": 0, **kwargs})


def test_quantize_retrain_rejects_thread_below_one(tmp_path):
    data = get_random_data(3000, max_vocab_size=600)
    model = build_supervised_model(data, {"dim": 16, "verbose": 0})
    words = model.get_words()
    # thread=0 is falsy: quantize() swaps in the model's own thread
    with pytest.raises(ValueError, match="thread must be >= 1"):
        model.quantize(
            input=_write(tmp_path / "t.txt", data), retrain=True, cutoff=300, thread=-1
        )
    assert not model.is_quantized() and model.get_words() == words


def test_default_thread_at_least_one():
    assert unsupervised_default["thread"] >= 1
