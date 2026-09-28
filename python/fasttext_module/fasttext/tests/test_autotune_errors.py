# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""Autotune errors must raise, not terminate the process."""

import pytest

from .helpers import build_supervised_model, get_random_data


def test_autotune_error_raises(tmp_path):
    data = get_random_data(3000, max_vocab_size=600)
    valid = tmp_path / "valid.txt"
    valid.write_text("".join(f"__label__{line}\n" for line in data))
    kwargs = {
        # thread=12: thread <= 10 leaves the input matrix partly uninitialized.
        "thread": 12,
        "verbose": 0,
        "autotuneValidationFile": str(valid),
        "autotuneMetric": "f1:__label__missing",  # fails after the first trial
        "autotuneDuration": 60,  # long enough for slow runners to reach it
    }
    with pytest.raises(RuntimeError, match="Unknown autotune metric label"):
        build_supervised_model(data, kwargs)
