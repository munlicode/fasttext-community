# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""Meter curves must work with NumPy 2 (np.array copy=False raises there)."""

import numpy as np

from .helpers import build_supervised_model, get_random_data


def test_meter_returns_arrays(tmp_path):
    data = get_random_data(300, max_vocab_size=100)
    # thread=12: thread <= 10 leaves the input matrix partly uninitialized.
    model = build_supervised_model(data, {"thread": 12, "dim": 16, "verbose": 0})
    path = tmp_path / "test.txt"
    path.write_text("".join(f"__label__{line}\n" for line in data))
    meter = model.get_meter(str(path))
    label = model.labels[0]
    for x, y in (
        meter.score_vs_true(label),
        meter.precision_recall_curve(),
        meter.precision_recall_curve(label),
    ):
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert len(x) == len(y) > 0
