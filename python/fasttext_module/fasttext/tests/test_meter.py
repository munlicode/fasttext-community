# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""Meter curves must work with NumPy 2 (np.array copy=False raises there)."""

import numpy as np

from .helpers import build_supervised_model, get_random_data


def test_meter_returns_arrays(tmp_path):
    data = get_random_data(100)
    # thread=12: thread <= 10 leaves the input matrix partly uninitialized.
    model = build_supervised_model(data, {"thread": 12})
    path = tmp_path / "test.txt"
    path.write_text("".join(f"__label__{line}\n" for line in data))
    meter = model.get_meter(str(path))
    for pair in (meter.score_vs_true(model.labels[0]), meter.precision_recall_curve()):
        assert all(isinstance(a, np.ndarray) for a in pair)
