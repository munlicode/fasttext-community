# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""quantize() must raise on a quantized model, and only on one."""

import pytest

from .helpers import build_supervised_model, get_random_data


def _model():
    # thread=12: thread <= 10 leaves the input matrix partly uninitialized.
    data = get_random_data(3000, max_vocab_size=600)
    return build_supervised_model(data, {"thread": 12, "dim": 16, "verbose": 0})


def test_quantize_twice_raises():
    model = _model()
    model.quantize()
    with pytest.raises(ValueError, match="already quantized"):
        model.quantize()


def test_load_model_clears_quantized(tmp_path):
    model = _model()
    path = str(tmp_path / "model.bin")
    model.save_model(path)
    model.quantize()
    model.f.loadModel(path)
    assert not model.is_quantized()
    model.quantize()


def test_set_matrices_clears_quantized():
    model = _model()
    matrices = model.get_input_matrix(), model.get_output_matrix()
    model.quantize()
    model.set_matrices(*matrices)
    assert not model.is_quantized()
    model.quantize()
