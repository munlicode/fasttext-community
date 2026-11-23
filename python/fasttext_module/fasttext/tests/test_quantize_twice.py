# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""quantize() must raise on a quantized model, and leave the model unchanged
when it fails."""

import pytest

import fasttext

from .helpers import build_supervised_model, get_random_data


def _model():
    data = get_random_data(3000, max_vocab_size=600)
    return build_supervised_model(data, {"dim": 16, "verbose": 0})


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


@pytest.mark.parametrize("kwargs", [{"qout": True}, {"cutoff": 100}])
def test_failed_quantize_leaves_model_unchanged(tmp_path, kwargs):
    """Used to prune the dict, or quantize input only (retry crashed)."""
    data = get_random_data(3000, max_vocab_size=600)
    path = tmp_path / "train.txt"
    # 3 labels: too few output rows for qout.
    path.write_text("".join(f"__label__{i % 3} {x}\n" for i, x in enumerate(data)))
    model = fasttext.train_supervised(str(path), dim=16, verbose=0)
    nwords = len(model.get_words())
    with pytest.raises(ValueError, match="too small"):
        model.quantize(**kwargs)
    assert not model.is_quantized()
    assert len(model.get_words()) == nwords
    model.quantize()


def test_set_matrices_clears_quantized():
    model = _model()
    matrices = model.get_input_matrix(), model.get_output_matrix()
    model.quantize()
    model.set_matrices(*matrices)
    assert not model.is_quantized()
    model.quantize()
