# SPDX-FileContributor: Arthit Suriyawongkul
# SPDX-FileCopyrightText: 2026-present, fasttext-community
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT

"""Vector index and size errors must raise, not access out of bounds."""

import pytest

import fasttext_pybind

from .helpers import build_supervised_model, get_random_data


def _model(data=None):
    data = data or get_random_data(300, max_vocab_size=100)
    return build_supervised_model(data, {"dim": 16, "verbose": 0})


def _shrink_output_dim(model):
    output = model.get_output_matrix()[:, :8].copy()  # input dim is 16
    model.set_matrices(model.get_input_matrix(), output)


@pytest.mark.parametrize("ind", [-1, 10**9])
def test_get_input_vector_out_of_range_raises(ind):
    with pytest.raises(ValueError, match="out of range"):
        _model().get_input_vector(ind)


def test_get_word_vector_size_mismatch_raises():
    model = _model()
    vec = fasttext_pybind.Vector(model.get_dimension() - 1)
    with pytest.raises(ValueError, match="size mismatch"):
        model.f.getWordVector(vec, model.words[1])


def test_predict_dim_mismatch_raises():
    model = _model()
    _shrink_output_dim(model)
    with pytest.raises(ValueError, match="size mismatch"):
        model.predict(model.words[1])


def test_training_thread_error_raises(tmp_path):
    """Used to call std::terminate: exception escaped a training thread."""
    data = get_random_data(3000, max_vocab_size=600)  # cutoff needs >= 256 rows
    model = _model(data)
    _shrink_output_dim(model)
    train_txt = tmp_path / "train.txt"
    train_txt.write_text("".join(f"__label__{line}\n" for line in data))
    with pytest.raises(ValueError, match="size mismatch"):
        model.quantize(input=str(train_txt), cutoff=300, retrain=True)
