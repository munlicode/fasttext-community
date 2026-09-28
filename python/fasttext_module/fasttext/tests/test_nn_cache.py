"""Cached word data must be rebuilt when the model changes."""

import pytest

import fasttext
import fasttext_pybind

from .helpers import build_supervised_model, get_random_data


def _model():
    # thread=12: thread <= 10 leaves the input matrix partly uninitialized.
    data = get_random_data(3000, max_vocab_size=600)
    return build_supervised_model(data, {"thread": 12, "dim": 16, "verbose": 0})


def _filled_model():
    model = _model()
    model.get_nearest_neighbors(model.words[1])  # fills the caches
    return model


def _reload(model, tmp_path):
    path = str(tmp_path / "model.bin")
    model.save_model(path)
    return fasttext.load_model(path)


def _assert_same_nn(model, expected):
    word = expected.words[1]  # words[0] is the end-of-sentence token
    assert model.get_nearest_neighbors(word) == expected.get_nearest_neighbors(word)


# cutoff=300 also prunes the dictionary (quantize needs >= 256 rows).
@pytest.mark.parametrize("cutoff", [0, 300])
def test_quantize_resets_caches(tmp_path, cutoff):
    model = _filled_model()
    model.quantize(cutoff=cutoff)
    expected = _reload(model, tmp_path)
    assert model.words == expected.words
    _assert_same_nn(model, expected)


def test_load_model_resets_cache(tmp_path):
    other = _reload(_model(), tmp_path)
    model = _filled_model()
    model.f.loadModel(str(tmp_path / "model.bin"))
    _assert_same_nn(model, other)


def test_train_resets_cache(tmp_path):
    model = _filled_model()
    train_txt = tmp_path / "train.txt"
    data = get_random_data(3000)
    train_txt.write_text("".join(f"__label__{line}\n" for line in data))
    args = model.f.getArgs()
    args.input = str(train_txt)
    fasttext_pybind.train(model.f, args)
    _assert_same_nn(model, _reload(model, tmp_path))
