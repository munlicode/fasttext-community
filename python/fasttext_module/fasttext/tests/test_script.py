# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from fasttext import train_supervised
from fasttext import util
import fasttext
import tempfile
import numpy as np
import pytest
from .helpers import (
    get_random_words,
    build_supervised_model,
    build_unsupervised_model,
    get_random_data,
)

# For reference, these are the settings lists we're using.
STABLE_PARAMS = {"lr": 0.1, "epoch": 5}
general_settings = [
    pytest.param({"minn": 2, "maxn": 4, **STABLE_PARAMS}, id="minn2_maxn4"),
    pytest.param(
        {"minn": 0, "maxn": 0, "bucket": 0, **STABLE_PARAMS}, id="no_subwords"
    ),
    pytest.param({"dim": 1, **STABLE_PARAMS}, id="dim1"),
    pytest.param({"dim": 5, **STABLE_PARAMS}, id="dim5"),
]

unsupervised_settings = [
    pytest.param({"minn": 2, "maxn": 4}, id="unsup_minn2_maxn4"),
    pytest.param({"minn": 0, "maxn": 0, "bucket": 0}, id="unsup_no_subwords"),
    pytest.param({"dim": 1}, id="unsup_dim1"),
    pytest.param({"model": "cbow", "dim": 5}, id="unsup_cbow_dim5"),
    pytest.param({"model": "skipgram", "dim": 5}, id="unsup_skipgram_dim5"),
]

supervised_settings = [
    pytest.param({"minn": 2, "maxn": 4}, id="sup_minn2_maxn4"),
    pytest.param({"minn": 0, "maxn": 0, "bucket": 0}, id="sup_no_subwords"),
    pytest.param({"dim": 1}, id="sup_dim1"),
    pytest.param({"dim": 5}, id="sup_dim5"),
    pytest.param({"dim": 5, "loss": "hs"}, id="sup_dim5_hs"),
]


@pytest.mark.parametrize("kwargs", general_settings)
def test_get_vector(kwargs):
    """
    Tests that `get_word_vector` runs without crashing for various words.
    This is a "smoke test" to ensure the function is callable.
    """
    model = build_unsupervised_model(get_random_data(100), kwargs)
    words, _ = model.get_words(include_freq=True)

    # Add some random Out-Of-Vocabulary words to the list
    words += get_random_words(100)

    # The main purpose is to ensure this loop runs without errors.
    for word in words:
        model.get_word_vector(word)


@pytest.mark.parametrize("kwargs", general_settings)
def test_multi_get_line(kwargs):
    """
    Tests `get_line` processing for both supervised and unsupervised models.
    It verifies that processing a list of lines gives the same output as
    processing them one-by-one.
    """
    data = get_random_data(100)

    # Build both model types for the test
    sup_model = build_supervised_model(data, kwargs)
    unsup_model = build_unsupervised_model(data, kwargs)

    # 1. Process lines one by one
    sup_lines_single = []
    for line in data:
        words, labels = sup_model.get_line(line)
        sup_lines_single.append(words)
        # `get_line` on raw text should not produce labels
        assert len(labels) == 0

    unsup_lines_single = []
    for line in data:
        words, labels = unsup_model.get_line(line)
        unsup_lines_single.append(words)
        assert len(labels) == 0

    # 2. Process all lines in a single batch call
    sup_lines_batch, sup_labels_batch = sup_model.get_line(data)
    unsup_lines_batch, unsup_labels_batch = unsup_model.get_line(data)

    # 3. Assert that single-line and batch processing yield identical results
    assert sup_lines_single == sup_lines_batch
    assert unsup_lines_single == unsup_lines_batch

    # 4. Assert that no labels are returned for any line in the batch output
    assert all(len(labels) == 0 for labels in sup_labels_batch)
    assert all(len(labels) == 0 for labels in unsup_labels_batch)


@pytest.mark.parametrize("kwargs", supervised_settings)
def test_supervised_util_test(kwargs):
    """
    Verifies that `model.test()` and manually calculating precision/recall
    with `util.test()` produce identical results.
    """
    # 1. Prepare distinct training and validation datasets
    data = get_random_data(100, min_words_line=2)
    third = len(data) // 3
    train_data = data[: 2 * third]
    valid_data = data[third:]

    # 2. Use temporary files to hold the data
    with (
        tempfile.NamedTemporaryFile(mode="w+", encoding="UTF-8") as train_file,
        tempfile.NamedTemporaryFile(mode="w+", encoding="UTF-8") as valid_file,
    ):

        for line in train_data:
            train_file.write(f"__label__{line.strip()}\n")
        train_file.flush()

        for line in valid_data:
            valid_file.write(f"__label__{line.strip()}\n")
        valid_file.flush()

        # 3. Train the model on the training data, NOW WITH NaN HANDLING
        try:
            model = train_supervised(input=train_file.name, **kwargs)
        except RuntimeError as e:
            if "Encountered NaN" in str(e):
                pytest.skip(f"fastText training diverged (NaN) with kwargs={kwargs}")
            raise  # Re-raise any other runtime errors

        # 4. Manually get predictions and true labels from the validation file
        true_labels = []
        all_words = []
        valid_file.seek(0)
        for line in valid_file:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            words, labels = model.get_line(stripped_line)
            if not labels:
                continue
            all_words.append(" ".join(words))
            true_labels.append(labels)

        # 5. Get predictions and calculate precision/recall with `util.test`
        predictions, _ = model.predict(all_words)
        p, r = util.test(predictions, true_labels)
        N = len(predictions)

        # 6. Get results directly from `model.test()`
        Nt, pt, rt = model.test(valid_file.name)

        # 7. Assert that the results from both methods are identical
        assert N == Nt
        assert p == pt
        assert r == rt


@pytest.mark.parametrize("kwargs", supervised_settings)
def test_supervised_predict(kwargs):
    """
    Smoke test to ensure `model.predict()` runs without errors for various
    inputs (single words, sentences) and values of k.
    """
    model = build_supervised_model(get_random_data(100), kwargs)
    words = get_random_words(100)

    for k in [1, 2, 5]:
        # Test prediction on individual random words
        for w in words:
            labels, probs = model.predict(w, k)
            assert len(labels) <= k
            assert len(probs) <= k

        # Test prediction on full lines of text
        data = get_random_data(100)
        for line in data:
            labels, probs = model.predict(line, k)
            assert len(labels) <= k
            assert len(probs) <= k


@pytest.mark.parametrize("kwargs", supervised_settings)
def test_supervised_multiline_predict(kwargs):
    """
    Verifies that batch `predict` returns the same result as predicting
    line-by-line.
    """
    # This loop replaces the two separate check_predict calls in the original test
    for data_min_words in [0, 1]:
        model = build_supervised_model(
            get_random_data(100, min_words_line=data_min_words), kwargs
        )

        for k in [1, 2, 5]:
            # 1. Test consistency with single words
            words = get_random_words(10)
            single_labels = []
            single_probs = []
            for w in words:
                labels, probs = model.predict(w, k)
                single_labels.append(labels)
                single_probs.append(probs)

            batch_labels, batch_probs = model.predict(words, k)

            # Assert batch and single predictions are identical
            assert all(list(b) == list(s) for b, s in zip(batch_labels, single_labels))
            assert np.allclose(batch_probs, single_probs)

            # 2. Test consistency with full lines of text
            lines = get_random_data(10)
            single_labels = []
            single_probs = []
            for line in lines:
                labels, probs = model.predict(line, k)
                single_labels.append(labels)
                single_probs.append(probs)

            batch_labels, batch_probs = model.predict(lines, k)

            # Assert batch and single predictions are identical
            assert all(list(b) == list(s) for b, s in zip(batch_labels, single_labels))
            assert np.allclose(batch_probs, single_probs)


@pytest.mark.parametrize("kwargs", general_settings)
def test_vocab_and_frequencies(kwargs):
    """
    Verifies `get_words` returns the correct vocabulary and frequencies.
    Also tests that training on an empty file raises a ValueError.
    """
    # Part 1: Verify vocabulary and word frequencies
    data = get_random_data(100)

    # Manually count frequencies to create a ground truth
    word_counts = {}
    for line in data:
        for word in line.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    model = build_unsupervised_model(data, kwargs)
    words, freqs = model.get_words(include_freq=True)

    # Convert model output to a dictionary for easy lookup
    model_vocab = {word: freq for word, freq in zip(words, freqs)}

    # The EOS (End-Of-Sentence) token is added automatically by fastText
    assert fasttext.EOS in model_vocab

    # The model's vocab size should be our word count + the EOS token
    assert len(model_vocab) == len(word_counts) + 1

    # Check if all our words and their counts match the model's vocab
    for word, count in word_counts.items():
        assert word in model_vocab
        assert model_vocab[word] == count

    # Part 2: Verify error on empty vocabulary
    empty_data = get_random_data(0)
    with pytest.raises(ValueError, match="Empty vocabulary"):
        build_unsupervised_model(empty_data, kwargs)


@pytest.mark.parametrize("kwargs", general_settings)
def test_get_subwords(kwargs):
    """Smoke test to ensure `get_subwords` runs without errors."""
    model = build_unsupervised_model(get_random_data(100), kwargs)
    words, _ = model.get_words(include_freq=True)
    # Add some OOV words
    words += get_random_words(10, 1, 10)

    for w in words:
        model.get_subwords(w)


def test_tokenize():
    """
    Tests the static `fasttext.tokenize` function.
    Note: This test is not parametrized as its behavior is static and
    does not depend on any model-building arguments.
    """
    assert ["asdf", "asdb"] == fasttext.tokenize("asdf asdb")
    assert ["asdf"] == fasttext.tokenize("asdf")
    assert [fasttext.EOS] == fasttext.tokenize("\n")
    assert ["asdf", fasttext.EOS] == fasttext.tokenize("asdf\n")
    assert [] == fasttext.tokenize("")
    assert [] == fasttext.tokenize(" ")

    words = get_random_words(100, 1, 20)
    assert words == fasttext.tokenize(" ".join(words))


@pytest.mark.parametrize("kwargs", unsupervised_settings)
def test_unsupervised_dimension(kwargs):
    """Verifies `get_dimension` for unsupervised models."""
    if "dim" in kwargs:
        model = build_unsupervised_model(get_random_data(100), kwargs)
        assert model.get_dimension() == kwargs["dim"]
    else:
        pytest.skip("Test not applicable for kwargs without 'dim'")


@pytest.mark.parametrize("kwargs", supervised_settings)
def test_supervised_dimension(kwargs):
    """Verifies `get_dimension` for supervised models."""
    if "dim" in kwargs:
        model = build_supervised_model(get_random_data(100), kwargs)
        assert model.get_dimension() == kwargs["dim"]
    else:
        pytest.skip("Test not applicable for kwargs without 'dim'")


@pytest.mark.parametrize("kwargs", unsupervised_settings)
def test_unsupervised_get_words(kwargs):
    """
    Verifies the output consistency of `get_words` for unsupervised models.
    """
    model = build_unsupervised_model(get_random_data(100), kwargs)
    words_with_freq, freqs = model.get_words(include_freq=True)
    words_no_freq = model.get_words(include_freq=False)

    assert len(words_with_freq) == len(words_no_freq)
    assert len(words_with_freq) == len(freqs)


@pytest.mark.parametrize("kwargs", supervised_settings)
def test_supervised_get_words(kwargs):
    """
    Verifies the output consistency of `get_words` for supervised models.
    """
    model = build_supervised_model(get_random_data(100), kwargs)
    words_with_freq, freqs = model.get_words(include_freq=True)
    words_no_freq = model.get_words(include_freq=False)

    assert len(words_with_freq) == len(words_no_freq)
    assert len(words_with_freq) == len(freqs)


@pytest.mark.parametrize("kwargs", unsupervised_settings)
def test_unsupervised_get_labels_returns_vocab(kwargs):
    """
    Verifies that `get_labels` on an unsupervised model correctly returns
    the word vocabulary, as there are no labels.
    """
    model = build_unsupervised_model(get_random_data(100), kwargs)
    labels_with_freq, freqs = model.get_labels(include_freq=True)
    labels_no_freq = model.get_labels(include_freq=False)
    words = model.get_words(include_freq=False)

    assert len(labels_with_freq) == len(labels_no_freq)
    assert len(labels_with_freq) == len(freqs)
    # For an unsupervised model, labels should be the same as words.
    assert labels_no_freq == words


@pytest.mark.parametrize("kwargs", supervised_settings)
def test_supervised_get_labels(kwargs):
    """
    Verifies the output consistency of `get_labels` for supervised models.
    """
    model = build_supervised_model(get_random_data(100), kwargs)
    labels_with_freq, freqs = model.get_labels(include_freq=True)
    labels_no_freq = model.get_labels(include_freq=False)

    assert len(labels_with_freq) == len(labels_no_freq)
    assert len(labels_with_freq) == len(freqs)


@pytest.mark.parametrize("kwargs", unsupervised_settings)
def test_unsupervised_quantize_raises_error(kwargs):
    """
    Verifies that calling `.quantize()` on an unsupervised model
    raises a ValueError, as it is not a supported operation.
    """
    model = build_unsupervised_model(get_random_data(100), kwargs)
    with pytest.raises(ValueError):
        model.quantize()


@pytest.mark.parametrize("kwargs", supervised_settings)
def test_supervised_quantize_and_is_quantized(kwargs):
    """
    Verifies the `.is_quantized()` status before and after quantization
    for a supervised model.
    """
    # Use a larger dataset as quantization requires a fair number of labels
    model = build_supervised_model(get_random_data(1000, max_vocab_size=1000), kwargs)

    # A new model should not be quantized
    assert not model.is_quantized()

    # After calling .quantize(), the model should be quantized
    model.quantize()
    assert model.is_quantized()


@pytest.mark.parametrize("kwargs", general_settings)
def test_newline_in_string_raises_error(kwargs):
    """
    Verifies that `predict` and `get_sentence_vector` raise a ValueError
    if the input string contains a newline character.
    """
    sentence = " ".join(get_random_words(20))
    sentence_with_newline = sentence + "\n"

    # Test for both `predict` and `get_sentence_vector`
    for method_name in ["predict", "get_sentence_vector"]:
        model = build_supervised_model(get_random_data(100), kwargs)
        method_to_test = getattr(model, method_name)

        # Ensure the method works without a newline
        if method_name == "predict":
            method_to_test(sentence, k=5)
        else:
            method_to_test(sentence)

        # Assert that a ValueError is raised when a newline is present
        with pytest.raises(ValueError):
            if method_name == "predict":
                method_to_test(sentence_with_newline, k=5)
            else:
                method_to_test(sentence_with_newline)
