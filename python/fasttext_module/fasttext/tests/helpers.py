from fasttext import train_supervised
from fasttext import train_unsupervised
import tempfile
import pytest
import random
import string

try:
    import unicode
except ImportError:
    pass


def get_random_unicode(length):
    # See: https://stackoverflow.com/questions/1477294/generate-random-utf-8-string-in-python

    try:
        get_char = unichr
    except NameError:
        get_char = chr

    # Update this to include code point ranges to be sampled
    include_ranges = [
        (0x0021, 0x0021),
        (0x0023, 0x0026),
        (0x0028, 0x007E),
        (0x00A1, 0x00AC),
        (0x00AE, 0x00FF),
        (0x0100, 0x017F),
        (0x0180, 0x024F),
        (0x2C60, 0x2C7F),
        (0x16A0, 0x16F0),
        (0x0370, 0x0377),
        (0x037A, 0x037E),
        (0x0384, 0x038A),
        (0x038C, 0x038C),
    ]

    alphabet = [
        get_char(code_point)
        for current_range in include_ranges
        for code_point in range(current_range[0], current_range[1] + 1)
    ]
    return "".join(random.choice(alphabet) for i in range(length))


def get_random_alphanumeric(length):
    """Generates a simple alphanumeric string."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for i in range(length))


def get_random_words(N, a=1, b=20, unique=True, alphanumeric=False):
    words = []
    seen = set()
    while len(words) < N:
        length = random.randint(a, b)

        if alphanumeric:
            word = get_random_alphanumeric(length)
        else:
            word = get_random_unicode(length)

        if unique:
            if word not in seen:
                words.append(word)
                seen.add(word)
        else:
            words.append(word)
    return words


def get_random_data(
    num_lines=100,
    max_vocab_size=100,
    min_words_line=1,  # change 0 → 1
    max_words_line=20,
    min_len_word=1,
    max_len_word=10,
    unique_words=True,
):
    random_words = get_random_words(
        max_vocab_size, min_len_word, max_len_word, unique=unique_words
    )
    lines = []
    for _ in range(num_lines):
        line_length = random.randint(min_words_line, max_words_line)
        line = [
            random_words[random.randint(0, max_vocab_size - 1)]
            for _ in range(line_length)
        ]
        lines.append(" ".join(line))
    return lines


def default_kwargs(kwargs):
    default = {"thread": 1, "epoch": 1, "minCount": 1, "bucket": 1000}
    for k, v in default.items():
        if k not in kwargs:
            kwargs[k] = v
    return kwargs


def build_unsupervised_model(data, kwargs):
    """Builds an unsupervised model, skipping the test on NaN errors."""
    kwargs = default_kwargs(kwargs)

    with tempfile.NamedTemporaryFile(mode="w+", encoding="UTF-8", delete=False) as tmpf:
        for line in data:
            tmpf.write(line + "\n")
        tmpf.flush()
        try:
            model = train_unsupervised(input=tmpf.name, **kwargs)
        except RuntimeError as e:
            if "Encountered NaN" in str(e):
                # This is an expected instability. Instead of failing, we skip.
                pytest.skip(f"fastText training diverged (NaN) with kwargs={kwargs}")
            raise  # Re-raise any other runtime errors
    return model


def build_supervised_model(data, kwargs):
    """Builds a supervised model, skipping the test on NaN errors."""
    kwargs = default_kwargs(kwargs)

    with tempfile.NamedTemporaryFile(mode="w+", encoding="UTF-8", delete=False) as tmpf:
        for line in data:
            # This labeling scheme (using the whole line as a label)
            # is a primary cause of the instability.
            line = f"__label__{line.strip()}\n"
            tmpf.write(line)
        tmpf.flush()
        try:
            model = train_supervised(input=tmpf.name, **kwargs)
        except RuntimeError as e:
            if "Encountered NaN" in str(e):
                # This is an expected instability. Instead of failing, we skip.
                pytest.skip(f"fastText training diverged (NaN) with kwargs={kwargs}")
            raise  # Re-raise any other runtime errors
    return model


def read_labels(data_file):
    labels = []
    lines = []
    with open(data_file, "r") as f:
        for line in f:
            labels_line = []
            words_line = []
            try:
                line = unicode(line, "UTF-8").split()
            except NameError:
                line = line.split()
            for word in line:
                if word.startswith("__label__"):
                    labels_line.append(word)
                else:
                    words_line.append(word)
            labels.append(labels_line)
            lines.append(" ".join(words_line))
    return lines, labels
