#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: The purpose of this file is not to accumulate all useful utility
# functions. This file should contain very commonly used and requested functions
# (such as test). If you think you have a function at that level, please create
# an issue and we will happily review your suggestion. This file is also not supposed
# to pull in dependencies outside of numpy/scipy without very good reasons. For
# example, this file should not use sklearn and matplotlib to produce a t-sne
# plot of word embeddings or such.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from typing import Literal, Optional, Union, overload
import shutil
import os
import gzip
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from .constants import (
    OFFICIAL_MODELS_LINKS,
    VECTORS_WIKI_LANG_IDS,
    VECTORS_CRAWL_LANG_IDS,
    VECTORS_ALIGNED_LANG_IDS,
    IF_EXISTS_TYPE,
    LangIdType,
    TextClassificationType,
    VectorsWikiType,
    VectorsCrawlType,
    VectorsEnglishType,
    ModelCategory,
)


# TODO: Add example on reproducing model.test with util.test and model.get_line
def test(predictions, labels, k=1):
    """
    Return precision and recall modeled after fasttext's test
    """
    precision = 0.0
    nexamples = 0
    nlabels = 0
    for prediction, labels in zip(predictions, labels):
        for p in prediction:
            if p in labels:
                precision += 1
        nexamples += 1
        nlabels += len(labels)
    return (precision / (k * nexamples), precision / nlabels)


def find_nearest_neighbor(query, vectors, ban_set, cossims=None):
    """
    query is a 1d numpy array corresponding to the vector to which you want to
    find the closest vector
    vectors is a 2d numpy array corresponding to the vectors you want to consider
    ban_set is a set of indicies within vectors you want to ignore for nearest match
    cossims is a 1d numpy array of size len(vectors), which can be passed for efficiency

    returns the index of the closest match to query within vectors

    """
    if cossims is None:
        cossims = np.matmul(vectors, query, out=cossims)
    else:
        np.matmul(vectors, query, out=cossims)
    rank = len(cossims) - 1
    result_i = np.argpartition(cossims, rank)[rank]
    while result_i in ban_set:
        rank -= 1
        result_i = np.argpartition(cossims, rank)[rank]
    return result_i


def _reduce_matrix(X_orig, dim, eigv):
    """
    Reduces the dimension of a (m × n)   matrix `X_orig` to
                          to a (m × dim) matrix `X_reduced`
    It uses only the first 100000 rows of `X_orig` to do the mapping.
    Matrix types are all `np.float32` in order to avoid unncessary copies.
    """
    if eigv is None:
        mapping_size = 100000
        X = X_orig[:mapping_size]
        X = X - X.mean(axis=0, dtype=np.float32)
        C = np.divide(np.matmul(X.T, X), X.shape[0] - 1, dtype=np.float32)
        _, U = np.linalg.eig(C)
        eigv = U[:, :dim]

    X_reduced = np.matmul(X_orig, eigv)

    return (X_reduced, eigv)


def reduce_model(ft_model, target_dim):
    """
    ft_model is an instance of `_FastText` class
    This function computes the PCA of the input and the output matrices
    and sets the reduced ones.
    """
    inp_reduced, proj = _reduce_matrix(ft_model.get_input_matrix(), target_dim, None)
    out_reduced, _ = _reduce_matrix(ft_model.get_output_matrix(), target_dim, proj)

    ft_model.set_matrices(inp_reduced, out_reduced)

    return ft_model


def get_model_url(
    category: str, lang_id: Optional[str] = None, model_type: Optional[str] = None
) -> str:
    """Navigates the dictionary to find the correct model URL."""
    if category not in OFFICIAL_MODELS_LINKS:
        raise ValueError(
            f"Unknown category '{category}'. Available: {list(OFFICIAL_MODELS_LINKS.keys())}"
        )

    entry = OFFICIAL_MODELS_LINKS[category]

    if isinstance(entry, str):  # Direct URL or a simple template
        if "{lang_id}" in entry:
            if not lang_id:
                raise ValueError(f"Category '{category}' requires a 'lang_id'.")
            return entry.format(lang_id=lang_id)
        return entry

    elif isinstance(entry, list):  # List of full URLs
        if not model_type:
            filenames = [os.path.basename(urlparse(u).path) for u in entry]
            raise ValueError(
                f"Category '{category}' has multiple models. Specify 'model_type' from: {filenames}"
            )

        for url in entry:
            if url.endswith(model_type):
                return url
        raise ValueError(
            f"Model type '{model_type}' not found in category '{category}'."
        )

    elif isinstance(entry, dict):  # Dictionary of templates
        if not model_type:
            raise ValueError(
                f"Category '{category}' requires a 'model_type'. Available: {list(entry.keys())}"
            )
        if model_type not in entry:
            raise ValueError(
                f"Model type '{model_type}' not found for category '{category}'."
            )

        template = entry[model_type]
        if "{lang_id}" in template:
            if not lang_id:
                raise ValueError(f"This model type requires a 'lang_id'.")
            return template.format(lang_id=lang_id)
        return template

    raise TypeError(f"Unsupported entry type for category '{category}'.")


@overload
def download_model(
    category: Literal["vectors_english"],
    *,
    model_type: VectorsEnglishType,
    save_dir: str = ".",
    if_exists: IF_EXISTS_TYPE = "ignore",
    auth: Optional[Union[tuple, str]] = None,
) -> Optional[str]: ...
@overload
def download_model(
    category: Literal["vectors_crawl"],
    *,
    lang_id: VECTORS_CRAWL_LANG_IDS,
    model_type: VectorsCrawlType,
    save_dir: str = ".",
    if_exists: IF_EXISTS_TYPE = "ignore",
    auth: Optional[Union[tuple, str]] = None,
) -> Optional[str]: ...
@overload
def download_model(
    category: Literal["vectors_wiki"],
    *,
    lang_id: VECTORS_WIKI_LANG_IDS,
    model_type: VectorsWikiType,
    save_dir: str = ".",
    if_exists: IF_EXISTS_TYPE = "ignore",
    auth: Optional[Union[tuple, str]] = None,
) -> Optional[str]: ...
@overload
def download_model(
    category: Literal["language_identification"],
    *,
    model_type: LangIdType,
    save_dir: str = ".",
    if_exists: IF_EXISTS_TYPE = "ignore",
    auth: Optional[Union[tuple, str]] = None,
) -> Optional[str]: ...
@overload
def download_model(
    category: Literal["text_classification"],
    *,
    model_type: TextClassificationType,
    save_dir: str = ".",
    if_exists: IF_EXISTS_TYPE = "ignore",
    auth: Optional[Union[tuple, str]] = None,
) -> Optional[str]: ...
@overload
def download_model(
    category: Literal["vectors_aligned"],
    *,
    lang_id: VECTORS_ALIGNED_LANG_IDS,
    save_dir: str = ".",
    if_exists: IF_EXISTS_TYPE = "ignore",
    auth: Optional[Union[tuple, str]] = None,
) -> Optional[str]: ...


@overload
def download_model(
    category: Literal["datasets"],
    *,
    model_type: Literal["default"],
    save_dir: str = ".",
    if_exists: IF_EXISTS_TYPE = "ignore",
    auth: Optional[Union[tuple, str]] = None,
) -> Optional[str]: ...


def download_custom_model(
    url: str,
    save_dir: str = ".",
    output_filename: Optional[str] = None,
    if_exists: IF_EXISTS_TYPE = "ignore",
    auth: Optional[Union[tuple, str]] = None,
) -> Optional[str]:
    """Downloads a model from a URL, with progress tracking and auto-decompression.

    This function provides a robust way to download large files, such as machine
    learning models. It streams the download to avoid high memory usage, displays
    a `tqdm` progress bar, and handles `.gz` files by automatically decompressing
    them after download. The download is performed atomically by using a `.part`
    file, which is renamed only upon successful completion.

    Args:
        url: The direct download URL for the model.
        save_dir: The directory where the model will be saved. Defaults to the
            current directory (".").
        output_filename: An optional name for the saved file. If None, the
            filename is inferred from the URL. Defaults to None.
        if_exists: The action to take if the destination file already exists.
            - "ignore": (Default) Skips the download and returns the file path.
            - "strict": Skips the download and returns None.
            - "overwrite": Downloads and replaces the existing file.
        auth: Authentication credentials. Can be one of:
            - A tuple of (username, password) for HTTP Basic Auth.
            - A string containing an API key for Bearer Token Auth.
            Defaults to None.

    Returns:
        The absolute path to the downloaded (and decompressed) model file on
        success, or None if the download fails or is skipped due to the
        `if_exists` policy.

    Example:
        >>> # Download the Language Identification Model
        >>> model_path = download_custom_model(
        ...     url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        ...     save_dir="./models",
        ... )
        >>> assert model_path is not None
    """
    os.makedirs(save_dir, exist_ok=True)

    download_filename = os.path.basename(urlparse(url).path)

    # Overwrite url basename with output filename
    if output_filename:
        download_filename = output_filename

    final_filename = (
        download_filename[:-3]
        if download_filename.endswith(".gz")
        else download_filename
    )
    download_path = os.path.join(save_dir, download_filename)
    final_path = os.path.join(save_dir, final_filename)

    if os.path.exists(final_path):
        if if_exists == "ignore":
            print(f"Model '{final_filename}' already exists. Skipping.")
            return final_path
        elif if_exists == "strict":
            print("File exists. Use --overwrite to download anyway.")
            return
        elif if_exists == "overwrite":
            pass

    headers = {}
    request_auth = None

    if isinstance(auth, tuple):
        # Let the requests library handle Basic Auth via its `auth` param
        request_auth = auth
    elif isinstance(auth, str):
        # Manually create the header for Bearer Token (API Key)
        headers["Authorization"] = f"Bearer {auth}"

    # Core download and decompression logic
    part_path = download_path + ".part"
    try:
        with requests.get(url, stream=True, headers=headers, auth=request_auth) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with (
                open(part_path, "wb") as f,
                tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=download_filename
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        os.rename(part_path, download_path)
    except Exception as e:
        print(f"Download failed: {e}")
        return None

    finally:
        # --- Cleanup Logic ---
        if os.path.exists(part_path):
            print("\nCleaning up partial download file...")
            os.remove(part_path)

    # --- Decompression Logic ---
    if download_filename.endswith(".gz"):
        print(f"Decompressing '{download_filename}'...")
        with gzip.open(download_path, "rb") as f_in, open(final_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(download_path)

    return final_path


def download_model(
    category: ModelCategory,
    lang_id: Optional[str] = None,
    model_type: Optional[str] = None,
    save_dir: str = ".",
    output_filename: Optional[str] = None,
    if_exists: IF_EXISTS_TYPE = "ignore",
    auth: Optional[Union[tuple, str]] = None,
) -> Optional[str]:
    """Downloads an official fastText model using a simplified, high-level interface.

    This function acts as a convenient wrapper around `download_custom_model`.
    It abstracts away the need to know the exact download URL. The user specifies
    the desired model by its category, language, and type, and this function
    constructs the appropriate URL before passing the download job to the
    low-level handler.

    Args:
        category: The category of the official model to download, e.g.,
            "vectors_crawl" or "language_identification".
        lang_id: The two-letter language code (e.g., "en", "fr", "es") required
            for language-specific models. Defaults to None.
        model_type: The specific format or version of the model, e.g., "bin",
            "vec", or "zip". Defaults to None.
        save_dir: The directory where the model will be saved. Defaults to ".".
        output_filename: An optional name for the saved file. If None, the
            name is inferred from the URL. Defaults to None.
        if_exists: The action to take if the destination file already exists.
            Can be "ignore", "strict", or "overwrite". Defaults to "ignore".
        auth: Authentication credentials. Can be one of:
            - A tuple of (username, password) for HTTP Basic Auth.
            - A string containing an API key for Bearer Token Auth.
            Defaults to None.

    Returns:
        The absolute path to the downloaded model file on success, or None if
        the model could not be found or the download failed.

    Example:
        >>> # Download the Language Identification Model
        >>> model_path = download_model(
        ...     category="language_identification",
        ...     model_type="lid.176.ftz",
        ...     save_dir="./models",
        ... )
        >>> assert model_path is not None
    """
    try:
        url = get_model_url(category, lang_id, model_type)
        print(f"Found URL for download: {url}")
        return download_custom_model(
            url,
            save_dir,
            output_filename,
            if_exists,
            auth,
        )
    except (ValueError, TypeError) as e:
        print(f"Error: Could not find model. {e}")
        return None
