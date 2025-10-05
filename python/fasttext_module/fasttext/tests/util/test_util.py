import os
import gzip
import pytest
import requests
from unittest.mock import patch, MagicMock

from fasttext.util.util import download_custom_model, download_model

# ===============================================================
# == Tests for download_custom_model (Low-Level Downloader)
# ===============================================================


@patch("requests.get")
def test_download_custom_model_success(mock_get, tmp_path):
    """
    Tests a successful download of a standard file.
    """
    # --- Setup ---
    # Configure the mock to simulate a successful web request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": "10"}
    mock_response.iter_content.return_value = [b"model_data"]
    mock_get.return_value.__enter__.return_value = mock_response

    # --- Action ---
    file_path = download_custom_model(
        url="http://fake.url/model.bin", save_dir=str(tmp_path)
    )

    # --- Assertions ---
    expected_path = tmp_path / "model.bin"
    assert file_path == str(expected_path)
    assert expected_path.exists()
    assert expected_path.read_text() == "model_data"
    mock_get.assert_called_once()  # Ensure a network call was made


@patch("requests.get")
def test_download_with_decompression(mock_get, tmp_path):
    """
    Tests that a .gz file is downloaded, decompressed, and the original is cleaned up.
    """
    # --- Setup ---
    original_content = b"this is the model"
    compressed_content = gzip.compress(original_content)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": str(len(compressed_content))}
    mock_response.iter_content.return_value = [compressed_content]
    mock_get.return_value.__enter__.return_value = mock_response

    # --- Action ---
    final_path = download_custom_model(
        url="http://fake.url/model.bin.gz", save_dir=str(tmp_path)
    )

    # --- Assertions ---
    expected_path = tmp_path / "model.bin"
    gz_path = tmp_path / "model.bin.gz"

    assert final_path == str(expected_path)
    assert expected_path.exists()
    assert expected_path.read_bytes() == original_content
    assert (
        not gz_path.exists()
    ), "The .gz file should have been deleted after decompression"


@patch("requests.get")
def test_download_http_error(mock_get, tmp_path):
    """
    Tests that the function returns None on an HTTP error.
    """
    # --- Setup ---
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Not Found"
    )
    mock_get.return_value.__enter__.return_value = mock_response

    # --- Action ---
    result = download_custom_model(
        url="http://fake.url/not_found.bin", save_dir=str(tmp_path)
    )

    # --- Assertions ---
    assert result is None
    assert len(list(tmp_path.iterdir())) == 0, "No files should be created on failure"


@patch("requests.get")
def test_download_file_exists_ignore(mock_get, tmp_path):
    """
    Tests that the download is skipped if the file exists and if_exists is 'ignore'.
    """
    # --- Setup ---
    expected_path = tmp_path / "model.bin"
    expected_path.write_text("already exists")

    # --- Action ---
    result = download_custom_model(
        url="http://fake.url/model.bin", save_dir=str(tmp_path), if_exists="ignore"
    )

    # --- Assertions ---
    assert result == str(expected_path)
    mock_get.assert_not_called(), "requests.get should not be called if the file is ignored"


@patch("requests.get")
def test_authentication_forwarding(mock_get, mocker, tmp_path):
    """
    Tests that auth parameters are correctly passed to requests.get.
    """
    # --- Setup ---
    # We just need a basic successful response for the function to run
    mock_response = MagicMock(status_code=200, headers={"content-length": "0"})
    mock_response.iter_content.return_value = [b""]
    mock_get.return_value.__enter__.return_value = mock_response

    # --- Action & Assertions for Basic Auth ---
    basic_auth_creds = ("user", "pass")
    download_custom_model(
        url="http://fake.url/basic.bin", save_dir=str(tmp_path), auth=basic_auth_creds
    )
    # Check the keyword arguments of the call to requests.get
    call_args, call_kwargs = mock_get.call_args
    assert call_kwargs.get("auth") == basic_auth_creds
    assert "Authorization" not in call_kwargs.get("headers", {})

    # --- Action & Assertions for Bearer Token ---
    api_key = "my-secret-key"
    download_custom_model(
        url="http://fake.url/bearer.bin", save_dir=str(tmp_path), auth=api_key
    )
    call_args, call_kwargs = mock_get.call_args
    assert call_kwargs.get("auth") is None
    assert call_kwargs.get("headers", {}).get("Authorization") == f"Bearer {api_key}"


# ===============================================================
# == Tests for download_model (High-Level Wrapper)
# ===============================================================


@patch("fasttext.util.util.download_custom_model")
@patch("fasttext.util.util.get_model_url")
def test_download_model_success_wrapper(mock_get_url, mock_downloader, tmp_path):
    """
    Tests that the high-level wrapper correctly calls the low-level function.
    """
    # --- Setup ---
    fake_url = "http://resolved.url/cc.en.300.bin.gz"
    expected_path = str(tmp_path / "cc.en.300.bin")

    mock_get_url.return_value = fake_url
    mock_downloader.return_value = expected_path

    # --- Action ---
    result = download_model(
        category="vectors_crawl",
        lang_id="en",
        model_type="bin",
        save_dir=str(tmp_path),
        if_exists="overwrite",
    )

    # --- Assertions ---
    mock_get_url.assert_called_once_with("vectors_crawl", "en", "bin")
    mock_downloader.assert_called_once_with(
        fake_url, str(tmp_path), None, "overwrite", None
    )
    assert result == expected_path


@patch("fasttext.util.util.download_custom_model")
@patch("fasttext.util.util.get_model_url")
def test_download_model_url_not_found(mock_get_url, mock_downloader):
    """
    Tests that the wrapper handles errors from URL resolution.
    """
    # --- Setup ---
    mock_get_url.side_effect = ValueError("Model not found")

    # --- Action ---
    result = download_model(category="invalid_category")

    # --- Assertions ---
    assert result is None
    mock_downloader.assert_not_called(), "Downloader should not be called if URL resolution fails"
