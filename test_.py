from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from datasets import Dataset

from utils import filter_by_cpt, filter_by_user_codes


def create_test_dataset():
    data = {
        "id": [4586, 7860, 8031],
        "case_version": [7326.0, 10828.0, 10999.0],
        "patient_type": ["Inpatient", "Ambulatory Surgery", "Outpatient Clinic Dept"],
        "text": ["Discharge Summary Text", "History and Physical Text", "Medical Record Text"],
        "pdx": ["M17.12", "K29.50", "N20.0"],
        "sdx": ["J45.909, I25.10, E78.5, I10, E03.9, M21.162, M17.12", "E11.9, I10, G47.33, J45.909, E78.5, K21.9, M50...", "N28.89, N40.0"],
        "adx": ["M17.12", None, None],
        "drg": [470, None, None],
        "px": [0, None, None],
        "cpt": [None, "43239", "74176"],
        "icd_codes": [
            "M17.12, J45.909, I25.10, E78.5, I10, E03.9, M21.162, M17.12",
            "K29.50, E11.9, I10, G47.33, J45.909, E78.5, K21.9, M50...",
            "N28.89, N40.0, N20.0",
        ],
        "specialty": [None, None, None],
        "layterm_data": [["Data 1"], ["Data 2"], []],
    }
    return pd.DataFrame(data)


def test_filter_by_cpt_with_updated_dataset():
    dataset = create_test_dataset()
    cpts = "43239,74176"
    expected_ids = [7860, 8031]

    filtered_dataset = filter_by_cpt(Dataset.from_pandas(dataset), cpts)

    # Verify that the filtered dataset only contains rows with the expected IDs
    assert all(row["id"] in expected_ids for row in filtered_dataset), "Filtered dataset contains unexpected IDs."

    assert len(filtered_dataset) == 2, f"Filtered dataset contains {len(filtered_dataset)} rows, expected 2."


@patch("datasets.Dataset")
def test_filter_by_user_codes(mock_dataset):
    # Mock data setup
    mock_data = [
        {"code": "A123", "description": "Test Description 1"},
        {"code": "B456", "description": "Test Description 2"},
        {"code": "C789", "description": "Test Description 3"},
    ]

    # Configure the mock to return a filtered version of mock_data when .filter() is called
    def filter_func(func):
        return [item for item in mock_data if func(item)]

    # Mock the filter method of the Dataset object
    mock_dataset.filter.side_effect = filter_func

    # Create an instance of the mocked Dataset filled with mock_data
    dataset_instance = MagicMock()
    dataset_instance.filter.side_effect = filter_func

    # Test with user_codes as a list
    user_codes_list = ["A123", "C789"]
    filtered_dataset_list = filter_by_user_codes(dataset_instance, user_codes_list)
    assert len(filtered_dataset_list) == 2
    assert all(item["code"] in user_codes_list for item in filtered_dataset_list), "Filtering with a list of user codes did not work as expected."

    # Test with user_codes as a string
    user_codes_str = "A123, C789"
    filtered_dataset_str = filter_by_user_codes(dataset_instance, user_codes_str)
    assert len(filtered_dataset_str) == 2
    assert all(item["code"] in user_codes_list for item in filtered_dataset_str), "Filtering with a string of user codes did not work as expected."
