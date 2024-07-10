import pytest
from sentiment_analysis.data_visualization import extract_file_name


def test_extract_file_name():
    # Test input data
    file = "data/DEreviews_processed_multi.csv"

    # Call the function to extract the file name
    extracted_file_name = extract_file_name(file)

    # Define the expected output
    expected_output = "DEreviews_multi"

    # Check if the extracted file name matches the expected output
    assert extracted_file_name == expected_output
