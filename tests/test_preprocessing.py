import pytest
import pandas as pd
from sentiment_analysis.preprocessing import process_review_body


def test_process_review_body():
    # Test input data
    row = pd.Series({'review_body': "This is a simple review."})
    language = 'english'

    # Define expected output
    expected_output = "simple review"

    # Call the function to process the review body
    processed_review_body = process_review_body(row, language)

    # Check if the processed review body matches the expected output
    assert processed_review_body == expected_output
