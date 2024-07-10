import os
import pytest
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentiment_analysis.sentiment_analysis_EN import perform_sentiment_analysis_EN_multi
from sentiment_analysis.sentiment_analysis_DE import analyze_sentiment


def test_perform_sentiment_analysis_EN_multi():
    input_file_path = "data/ENreviews_processed.csv"
    # Replace the model and tokenizer paths with appropriate values for testing
    tokenizer_path = "LiYuan/amazon-review-sentiment-analysis"
    model_path = "LiYuan/amazon-review-sentiment-analysis"

    # Call the function
    perform_sentiment_analysis_EN_multi(input_file_path)

    # For example, check if the output file exists and has expected contents
    assert os.path.exists("data/ENreviews_processed_multi.csv")
    assert len(pd.read_csv("data/ENreviews_processed_multi.csv")) == len(pd.read_csv("data/ENreviews_processed.csv"))


def test_analyze_sentiment():
    # Test input data
    review = "Dies ist eine positive Bewertung."
    model = AutoModelForSequenceClassification.from_pretrained("oliverguhr/german-sentiment-bert")
    tokenizer = AutoTokenizer.from_pretrained("oliverguhr/german-sentiment-bert")

    # Call the function to analyze sentiment
    sentiment_label = analyze_sentiment(review, model, tokenizer)

    # Define the expected output
    expected_output = 2

    # Check if the sentiment label matches the expected output
    assert sentiment_label == expected_output
