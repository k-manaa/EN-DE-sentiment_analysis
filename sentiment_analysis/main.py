import argparse
import glob
from typing import Optional

from sentiment_analysis.preprocessing import process_dataset
from sentiment_analysis.sentiment_analysis_DE import perform_sentiment_analysis_DE_german, perform_sentiment_analysis_DE_multi
from sentiment_analysis.sentiment_analysis_EN import perform_sentiment_analysis_EN_english, perform_sentiment_analysis_EN_multi
from sentiment_analysis.data_visualization import print_sentiment_comparison


def perform_sentiment_analysis(input_files: list, length: Optional[int] = None, stats: bool = False) -> None:
    """
    Perform sentiment analysis on input files.
    """
    # Define the paths to the processed files
    german_processed_file = 'data/DEreviews_processed.csv'
    english_processed_file = 'data/ENreviews_processed.csv'

    for input_file in input_files:
        if "DE" in input_file:
            # German dataset processing
            process_dataset(input_file, 'german', length)
            perform_sentiment_analysis_DE_german(german_processed_file)
            perform_sentiment_analysis_DE_multi(german_processed_file)

        if "EN" in input_file:
            # English dataset processing
            process_dataset(input_file, 'english', length)
            perform_sentiment_analysis_EN_english(english_processed_file)
            perform_sentiment_analysis_EN_multi(english_processed_file)

    if stats:
        # Generate sentiment analysis visualizations
        english_processed_files = glob.glob('data/ENreviews_processed_*.csv')
        german_processed_files = glob.glob('data/DEreviews_processed_*.csv')
        input_files = english_processed_files + german_processed_files
        print_sentiment_comparison(input_files)


def main() -> None:
    """
    Main function to parse command line arguments and perform sentiment analysis.
    """
    parser = argparse.ArgumentParser(description='Perform sentiment analysis on a language.')
    parser.add_argument('--length', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--stats', action='store_true', help='Generate sentiment analysis visualizations')
    parser.add_argument('input_files', type=str, nargs='+', help='Paths to the input file(s)')
    args = parser.parse_args()

    perform_sentiment_analysis(args.input_files, args.length, args.stats)


if __name__ == "__main__":
    main()
