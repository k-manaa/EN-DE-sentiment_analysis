import string
import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

def process_review_body(row: pd.Series, language: str) -> str:
    """
    Processes the review body by removing punctuation, stop words, and emojis.
    """

    # Define punctuation
    punctuation = set(string.punctuation + "“”‘’…")

    # Get language-specific stop words
    stop_words = set(get_stop_words(language))

    # Initialize lemmatizer for the specified language
    lemmatizer = WordNetLemmatizer()

    # Tokenize the review body
    tokens = word_tokenize(row['review_body'].lower(), language=language)

    # Remove punctuation, stop words, and lemmatize the tokens
    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if all(char not in punctuation for char in token) and token.lower() not in stop_words
    ]

    # Remove any tokens containing emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"  # dingbats
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+")
    processed_tokens = [
        token for token in processed_tokens if not bool(emoji_pattern.search(token))
    ]

    # Join the processed tokens back into a string
    processed_review_body = ' '.join(processed_tokens)

    return processed_review_body


def process_dataset(input_file: str, language: str, length=None) -> None:
    """
    Process the dataset by reading the input file, performing necessary transformations, and saving the processed file.
    """
    # Create output CSV file name based on language
    output_file = input_file.replace('.csv', f'_processed.csv')

    # Read the entire input CSV file
    df = pd.read_csv(input_file)

    if length is not None:
        # Process the dataset based on the specified length
        processed_df = df.sample(n=length, random_state=42)
    else:
        # Process the full dataset
        processed_df = df

    # Shuffle the DataFrame randomly by shuffling the index
    processed_df = processed_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract text from review_title and add it to review_body
    processed_df['review_body'] += ' ' + processed_df['review_title'].fillna('')

    # Remove the empty review_title column
    processed_df.drop(columns=['review_title'], inplace=True)

    # Process the review bodies
    processed_df['review_body'] = processed_df.apply(lambda row: process_review_body(row, language), axis=1)

    # Write the processed dataset to the output file
    processed_df.to_csv(output_file, index=False)

    print(f"Processing complete for {language} reviews. The processed data has been saved to '{output_file}'.")


def process_datasets() -> None:
    """
    Processes both German and English datasets.
    """
    # Specify the input file paths for German and English datasets
    german_input_file = 'data/DEreviews.csv'
    english_input_file = 'data/ENreviews.csv'

    # German dataset processing
    process_dataset(german_input_file, 'german')

    # English dataset processing
    process_dataset(english_input_file, 'english')


if __name__ == "__main__":
    process_datasets()
