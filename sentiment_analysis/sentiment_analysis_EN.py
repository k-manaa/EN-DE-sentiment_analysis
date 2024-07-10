import torch
import os
import csv
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def analyze_sentiment(review: str, model, tokenizer) -> int:
    """
    Analyzes the sentiment of a review using a given model and tokenizer. Returns the predicted class.
    """
    encoded_input = tokenizer(review, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        logits = model(**encoded_input).logits
        predicted_class = logits.argmax(dim=1)
    return predicted_class.item()

def perform_sentiment_analysis_EN_multi(input_file_path: str) -> None:
    """
    Performs sentiment analysis for English reviews using a multi-lingual model.
    """
    tokenizer_multi = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
    model_multi = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

    output_folder = os.path.dirname(input_file_path)  # Output folder same as input folder
    output_file_multi = os.path.join(output_folder, "ENreviews_processed_multi.csv")

    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
            open(output_file_multi, 'w', newline='', encoding='utf-8') as output_file_multi:

        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames + ['sentiment_label']
        writer_multi = csv.writer(output_file_multi)
        writer_multi.writerow(fieldnames)

        print(f"Performing sentiment analysis for English reviews (Multi model)")

        for line in tqdm(reader, desc="Processing", unit=" reviews"):
            review = line['review_body']
            sentiment_label = analyze_sentiment(review, model_multi, tokenizer_multi)
            line['sentiment_label'] = sentiment_label
            writer_multi.writerow([line[field] for field in fieldnames])

    print("Sentiment analysis complete for English reviews (Multi model).")

def perform_sentiment_analysis_EN_english(input_file_path: str) -> None:
    """
    Performs sentiment analysis for English reviews using an English model.
    """
    tokenizer_english = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
    model_english = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")

    output_folder = os.path.dirname(input_file_path)  # Output folder same as input folder
    output_file_language = os.path.join(output_folder, "ENreviews_processed_english.csv")

    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
            open(output_file_language, 'w', newline='', encoding='utf-8') as output_file_language:

        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames + ['sentiment_label']
        writer_language = csv.writer(output_file_language)
        writer_language.writerow(fieldnames)

        print(f"Performing sentiment analysis for English reviews (English model)")

        for line in tqdm(reader, desc="Processing", unit=" reviews"):
            review = line['review_body']
            sentiment_label = analyze_sentiment(review, model_english, tokenizer_english)
            line['sentiment_label'] = sentiment_label
            writer_language.writerow([line[field] for field in fieldnames])

    print("Sentiment analysis complete for English reviews (English model).")

if __name__ == '__main__':
    input_file = "data/ENreviews_processed.csv"
    perform_sentiment_analysis_EN_multi(input_file)
    perform_sentiment_analysis_EN_english(input_file)
