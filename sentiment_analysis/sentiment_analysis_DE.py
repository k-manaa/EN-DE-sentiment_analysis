import torch
import os
import csv
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def analyze_sentiment(review: str, model, tokenizer) -> int:
    """
    Analyzes the sentiment of a review using a given model and tokenizer, returns the predicted class.
    """
    encoded_input = tokenizer(review, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        logits = model(**encoded_input).logits
        predicted_class = logits.argmax(dim=1)
    return predicted_class.item()

def perform_sentiment_analysis_DE_multi(input_file_path: str) -> None:
    """
    Performs sentiment analysis for German reviews using a multi-lingual model.
    """
    tokenizer_multi = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
    model_multi = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

    output_folder = os.path.dirname(input_file_path)  # Output folder same as input folder
    output_file_multi = os.path.join(output_folder, "DEreviews_processed_multi.csv")

    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
            open(output_file_multi, 'w', newline='', encoding='utf-8') as output_file_multi:

        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames + ['sentiment_label']
        writer_multi = csv.writer(output_file_multi)
        writer_multi.writerow(fieldnames)

        print(f"Performing sentiment analysis for German reviews (Multi model)")

        for line in tqdm(reader, desc="Processing", unit=" reviews"):
            review = line['review_body']
            sentiment_label = analyze_sentiment(review, model_multi, tokenizer_multi)
            line['sentiment_label'] = sentiment_label
            writer_multi.writerow([line[field] for field in fieldnames])

    print("Sentiment analysis complete for German reviews (Multi model).")

def perform_sentiment_analysis_DE_german(input_file_path: str) -> None:
    """
    Performs sentiment analysis for German reviews using a German model.
    """
    tokenizer_german = AutoTokenizer.from_pretrained("oliverguhr/german-sentiment-bert")
    model_german = AutoModelForSequenceClassification.from_pretrained("oliverguhr/german-sentiment-bert")

    output_folder = os.path.dirname(input_file_path)  # Output folder same as input folder
    output_file_language = os.path.join(output_folder, "DEreviews_processed_german.csv")

    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
            open(output_file_language, 'w', newline='', encoding='utf-8') as output_file_language:

        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames + ['sentiment_label']
        writer_language = csv.writer(output_file_language)
        writer_language.writerow(fieldnames)

        print(f"Performing sentiment analysis for German reviews (German model)")

        for line in tqdm(reader, desc="Processing", unit=" reviews"):
            review = line['review_body']
            sentiment_label = analyze_sentiment(review, model_german, tokenizer_german)
            line['sentiment_label'] = sentiment_label
            writer_language.writerow([line[field] for field in fieldnames])

    print("Sentiment analysis complete for German reviews (German model).")

if __name__ == '__main__':
    input_file = "data/DEreviews_output/DEreviews_processed.csv"
    perform_sentiment_analysis_DE_multi(input_file)
    perform_sentiment_analysis_DE_german(input_file)
