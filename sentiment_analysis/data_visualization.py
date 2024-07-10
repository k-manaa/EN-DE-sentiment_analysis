import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import re
import numpy as np

def extract_file_name(file: str) -> str:
    """
    Extracts the file name from the file path and returns a modified file name based on the language and model type.
    """
    file_name = os.path.basename(file)
    match = re.search(r"(DE|EN)reviews_processed_(multi|german|english).csv", file_name)
    if match:
        language = match.group(1)
        model_type = match.group(2)
        file_name = f"{language}reviews_{model_type}"
    return file_name


def plot_top_categories(df: pd.DataFrame, category_column: str, sentiment_column: str, file_name: str, normalized_scores: list) -> None:
    """
    Plots the top categories based on mean sentiment scores and saves the plot as an image file.
    """
    # Group the data by category and calculate the mean sentiment for each category
    category_sentiment = df.groupby(category_column)[sentiment_column].mean()

    # Sort the categories based on mean sentiment score
    sorted_categories = category_sentiment.sort_values()

    # Get the top and bottom categories
    top_negative = sorted_categories.head(5)
    top_positive = sorted_categories.tail(5)

    # Concatenate the top negative and top positive categories
    top_categories = pd.concat([top_negative, top_positive])

    # Plot the top categories
    plt.figure(figsize=(10, 6))
    colors = ['red'] * 5 + ['green'] * 5
    ax = top_categories.plot(kind='bar', color=colors)
    plt.xlabel(category_column)
    plt.ylabel('Mean Sentiment Score')
    plt.title(f'Top 5 Most Negative and Positive Categories ({file_name})')
    plt.xticks(rotation=45)

    # Add data labels above each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

    # Save the plot as an image file
    plot_folder = 'plots'
    os.makedirs(plot_folder, exist_ok=True)
    plot_file = os.path.join(plot_folder, f"{file_name}_top_categories.png")
    plt.savefig(plot_file)
    plt.close()  # Close the figure to free up memory

    # Print the file name of the saved plot
    print(f"Saved plot of top categories as '{plot_file}'.")


def compare_sentiment_by_language(input_files: list, language_column: str, sentiment_column: str, normalized_scores: list) -> None:
    """
    Compares the sentiment scores by language and saves the plot as an image file.
    """
    dfs = []

    for file in input_files:
        chunk_iterator = pd.read_csv(file, chunksize=1000)
        for chunk in chunk_iterator:
            if sentiment_column in chunk.columns:
                dfs.append(chunk)

    if not dfs:
        print(f"No data found for sentiment column '{sentiment_column}' in any of the input files.")
        return

    combined_df = pd.concat(dfs)

    # Group the combined data by language and calculate the mean normalized sentiment for each language
    language_sentiment = combined_df.groupby(language_column)[sentiment_column].mean()

    # Plot the sentiment scores by language
    plt.figure(figsize=(10, 6))
    language_sentiment.plot(kind='bar')
    plt.xlabel(language_column)
    plt.ylabel('Mean Normalized Sentiment')
    plt.title('Sentiment by Language')
    plt.xticks(rotation=45)

    # Save the plot as an image file
    plot_folder = 'plots'
    os.makedirs(plot_folder, exist_ok=True)
    plot_file = os.path.join(plot_folder, "sentiment_by_language.png")
    plt.savefig(plot_file)
    plt.close()  # Close the figure to free up memory

    # Print the file name of the saved plot
    print(f"Saved plot of sentiment by language as '{plot_file}'.")


def compare_sentiment_by_module(input_files: list, normalized_scores: list) -> None:
    """
    Compares the mean normalized sentiment scores by module and saves the plot as an image file.
    """
    module_scores = {'Multi': [], 'English': [], 'German': []}

    for file, score in zip(input_files, normalized_scores):
        file_name = extract_file_name(file)

        match = re.search(r"(multi|english|german)$", file_name, re.IGNORECASE)
        if match:
            module_name = match.group(1).capitalize()
            module_scores[module_name].append(score)

    # Plot the mean normalized scores for each module
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    labels = ['Multi', 'English', 'German']
    x = np.arange(len(module_scores))

    for i, (module_name, scores) in enumerate(module_scores.items()):
        if scores:
            plt.bar(x[i], sum(scores) / len(scores), color=colors[i], label=labels[i])

    plt.xlabel('Module')
    plt.ylabel('Mean Normalized Sentiment')
    plt.title('Mean Normalized Sentiment by Module')
    plt.xticks(x, list(module_scores.keys()))

    # Save the plot as an image file
    plot_folder = 'plots'
    os.makedirs(plot_folder, exist_ok=True)
    plot_file = os.path.join(plot_folder, "sentiment_by_module.png")
    plt.savefig(plot_file)
    plt.close()  # Close the figure to free up memory

    # Print the file name of the saved plot
    print(f"Saved plot of mean normalized sentiment by module as '{plot_file}'.")


def print_sentiment_comparison(processed_files: list) -> None:
    """
    Prints the sentiment comparison based on processed files and saves the overall sentiment comparison plot as an image file.
    """
    file_names = []
    normalized_scores = []

    for file in processed_files:
        file_name = extract_file_name(file)
        file_names.append(file_name)

        chunk_iterator = pd.read_csv(file, chunksize=1000)

        scores_before = []
        scores_after = []

        for chunk in chunk_iterator:
            scores = chunk['sentiment_label']
            scores_before.extend(scores.values)
            scores = preprocessing.MinMaxScaler().fit_transform(np.array(scores.values).reshape(-1, 1))
            scores_after.extend(scores.flatten())

        scores_before = np.array(scores_before)
        scores_after = np.array(scores_after)

        normalized_scores.append(scores_after.mean())

        print("Mean Sentiment Scores for", file_name)
        print("Before Normalization:", scores_before.mean())
        print("After Normalization:", scores_after.mean())
        print()

        df = pd.read_csv(file)  # Read the entire file for plotting
        plot_top_categories(df, 'product_category', 'sentiment_label', file_name, normalized_scores)

    compare_sentiment_by_language(processed_files, 'language', 'sentiment_label', normalized_scores)
    compare_sentiment_by_module(processed_files, normalized_scores)

    result_df = pd.DataFrame({'File Name': file_names, 'Mean Normalized Sentiment': normalized_scores})
    result_df.set_index('File Name', inplace=True)

    ax = result_df.plot(kind='bar', rot=0)
    ax.set_xticklabels(result_df.index, rotation=45)
    plt.xlabel('File Name')
    plt.ylabel('Mean Normalized Sentiment')
    plt.title('Sentiment Comparison')

    # Save the plot as an image file
    plot_folder = 'plots'
    os.makedirs(plot_folder, exist_ok=True)
    plot_file = os.path.join(plot_folder, "sentiment_comparison.png")
    plt.savefig(plot_file)
    plt.close()  # Close the figure to free up memory

    # Print the file name of the saved plot
    print(f"Saved plot of sentiment comparison as '{plot_file}'.")


if __name__ == '__main__':
    processed_data_directory = 'data'
    processed_files = [os.path.join(processed_data_directory, file) for file in os.listdir(processed_data_directory) if re.search(r"(DE|EN)reviews_processed_(multi|german|english)\.csv$", file, re.IGNORECASE)]
    print_sentiment_comparison(processed_files)
