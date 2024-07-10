# Sentiment analysis of English and German Amazon product reviews

Analyzing and comparing sentiment analysis scores in the two languages using two modules, which are a 
language specific module, and a multi-lingual module.


# Introduction

Amazon product reviews are a good option for Natural Language Processing tasks, mainly due to their wide availability. 

The general aim of this project is to perform a sentiment analysis on a corpus of Amazon product reviews, in two languages: English and German. 

Additionally, the sentiment analyser will use two modules for each language. The sentiment analysis on the English reviews uses the following BERT modules:

1. "LiYuan/amazon-review-sentiment-analysis"

2. "siebert/sentiment-roberta-large-english"

As for the German reviews, the following BERT modules were used:

1. "LiYuan/amazon-review-sentiment-analysis"

2. "oliverguhr/german-sentiment-bert"

In addition to the preprocessing and tokenization of each module, I developed my own preprocessing script in order to add the review_title contents to the review_body, for more data to analyze, alongside other adjustments for removal of emojis. 


# Commandline Examples

Included are two files for the Amazon product reviews in their respective languages. They are:

1. "ENreviews.csv"

2. "DEreviews.csv"

They will first be preprocessed, then analyzed for sentiment, and finally used to visualize sentiment scores. In order to do that for both languages, use this command: 

poetry run python sentiment_analysis/main.py data/ENreviews.csv data/DEreviews.csv --length 100 --stats

Below is an explanation of the arguments used:

1. The main.py file that executes the programs. (MANDATORY)

2. The input file path(s). (MANDATORY)
The obligatory number is one, and the maximum number is two. That allows for a language specific study. In that case, only one path should be provided.

3. The length of rows to be processed. (OPTIONAL)
IMPORTANT: If no length is specified, the full amount of 200,000 rows will be used. 

4. Data visualization and statistics. (OPTIONAL)
If chosen, several plots will be generated and saved in a separate folder, fittingly named "plots".

NOTE: Seeing as running my program generates a generous amount of output, I provided a cleaner script whose command allows you to start over again. In that case, the command is as follows:

poetry run python sentiment_analysis/cleaner.py data plots

Below is an explanation of the arguments:

1. The main cleaner.py file that executes the program. (MANDATORY)

2. The name of the folder to be cleaned.
This argument expects at least one name, and can be extended to two in order to accomodate cleaning both the "data" and "plots". It is implemented in a way such that it will not delete the two original .csv files for each language's Amazon reviews.


# General Remarks

1. The English language module runs at a slower speed than the others due to its size.

2. The sentiment scores had to be normalized, as the different modules had different scales. 

3. I had a lot of fun creating this!


# Links to Modules 

1.	Multilingual BERT-based model for Amazon reviews.
https://huggingface.co/LiYuan/amazon-review-sentiment-analysis?text=I+like+you.+I+love+you

2.	German language texts sentiment classifying model.
https://huggingface.co/oliverguhr/german-sentiment-bert?text=Das+ist+gar+nicht+mal+so+schlecht

3.	English language texts sentiment classifying model.
https://huggingface.co/siebert/sentiment-roberta-large-english


# References

Phillip Keung, Yichao Lu, György Szarvas and Noah A. Smith. “The Multilingual Amazon Reviews Corpus.” In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020. https://huggingface.co/datasets/amazon_reviews_multi 
 
Haque, Tanjim & Saber, Nudrat & Shah, Faisal. (2018). Sentiment analysis on large scale Amazon product reviews. 10.1109/ICIRD.2018.8376299. 
 
Elmurngi, Elshrif & Gherbi, Abdelouahed. (2018). Unfair reviews detection on Amazon reviews using sentiment analysis with supervised learning techniques. JCS. 14. 714-726. 10.3844/jcssp.2018.714.726. 
 
Srujan, K. & Nikhil, s & Rao, Raghav & Kedage, Karthik & Harish, B S & Keerthi Kumar, H M. (2018). Classification of Amazon Book Reviews Based on Sentiment Analysis. 
10.1007/978-981-10-7512-4_40.  
 
Hartmann, J., Heitmann, M., Siebert, C., & Schamp, C. (2023). More than a feeling: Accuracy and application of sentiment analysis. International Journal of Research in Marketing, 40(1), 75–87. https://doi.org/10.1016/j.ijresmar.2022.05.005

