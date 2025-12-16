import pandas as pd
from transformers import pipeline


# Specify model to use
MODEL = "srimeenakshiks/aspect-based-sentiment-analyzer-using-bert"

# Specify pipeline for sentiment analysis
classifier = pipeline("sentiment-analysis", model = MODEL)    

# model and pipeline outside function so it only get's loaded once when calling the wrapper function instead of being loaded and created every time a new chunk is processed in the wrapper function



def MovieReviewSentimentAnalyser(Movie_Review_DataFrame):
    """
    Perform Sentiment Analysis on a DataFrame containing Movie Reviews using the 'distilbert/distilbert-base-uncased-finetuned-sst-2-english' model published on Hugging Face.
    """
    # Create list containing the movie reviews, serving as input for the analysis pipeline
    review_list = Movie_Review_DataFrame["cleanedReviews"].to_list()

    # Perform sentiment analysis, max_length = 256 due to memory bottleneck
    result = classifier(review_list,
                        batch_size = 64,
                        truncation = True,
                        padding = True,
                        max_length = 256)
    
    # Store results in DataFrame
    sentiment_data = pd.DataFrame([res for res in result])
    sentiment_data.rename(columns={"label": "sentiment", "score": "sentimentScore"}, inplace=True)
    sentiment_data["sentiment"] = sentiment_data["sentiment"].replace(to_replace=["LABEL_1", "LABEL_0"], value=["Positive", "Negative"])

    # Add review Id to DataFrame for future merges with movie/review data
    sentiment_data["reviewId"] = Movie_Review_DataFrame["reviewId"].values

    # Add movie Id to DataFrame for future aggregation
    sentiment_data["id"] = Movie_Review_DataFrame["id"].values

    # Reordering columns to show "reviewId" first
    cols = ["reviewId"] + [c for c in sentiment_data.columns if c != "reviewId"]
    sentiment_data = sentiment_data[cols]

    return sentiment_data
