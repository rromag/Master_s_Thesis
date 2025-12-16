import pandas as pd
import torch
from transformers import pipeline


# Specify model
MODEL = "chkla/roberta-argument"

# Specify pipeline for argument detection
classifier = pipeline("text-classification", model=MODEL, top_k=None)

# model and pipeline outside function so it only get's loaded once when calling the wrapper function instead of being loaded and created every time a new chunk is processed in the wrapper function


def MovieReviewArgumentDetection(Movie_Review_DataFrame):
    """
    Perform Argument Detection on a DataFrame containing Movie Reviews using the 'chkla/roberta-argument' model published on Hugging Face.
    """

    # Create list containing movie review, serving as input for the analysis pipeline
    review_list = Movie_Review_DataFrame["cleanedReviews"].to_list()

    # Perform Argument Detection, max_length = 256 due to memory bottleneck
    result = classifier(review_list,
                        batch_size = 64,
                        truncation = True,
                        padding = True,
                        max_length = 256)
    
    # Store results in DataFrame
    argument_data = pd.DataFrame([{item["label"]: item["score"] for item in res} for res in result])

    # Add review Id to DataFrame for future merges with movie/review data
    argument_data["reviewId"] = Movie_Review_DataFrame["reviewId"].values

    # Reordering columns to show "reviewId" first
    cols = ["reviewId"] + [c for c in argument_data.columns if c != "reviewId"]
    argument_data = argument_data[cols]

    return argument_data
