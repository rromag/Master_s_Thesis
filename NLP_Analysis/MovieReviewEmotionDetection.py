import pandas as pd
import torch
from transformers import pipeline

# Load tokenizer and model
model_name = "borisn70/bert-43-multilabel-emotion-detection"

# Create pipeline
classifier = pipeline("text-classification", model = model_name, top_k = None, function_to_apply="softmax") #function_to_apply="sigmoid")

# model and pipeline outside function so it only get's loaded once when calling the wrapper function instead of being loaded and created every time a new chunk is processed in the wrapper function


def MovieReviewEmotionDetection(Movie_Review_DataFrame):
    """
     Perform Emotion Detection on a DataFrame containing Movie Reviews using the 'borisn70/bert-43-multilabel-emotion-detection' model published on hugging face.
    """
    # Create list containing the movie reviews, serving as input for the analysis pipeline
    review_list = Movie_Review_DataFrame["cleanedReviews"].tolist()

    # Run Emotion Detection, max_length = 256 due to memory bottleneck
    predictions = classifier(review_list,
                             batch_size = 64,
                             truncation = True,
                             padding = True,
                             max_length = 256)

    # Create DataFrame containing results
    emotion_data = pd.DataFrame({item["label"]: item["score"] for item in pred} for pred in predictions)
    emotion_data["reviewId"] = Movie_Review_DataFrame["reviewId"]

    # Reordering columns to show "reviewId first"
    cols = ["reviewId"] + [c for c in emotion_data.columns if c != "reviewId"]
    emotion_data = emotion_data[cols]

    return emotion_data
