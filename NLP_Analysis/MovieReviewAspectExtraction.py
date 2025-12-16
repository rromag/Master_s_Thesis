import pandas as pd
from pyabsa import AspectTermExtraction as ATE


# Specify Model
aspect_extractor = ATE.AspectExtractor("english", 
                                         cal_perplexity = False,
                                         auto_device = True)

# model outside function so it only get's loaded once when calling the wrapper function instead of being loaded and created every time a new chunk is processed in the wrapper function



def MovieReviewAspectExtraction(Movie_Review_DataFrame):
    """
    Run Aspect Extraction on Movie Reviews.
    """
    # Create list containing movie review, serving as input
    review_list = Movie_Review_DataFrame["cleanedReviews"].to_list()

    # Perform aspect extraction on a sample sentence
    result = aspect_extractor.extract_aspect(
        review_list,
        save_result=False,
        print_result=False,
        batch_size = 512)
    
    # Store results in DataFrame
    aspect_data = pd.DataFrame(result)

    # Add review Id to DataFrame for future merges with movie/review data
    aspect_data["reviewId"] = Movie_Review_DataFrame["reviewId"].values

    # Reordering columns to show "reviewId" first
    cols = ["reviewId", "sentence", "aspect", "sentiment", "probs", "confidence", "tokens", "position", "IOB"]
    aspect_data = aspect_data[cols]

    return aspect_data

