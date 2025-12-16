import time
import pandas as pd
from deep_translator import GoogleTranslator

def MovieReviewTranslatorGoogle(Movie_Review_Data):
    """
    Translate batches of movie reviews using google translate.
    """
    # Extract Information from the Dataset, truncate reviews to a max length of 5000 characters which is the Google Translate limit
    reviews = Movie_Review_Data["reviewText"].astype(str).apply(lambda x: x[:5000]).to_list()
    idx = Movie_Review_Data.index

    results = [None] * len(reviews) # return None if translation fails

    # Translate reviews
    try:
        results = GoogleTranslator(source="auto", target="en").translate_batch(reviews)
    
    # Translation failed -> fall back to per-review translation
    except Exception as e:
        print("Falling back to per-review translation…")
        for i , review in enumerate(reviews):
            try:
                results[i] = GoogleTranslator(source="auto", target="en").translate(review)
            except Exception as e:
                print(f"Translation failed for review {i}")
    
    # Give the server a break
    time.sleep(0.5)

    # Create DataFrame with translated reviews
    return pd.Series(results, index=idx, name="reviewText")
