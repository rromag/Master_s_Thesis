import pandas as pd
import deepl


translator = deepl.Translator(
    # API-Key
)


def MovieReviewTranslatorDeepl(Movie_Review_Data):
    """
    Translate movie reviews using the deepl api. Warning: Very expensive!
    """
    # Extract information from the Dataset
    reviews = Movie_Review_Data["reviewText"].to_list()
    idx = Movie_Review_Data.index

    # Translate reviews
    results = translator.translate_text(reviews, target_lang="EN-GB")

    # Create DataFrame with translated reviews
    return pd.Series([result.text for result in results], index=idx, name="reviewText")
