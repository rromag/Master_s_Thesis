import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException


DetectorFactory.seed = 0  # makes langdetect deterministic


def DetectLanguage(Movie_Review_Data: pd.DataFrame):
    """
    Detect the language of movie reviews using the `langdetect` library.

    For each review in the input DataFrame, this function attempts to identify 
    the language of the `reviewText` column and stores the result in a new 
    column called `language`. If detection fails, or if the review text is 
    missing, too short, or not a string, the language is set to "unknown".

    Parameters
    ----------
    Movie_Review_Data : pandas.DataFrame
        DataFrame containing at least a `reviewText` column with review strings.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional `language` column containing 
        detected language codes (e.g., "en", "fr", "de") or "unknown" if 
        detection was not possible.

    Notes
    -----
    - Language detection can fail for very short texts, non-text values, or 
      ambiguous strings. These cases are handled by assigning "unknown".
    - `DetectorFactory.seed = 0` ensures deterministic results across runs.
    """
    # Initialize language column with "unknown" for all rows
    Movie_Review_Data["language"] = "unknown"

    # Iterate over each review to detect language
    for idx, row in Movie_Review_Data.iterrows():
        try:
            # Skip invalid or very short reviews
            if not isinstance(row["reviewText"], str) or len(row["reviewText"].strip()) < 5:
                language = "unknown"
            else:
                # Detect language using langdetect
                language = detect(row["reviewText"])
        except LangDetectException:
                # Handle detection errors
                language = "unknown"

        # Store result back in the DataFrame
        Movie_Review_Data.loc[idx, "language"] = language
        
    return Movie_Review_Data
