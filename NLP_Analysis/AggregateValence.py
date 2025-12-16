import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm



def AggregateValence(Review_Type):
    """
    Aggregate sentiment valence scores for movie reviews of a given type.

    This function loads sentiment analysis outputs for either audience or critic
    reviews, transforms negative sentiment scores into negative values, and 
    computes the mean valence score per movie (grouped by movie ID). The aggregated
    results are written to a JSON file and returned as a DataFrame.

    Parameters
    ----------
    Review_Type : str
        The type of reviews to process. Must be either "Audience" or "Critic".

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns:
        - "id": The unique movie identifier.
        - "AvgValence": The mean valence score across all reviews for that movie.

    Notes
    -----
    - Input data must exist in the following folder structure relative to the project root:
        NLP Data/{Review_Type} Sentiment Data/rt_{review_type}_reviews_sentiment_{i}.json
    - Output is saved as:
        NLP Data/{Review_Type} Sentiment Data/rt_{review_type}_valence_aggregated.json
    - The function assumes a fixed number of files per review type
      (25 for audience reviews, 20 for critic reviews).
    - Negative sentiment scores are multiplied by -1 before aggregation to ensure
      the valence score correctly reflects sentiment polarity.

    Examples
    --------
    >>> AggregateValence("Audience")
    PROJECT_ROOT: /path/to/project
    Looking in folder: /path/to/project/NLP Data/Audience Sentiment Data
    Length Valence Data: 120000
    Processed 120000 reviews across 500 movies.
    """

    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    num_files = {"Audience": 25, "Critic": 20}

    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Check for valid Review_Type
    if Review_Type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review Type: {Review_Type}. Must be one of {VALID_REVIEW_TYPES}")
    
    # Set up paths for reading / writing data
    folder = PROJECT_ROOT / "NLP Data" / f"{Review_Type} Sentiment Data"
    output_path = folder / f"rt_{Review_Type.lower()}_valence_aggregated.json"

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Looking in folder:", folder)

    # Load Data
    valence_data = pd.concat([pd.read_json(folder / f"rt_{Review_Type.lower()}_reviews_sentiment_{i}.json") for i in range(num_files[Review_Type])], ignore_index=True)

    print(f"Length Valence Data: {valence_data.shape[0]}")

    # Transform negative sentiment scores to negative values
    valence_data.loc[valence_data["sentiment"] == "Negative", "sentimentScore"] *= -1

    # Aggregate Valence on movie id
    tqdm.pandas()
    AvgValence = valence_data.groupby("id")["sentimentScore"].progress_apply(np.mean).reset_index(name="AvgValence")

    print(f"Processed {len(valence_data)} reviews across {valence_data['id'].nunique()} movies.")

    # Write to file
    AvgValence.to_json(output_path, date_format="iso", orient="records", indent=2)

    return AvgValence
