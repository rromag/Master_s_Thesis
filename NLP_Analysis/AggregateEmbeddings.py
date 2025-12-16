import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def AggregateEmbeddings(Review_Type):
    """
    Aggregate review-level embeddings into a single average embedding per movie.

    Parameters
    ----------
    Review_Type : str
        Either "Audience" or "Critic".

    Returns
    -------
    AvgEmbeddings : pd.DataFrame
        DataFrame with columns:
        - 'id': Movie identifier
        - 'embeddings': List of floats representing the average embedding vector
    
    Side Effects
    ------------
    - Saves aggregated embeddings to:
      NLP Data/{Review_Type} Embeddings/rt_{review_type}_embeddings_aggregated.parquet
    """

    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    num_files = {"Audience": 25, "Critic": 20}

    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Check for valid Review_Type
    if Review_Type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review Type: {Review_Type}. Must be one of {VALID_REVIEW_TYPES}")
    
    # Set up paths for reading / writing data
    folder = PROJECT_ROOT / "NLP Data" / f"{Review_Type} Embeddings"
    output_path = folder / f"rt_{Review_Type.lower()}_embeddings_aggregated.parquet"

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Looking in folder:", folder)

    # Load Data
    embeddings_data = pd.concat([pd.read_parquet(folder / f"rt_{Review_Type.lower()}_embeddings_{i}.parquet") for i in range(num_files[Review_Type])], ignore_index = True)
    embeddings_data["embeddings"] = embeddings_data["embeddings"].apply(np.array)

    # Aggregate embeddings with progress tracker
    tqdm.pandas()
    AvgEmbeddings = (embeddings_data.groupby("id")["embeddings"].progress_apply(lambda x: np.mean(np.stack(x), axis=0).tolist()).reset_index())

    print(f"Processed {len(embeddings_data)} reviews across {embeddings_data['id'].nunique()} movies.")

    # Write to file
    AvgEmbeddings.to_parquet(output_path)

    return AvgEmbeddings