import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing
import time
import re, gc
from NLP_Preprocessing.ReplaceMovieTitles import ReplaceMovieTitles
from NLP_Preprocessing.ReplaceActorNames import ReplaceActorNames


def PreprocessMovieReviews(Review_Type, To_Replace, num_cores=8, timing=True):
    """
    Preprocess movie reviews by masking mentions of movies or actors in the review text.

    This function reads review JSON files (either raw or previously preprocessed), splits 
    them into subsets for parallel processing, and applies a text replacement function 
    (e.g., `ReplaceMovieTitles` or `ReplaceActorNames`) to replace movie or actor mentions 
    with standardized placeholders. The processed reviews are saved back to JSON files, 
    preserving original review indices.

    Parameters
    ----------
    Review_Type : str
        Type of review dataset to process. Must be either "Audience" or "Critic".
    To_Replace : str
        Type of entities to replace in reviews. Currently supports:
        - "movies": replaces movie titles using fuzzy matching.
        - "actors": replaces actor names using fuzzy matching.
    num_cores : int, optional, default=8
        Number of CPU cores to use for parallel processing.
    timing : bool, optional, default=True
        If True, prints elapsed processing time for each JSON file.

    Returns
    -------
    None
        The function writes processed JSON files to disk. Each review will have a new
        column `cleanedReviews` containing the masked text.

    Notes
    -----
    - Input JSON files should be located in:
        "Rotten Tomatoes Reviews/{Review_Type} Reviews Clean" (or previously preprocessed folder)
      and must contain columns including 'reviewId', 'id', 'title', and either 
      'reviewText' or 'cleanedReviews'.
    - The function automatically creates the output folder:
        "Rotten Tomatoes Reviews/{Review_Type} Reviews Preprocessed for Aspect Extraction"
      if it does not already exist.
    - Processing is performed in parallel by splitting reviews by movie ID to balance load 
      across cores.
    - The actual masking is performed by functions mapped in `PROCESSING_FUNCTIONS`:
        - `ReplaceMovieTitles` for movie titles
        - `ReplaceActorNames` for actor names
    - Original review indices are preserved when merging the masked reviews back into 
      the original DataFrame.
    """
    PROCESSING_FUNCTIONS = {"movies": ReplaceMovieTitles,
                            "actors": ReplaceActorNames}

    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    # Assign processing function
    processing_function = PROCESSING_FUNCTIONS.get(To_Replace.lower())
    
    # Check for valid Review_Type / To_Replace argument
    if processing_function is None:
        raise ValueError(f"Unsupported To_Replace Value: {To_Replace}.")
    if Review_Type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review_Type: {Review_Type}. Must be one of {VALID_REVIEW_TYPES}")
    
    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    print("PROJECT_ROOT:", PROJECT_ROOT)

    # Set up paths for reading / saving the data
    output_folder = PROJECT_ROOT / "Rotten Tomatoes Reviews" / f"{Review_Type} Reviews Preprocessed for NLP"
    
    if output_folder.exists():
        data_folder = output_folder
        print("Looking in folder:", data_folder)

        # List of files to process, sorted to maintain order
        json_files = sorted(data_folder.glob(f"rt_{Review_Type.lower()}_reviews_preprocessed_*.json"), key=lambda x: int(re.search(r"_(\d+)\.json$", x.name).group(1)))

    else:
        data_folder = PROJECT_ROOT / "Rotten Tomatoes Reviews" / f"{Review_Type} Reviews Clean"
        print("Looking in folder:", data_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # List of files to process, sorted to maintain order
        json_files = sorted(data_folder.glob(f"rt_{Review_Type.lower()}_reviews_clean_*.json"), key=lambda x: int(re.search(r"_(\d+)\.json$", x.name).group(1)))

    # Process files in parallel
    with multiprocessing.Pool(num_cores) as pool:

        for i, file_path in enumerate(json_files):
            # Load movie review data and initialize variables for processing and saving
            movie_data = pd.read_json(file_path)
            if "cleanedReviews" in movie_data.columns:
                processing_data = movie_data[["reviewId", "id", "title", "cleanedReviews"]].copy()
                processing_data.rename(columns={"cleanedReviews": "review"}, inplace=True)
            elif "reviewText" in movie_data.columns:
                processing_data = movie_data[["reviewId", "id", "title", "reviewText"]].copy()
                processing_data.rename(columns={"reviewText": "review"}, inplace=True)
            else:
                raise KeyError(f"No valid review column found in {file_path}. Expected 'reviewText' or 'cleanedReviews'.")

            # Track file being processed and the time required for processing
            print(f"[→] Processing file {i}/{len(json_files)-1}: {file_path.name}. Number of Reviews to Process: {processing_data.shape[0]}")
            start_time = time.time()

            # Split data for multiprocessing
            groups = [g for _, g in processing_data.groupby("id")]
            subsets = [[] for _ in range(num_cores)]
            sizes = [0] * num_cores

            for g in groups:
                idx = np.argmin(sizes)
                subsets[idx].append(g)
                sizes[idx] += len(g)

            subsets = [pd.concat(subset, ignore_index=False) if subset else pd.DataFrame(columns=processing_data.columns) for subset in subsets]

            # Call function
            ret = pool.map(processing_function, subsets)

            # Concat output from different Workers
            ret_data = pd.concat(ret)

            # Merge "cleanedReviews" column back into the original DataFrame
            # Note: indices in processing_data are preserved from movie_data,
            # so ret_data.index aligns correctly with movie_data.index
            movie_data.loc[ret_data.index, "cleanedReviews"] = ret_data

            # Write data to file
            movie_data.to_json(output_folder / f"rt_{Review_Type.lower()}_reviews_preprocessed_{i}.json", date_format="iso", orient="records", indent=2)

            # Clear variables
            del ret, ret_data, subsets
            gc.collect()

            end_time = time.time()
            # Display File Time
            if timing:
                elapsed = end_time-start_time
                print(f"[✓] Done! Runtime File {i}: {int(elapsed//60)}:{elapsed%60:05.2f}")
    return None
