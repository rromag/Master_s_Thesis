import pandas as pd
import numpy as np
import time
import re, gc
from pathlib import Path
import multiprocessing
from Translation.DetectLanguage import DetectLanguage
from Translation.MovieReviewTranslatorGoogle import MovieReviewTranslatorGoogle
from Translation.MovieReviewTranslatorDeepl import MovieReviewTranslatorDeepl



def TranslateMovieReview(Review_Type, Free=True, chunk_size = 100, num_cores = 5, timing = True):
    """
    Translate all non-English Rotten Tomatoes movie reviews into English.

    This function processes pre-translated review files, detects the language of each 
    review, and translates non-English reviews into English using either the Google 
    Translate API wrapper (free) or the DeepL API (paid). Reviews are processed in 
    parallel and in chunks to improve efficiency and scalability.

    Parameters
    ----------
    Review_Type : {"Audience", "Critic"}
        Type of reviews to process. Must be either "Audience" or "Critic".
    Free : bool, default=True
        If True, use Google Translate API wrapper (free). If False, use DeepL API.
    chunk_size : int, default=100
        Number of reviews to process per batch during translation.
    num_cores : int, default=5
        Number of CPU cores to use for parallel processing.
    timing : bool, default=True
        If True, print runtime information for each processed file and chunk.

    Returns
    -------
    None
        The function writes translated review files to the appropriate output folder.
        No value is returned.

    Workflow
    --------
    1. Load raw review files from the `pre Translation` folder.
    2. Detect the language of each review in parallel using `DetectLanguage`.
    3. Select only non-English and non-"unknown" reviews for translation.
    4. Translate reviews in parallel, processing them in `chunk_size` batches.
    5. Merge translated reviews back into the dataset, preserving the original text 
       in a new `originalReview` column.
    6. Save the translated dataset as a JSON file in the `Translated` folder.

    Notes
    -----
    - Files that are already translated are skipped.
    - Empty translation sets (files with only English reviews) are saved without 
      translation for completeness.
    - Column ordering differs slightly between Audience and Critic review files.
    - This function depends on the folder structure of the project:
        Rotten Tomatoes Reviews/
            ├── Audience Reviews pre Translation/
            └── Audience Reviews Translated/
            ├── Critic Reviews pre Translation/
            └── Critic Reviews Translated/
    - Heavy reliance on parallelization: performance scales with `num_cores`.
    - Memory-intensive when processing large review datasets; garbage collection 
      is forced after each file.
    """

    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    if Free:
        translator=MovieReviewTranslatorGoogle
    else:
        translator=MovieReviewTranslatorDeepl

    # Check for valid Review_Type/Analysis_Type argument.   
    if Review_Type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review Type: {Review_Type}. Must be one of {VALID_REVIEW_TYPES}")

    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
 
    # Setup paths for reading / writing data
    folder = Path(f"{PROJECT_ROOT}/Rotten Tomatoes Reviews/{Review_Type} Reviews pre Translation")
    output_folder = Path(f"{PROJECT_ROOT}/Rotten Tomatoes Reviews/{Review_Type} Reviews Translated")
    output_folder.mkdir(parents=True, exist_ok=True)

    # List of files to process, sorted to maintain order
    json_files = sorted(folder.glob(f"rt_{Review_Type.lower()}_reviews_pre_translation_*.json"), key=lambda x: int(re.search(r"_(\d+)\.json$", x.name).group(1)))

    # Process files
    for i, file_path in enumerate(json_files):
        
        # Load movie review data and initialize variables for processing and saving
        movie_data = pd.read_json(file_path)
        output_path = output_folder / f"rt_{Review_Type.lower()}_reviews_translated_{i}.json"
        chunks = []

        # Skip already processed files
        if output_path.exists():
            print(f"[✓] Skipping File {i} — already completed.")
            continue

        # Find reviews to translate in parallel
        split_data = [*np.array_split(movie_data, num_cores)]
        with multiprocessing.Pool(num_cores) as pool:
            res = pool.map(DetectLanguage, split_data)
        movie_data = pd.concat(res)

        # Select Reviews with non-english reviews for translation
        reviews_to_translate = movie_data[(movie_data["language"] != "en") & (movie_data["language"] != "unknown")].copy()      

        # Skip empty translations
        if reviews_to_translate.empty:
            print(f"[✓] File {i} has no non-English reviews — skipping translation.")
            movie_data.to_json(output_path, orient="records", indent=2)
            continue

        # Track File being processed and the time it takes
        print(f"[→] Translating file {i}/{len(json_files)-1}: {file_path.name}. Number of Reviews to translate: {reviews_to_translate.shape[0]}")
        start_time = time.time()

        # Calculate number of total chunks to keep track of progress
        num_chunks =int(np.ceil(len(reviews_to_translate)/chunk_size))

        # Processing the data in chunks
        with multiprocessing.Pool(num_cores) as pool:
            for n, start_idx in enumerate(range(0, len(reviews_to_translate), chunk_size)):
                chunk_start_time = time.time()

                # Create subsets to process file in batches
                subset = reviews_to_translate.iloc[start_idx:start_idx+chunk_size].copy()
                if subset.empty:
                    continue
                # Track progress within File processing
                print(f"Processing Chunk {n+1}/{num_chunks}")

                # Split non-english reviews in for parallelization
                split_chunk = np.array_split(subset, num_cores)

                # Translate non-english reviews in parallel
                ret_chunk = pool.map(translator, split_chunk)

                # Collect results
                chunks.append(pd.concat(ret_chunk))

                chunk_end_time = time.time()
                # Display chunk runtime
                if timing:
                    print(f"Runtime Chunk {n+1}: {chunk_end_time-chunk_start_time:.2f} seconds")
        
        # Concat chunk results into batch results
        translations = pd.concat(chunks)

        # Merge translated reviews back into the dataset
        movie_data["originalReview"] = movie_data["reviewText"]
        movie_data.loc[translations.index, "reviewText"] = translations

        # Reorder Columns
        if Review_Type == "Audience":
            cols = ["id", "reviewId", "title", "creationDate", "userId", "reviewText", "originalReview", "language", "ratingOutOfTen", "originalRating"]
        elif Review_Type == "Critic":
            cols = ["id", "reviewId", "title", "creationDate", "criticName", "reviewText", "originalReview", "language", "ratingOutOfTen", "originalRating", "reviewState"]
        movie_data = movie_data[cols]

        # Write data to file
        movie_data.to_json(output_path, date_format="iso", orient="records", indent=2)

        # Clear variables
        del translations, chunks, movie_data, reviews_to_translate
        gc.collect()

        end_time = time.time()
        # Display batch time
        if timing:
            elapsed = end_time-start_time
            print(f"[✓] Done! Runtime File {i}: {int(elapsed//60)}:{elapsed%60:05.2f}")

    return None
