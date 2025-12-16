from NLP_Analysis.MovieReviewEmotionDetection import MovieReviewEmotionDetection
from NLP_Analysis.MovieReviewSentimentAnalyser import MovieReviewSentimentAnalyser
from NLP_Analysis.MovieReviewArgumentDetection import MovieReviewArgumentDetection
from NLP_Analysis.MovieReviewAspectExtraction import MovieReviewAspectExtraction
import pandas as pd
import numpy as np
import torch
import time
import re, gc
from pathlib import Path


def NLPAnalysis(Review_Type, Analysis_Type, chunk_size = 1000, num_threads = 8, timing = True):
    """
    Perform NLP-based analysis (sentiment, emotion, arguments, or aspects) on 
    Rotten Tomatoes reviews.

    This function applies the specified NLP analysis model to either Audience 
    or Critic reviews in parallelized chunks. It supports multiple analysis 
    types (sentiment analysis, emotion detection, argument mining, and aspect-based 
    sentiment analysis) and saves the processed results to structured JSON files.

    Parameters
    ----------
    Review_Type : {"Audience", "Critic"}
        Type of reviews to analyze. Must be either "Audience" or "Critic".
    Analysis_Type : {"sentiment", "emotion", "argument", "aspects", "topics"}
        The type of analysis to run on the reviews. 
        - "sentiment" : Predict sentiment polarity (positive/negative/neutral).
        - "emotion" : Detect emotional categories in the text.
        - "argument" : Identify argumentative structures.
        - "aspects" : Perform aspect-based sentiment analysis.
    chunk_size : int, default=1000
        Number of reviews to process in memory at once. Helps prevent memory overflow.
    num_threads : int, default=8
        Number of CPU threads to allocate for model execution.
    timing : bool, default=True
        If True, prints runtime statistics for each processed chunk and file.

    Returns
    -------
    None
        The function writes processed JSON files to the output folder and 
        does not return a value.

    Workflow
    --------
    1. Validate input arguments (`Review_Type` and `Analysis_Type`).
    2. Assign the appropriate NLP analysis function based on `Analysis_Type`.
    3. Load review data from the `Clean` folder.
    4. Split reviews into chunks of size `chunk_size` for processing.
    5. Apply the analysis function to each chunk, running on `num_threads` CPU threads.
    6. Collect and merge chunk results into a single dataset.
    7. Save the analyzed reviews as JSON in the `NLP Data` folder, maintaining 
       consistent file numbering.

    Notes
    -----
    - Files that have already been processed are skipped automatically.
    - The function is memory-conscious by chunking and garbage collecting after each file.
    - Output structure depends on the analysis type and model used.
    - The function assumes the following folder structure exists:
        Rotten Tomatoes Reviews/
            ├── Audience Reviews Clean/
            └── Critic Reviews Clean/
        NLP Data/
            ├── Audience Sentiment Data/
            ├── Audience Emotion Data/
            ├── Critic Sentiment Data/
            └── Critic Emotion Data/
    """
    ANALYSIS_FUNCTIONS = {"sentiment": MovieReviewSentimentAnalyser,
                          "emotion": MovieReviewEmotionDetection,
                          "argument": MovieReviewArgumentDetection,
                          "aspects": MovieReviewAspectExtraction}
    
    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Assign analysis function to use
    analysisFunction = ANALYSIS_FUNCTIONS.get(Analysis_Type.lower())

    # Check for valid Review_Type/Analysis_Type argument. 
    if analysisFunction is None:
        raise ValueError(f"Unsupported Analysis_Type: {Analysis_Type}")
    
    if Review_Type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review Type: {Review_Type}. Must be one of {VALID_REVIEW_TYPES}")
    
    # Set number of threads when running on CPU
    torch.set_num_threads(num_threads)
    
    # Set up paths for reading / saving the data
    folder = PROJECT_ROOT / "Rotten Tomatoes Reviews" / f"{Review_Type} Reviews Preprocessed for NLP"
    output_folder = PROJECT_ROOT / "NLP Data" / f"{Review_Type} {Analysis_Type.capitalize()} Data"
    output_folder.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Looking in folder:", folder)
    print("Glob pattern:", f"rt_{Review_Type.lower()}_reviews_preprocessed_*.json")

    # List of files to process, sorted to maintain order
    json_files = sorted(folder.glob(f"rt_{Review_Type.lower()}_reviews_preprocessed_*.json"), key=lambda x: int(re.search(r"_(\d+)\.json$", x.name).group(1)))

    # process files using the MultiThreadingNLP and SentimentAnalysis / EmotionDetection Functions
    for i, file_path in enumerate(json_files):

        # Initialize variables
        movie_data = pd.read_json(file_path)
        chunks = []
        output_path = output_folder / f"rt_{Review_Type.lower()}_reviews_{Analysis_Type}_{i}.json"

        # Skip already processed files
        if output_path.exists():
            print(f"[✓] Skipping File {i} — already completed.")
            continue

        print(f"[→] Processing file {i}/{len(json_files)-1}: {file_path.name}, doing {Analysis_Type.capitalize()} analysis on {num_threads} threads…")
        start_time = time.time()

        # Calculate number of total chunks to keep track of progress
        num_chunks =int(np.ceil(len(movie_data)/chunk_size))

        # Processing the data in chunks to free up memory
        for n, start_idx in enumerate(range(0, len(movie_data), chunk_size)):
            chunk_start_time = time.time()

            subset = movie_data.iloc[start_idx:start_idx+chunk_size].copy()

            if subset.empty:
                continue

            print(f"Processing Chunk {n+1}/{num_chunks}")

            # Multithreading NLP analysis
            ret_chunk = analysisFunction(subset)

            # Collect results
            chunks.append(ret_chunk)

            chunk_end_time = time.time()
            # Display Chunk Runtime
            if timing:
                print(f"Runtime Chunk {n+1}: {chunk_end_time-chunk_start_time:.2f} seconds")

        # Concat chunk results into batch result
        result = pd.concat(chunks, ignore_index=True)

        # Write data to file
        result.to_json(output_path, orient = "records", indent=2)

        # Clear variables to free up memory
        del result, chunks, movie_data
        gc.collect()

        end_time = time.time()
        # Display batch time
        if timing:
            elapsed = end_time-start_time
            print(f"[✓] Done! Runtime File {i}: {int(elapsed//60)}:{elapsed%60:05.2f}")

    return None

