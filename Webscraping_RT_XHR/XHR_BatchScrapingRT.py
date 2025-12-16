from Webscraping_RT_XHR.XHR_RTScraper import XHR_RTScraper
from Webscraping_RT_XHR.XHR_ParalleliseScraping import XHR_ParalleliseScraping
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os


def XHR_BatchScrapingRT(movie_data, max_reviews = 250, review_type = "Audience", no_of_batches = 155, no_cores = 5):
    """
    Scrape Rotten Tomatoes reviews in parallelized batches and save results to disk.

    This function coordinates large-scale scraping of critic or audience reviews
    from Rotten Tomatoes using the XHR API. The movie dataset is split into 
    batches, and each batch is processed with `XHR_ParalleliseScraping`, which 
    internally calls `XHR_RTScraper`. Results from each batch are serialized into
    JSON files, enabling resumable scraping (already completed batches are skipped).

    Parameters
    ----------
    movie_data : pd.DataFrame
        DataFrame containing Rotten Tomatoes movie identifiers with the following 
        required columns:
        - 'slug' (str): Rotten Tomatoes movie slug.
        - 'emsId' (str): Rotten Tomatoes EMS identifier.
    max_reviews : int, optional, default=250
        Maximum number of reviews to scrape per movie.
    review_type : {"Audience", "Critic"}, optional, default="Audience"
        Type of reviews to scrape:
        - "Audience" : user reviews.
        - "Critic"   : critic reviews.
    no_of_batches : int, optional, default=155
        Number of batches to split `movie_data` into. Each batch is saved separately.
    no_cores : int, optional, default=5
        Number of CPU cores to use for multiprocessing in each batch.

    Returns
    -------
    None
        The function has no return value. Results are saved to disk as JSON files.

    Side Effects
    ------------
    - Creates an output directory under:
      ``<project_root>/Rotten Tomatoes Reviews/<review_type> Reviews Scraped/``
    - Saves one JSON file per batch with scraped reviews. Example:
      ``rt_audience_reviews_scraped_batch_0.json``

    Notes
    -----
    - If a batch output file already exists, it is skipped automatically.
    - The function prints progress updates for each batch, including errors.
    - Designed for large datasets where full scraping in a single run is 
      impractical.

    See Also
    --------
    XHR_RTScraper : Scrapes reviews for a batch of movies.
    XHR_ParalleliseScraping : Distributes scraping across multiple cores.
    """

    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    # Check for valid Review_Type/Analysis_Type argument.   
    if review_type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review Type: {review_type}. Must be one of {VALID_REVIEW_TYPES}")

    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    print("Total Movies:", movie_data.shape[0])
    # Split movie data into batches
    data_batches = np.array_split(movie_data, no_of_batches)

    # Output Folder for batch results
    batch_folder = PROJECT_ROOT / "Rotten Tomatoes Reviews" / f"{review_type} Reviews Scraped"
    batch_folder.mkdir(parents=True, exist_ok=True)
    
    for idx, batch in enumerate(data_batches):
        batch_path = batch_folder / f"rt_{review_type.lower()}_reviews_scraped_batch_{idx}.json"
        # Skip Batch if output file already exists
        if batch_path.exists():
            print(f"[✓] Skipping Batch {idx} — already completed.")
            continue
        # Call XHR Parallelising Function for Batch
        print(f"[→] Processing Batch {idx} ...")
        try:
            result = XHR_ParalleliseScraping(batch, max_reviews=max_reviews, review_type=review_type, no_cores=no_cores)

            # Save Batch file
            with open(batch_path, "w", encoding="utf-8") as p:
                json.dump(result, p, ensure_ascii=False, indent=2)
        
            print(f"[✓] Finished Batch {idx}")
        except Exception as e:
            print(f"[✗] Error in Batch {idx}: {e}")
            continue
    return None

