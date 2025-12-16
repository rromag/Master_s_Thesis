import pandas as pd
from Webscraping_RT_XHR.XHR_RTScraper import XHR_RTScraper
import multiprocessing
import numpy as np


def XHR_ParalleliseScraping(movie_slug_emsId, max_reviews, review_type, no_cores):
    """
    Parallelise the XHR_RTScraper function to scrape critic or audience reviews 
    for movies from rottentomatoes.com.
    """
    # Split Batch into Subsets for Parallelising
    split_data = np.array_split(movie_slug_emsId, no_cores)

    # Parallelise Review Scraper Function
    with multiprocessing.Pool(no_cores) as pool:
        results = pool.map(XHR_RTScraper, [(split, max_reviews, review_type) for split in split_data])

    # Flatten out the list of lists Returned by the Parallelising Function
    flattened = [review for sublist in results for review in sublist]

    return flattened

