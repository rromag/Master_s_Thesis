import requests
import time
import random



def XHR_RTScraper(args):
    """
    Scrape audience or critic reviews from Rotten Tomatoes using the site's XHR API.

    This function takes a DataFrame of movie slugs and corresponding `emsId`s, then 
    queries Rotten Tomatoes' review API endpoint to collect reviews. Reviews are 
    fetched page by page using the `endCursor` pagination token until the desired 
    number of reviews is collected or no further pages are available. Rotating 
    user-agent headers are used to reduce the chance of being blocked.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - movie_slug_data (pd.DataFrame): DataFrame with at least two columns:
              * 'slug' (str): the Rotten Tomatoes movie slug used in URLs.
              * 'emsId' (str): the Rotten Tomatoes EMS identifier used in API calls.
        - max_reviews (int): Maximum number of reviews to scrape per movie.
        - Review_Type (str): Type of reviews to scrape, must be one of:
              * "Audience" → user reviews
              * "Critic"   → critic reviews

    Returns
    -------
    all_reviews : list of dict
        A list of review objects (dicts) as returned by the Rotten Tomatoes API, 
        with an additional `"id"` field identifying the movie slug.

    Notes
    -----
    - Handles API pagination via the `"after"` token in `pageInfo`.
    - Introduces random delays and rotates headers to avoid request throttling.
    - Will stop early if fewer than `max_reviews` reviews exist for a given movie.
    - In case of request errors (e.g., 429 rate limits), retries with delays.

    Raises
    ------
    ValueError
        If `Review_Type` is not "Audience" or "Critic".
    """
    # Unpack Arguments
    movie_slug_data, max_reviews, Review_Type = args

    if Review_Type == "Critic":
        review_type = "all"
    elif Review_Type == "Audience":
        review_type = "user"
    else:
        raise ValueError("Review type must be 'Audience' or 'Critic'")
        # Initialize list for Review Collection
    all_reviews = []
        # Iterate over movies to collect Reviews for each one
    for _, row in movie_slug_data.iterrows():
        movie_slug = row["slug"]
        movie_id = row["emsId"]

        # Base URL and Rotating User-Agents/headers
        base_url = f"https://www.rottentomatoes.com/cnapi/movie/{movie_id}/reviews/{review_type}"
        headers = [{'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36','Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8','Referer': f'https://www.rottentomatoes.com/m/{movie_slug}/reviews?type=user'}, 
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36','Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8','Referer': f'https://www.rottentomatoes.com/m/{movie_slug}/reviews?type=user'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15','Accept-Language': 'en-GB,en;q=0.9','Referer': f'https://www.rottentomatoes.com/m/{movie_slug}/reviews?type=user'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:139.0) Gecko/20100101 Firefox/139.0','Accept-Language': 'en-US,en;q=0.5','Referer': f'https://www.rottentomatoes.com/m/{movie_slug}/reviews?type=user'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Gecko/20100101 Firefox/128.0','Accept-Language': 'en-US,en;q=0.5','Referer': f'https://www.rottentomatoes.com/m/{movie_slug}/reviews?type=user'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36','Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8','Referer': f'https://www.rottentomatoes.com/m/{movie_slug}/reviews?type=user'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36','Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8','Referer': f'https://www.rottentomatoes.com/m/{movie_slug}/reviews?type=user'}]


        # Initial parameters
        params = {"pageCount": 20}
        seen_after_tokens = set()
        collected = 0
        per_movie_reviews = []

        while collected < max_reviews:                                                                  # Loop Review Collection until desired number of Reviews have been collected
            try:
                response = requests.get(base_url, headers=random.choice(headers), params=params)        # Connect to URL
    
                if response.status_code == 429:
                    print(f"Rate limit hit for {movie_slug}. Sleeping for 60 seconds.")
                    time.sleep(60)
                    continue

                if response.status_code != 200:
                    print(f"{movie_slug} Error: {response.status_code}")
                    break

                data = response.json()                                                                  # Save Data to a variable for processing
            
            except requests.exceptions.RequestException as e:
                print(f"{movie_slug} Request failed: {e}")
                break

            except ValueError:
                print(f"{movie_slug} Error: Failed to parse JSON.")
                continue

            reviews = data.get("reviews", [])                                                           # Extract reviews, put them into list

            if not reviews:
                print(f"No more reviews found for {movie_slug}.")                                       # Break if no more reviews
                break

            for review in reviews:                                                                      # Add Movie Id (slug) to each Review to Identify the movie, add page reviews to the previous movie reviews
                if collected >= max_reviews:
                    break
                review["id"] = movie_slug                                           
                per_movie_reviews.append(review)
                collected += 1

            print(f"Fetched {len(reviews)} reviews for {movie_slug}. Total for this movie: {collected}")

            after = data.get("pageInfo", {}).get("endCursor")                                           # Get "after" parameter to connect to the next Page of Reviews
            if not after or after in seen_after_tokens:
                print(f"No more pages or duplicate cursor for {movie_slug}. Done.")                     # Break if no more Reviews Pages
                break

            seen_after_tokens.add(after)                                                                # Add after token to already used after tokens
            params["after"] = after                                                                     # Assign new after token for next Review Page

            time.sleep(1 + random.random())                                                             # Chill

        all_reviews.extend(per_movie_reviews)                                                           # Add movie reviews to list of all reviews

    return all_reviews
