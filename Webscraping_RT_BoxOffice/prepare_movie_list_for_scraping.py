import pandas as pd


def prepare_movie_list_for_scraping(movie_review_data = "Rotten Tomatoes reviews/rotten_tomatoes_critic_reviews.csv", min_critic_reviews = 100):
    """
    Filter movies with at least a minimum number of critic reviews.

    Reads a Rotten Tomatoes review dataset, counts the number of reviews per 
    movie, and returns a list of unique movie IDs that meet the 
    `min_critic_reviews` threshold.

    Parameters
    ----------
    movie_review_data : str, optional
        Path to the CSV file containing Rotten Tomatoes reviews. 
        Default is "Rotten Tomatoes reviews/rotten_tomatoes_critic_reviews.csv".
    min_critic_reviews : int, optional
        Minimum number of critic reviews required for a movie to be included. 
        Default is 100.

    Returns
    -------
    list of str
        List of unique movie IDs with at least `min_critic_reviews` reviews.
    """
    movies_data = pd.read_csv(movie_review_data)
    movies_data["NoOfReviews"] = movies_data.groupby("id")["id"].transform("count")
    movies_data_filtered = movies_data[movies_data["NoOfReviews"] >= min_critic_reviews].drop_duplicates(subset = "id").reset_index()
    movies = movies_data_filtered["id"].values.tolist()
    return movies

