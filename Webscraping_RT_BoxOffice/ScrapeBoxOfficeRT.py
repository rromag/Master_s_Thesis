import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np



def ScrapeBoxOfficeRT(movie_list):
    """
    Scrape box office earnings for a list of movies from Rotten Tomatoes.

    For each Rotten Tomatoes movie ID in `movie_list`, this function sends an HTTP
    request to the corresponding movie page, parses the "Movie Info" section, and
    extracts the reported box office revenue (if available). Results are returned
    in a pandas DataFrame.

    Parameters
    ----------
    movie_list : list of str
        List of Rotten Tomatoes movie IDs (slugs) to scrape. 
        Example: ["inception", "the_dark_knight"].

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
            - "id" : str
                The Rotten Tomatoes movie ID.
            - "BoxOffice" : str or NaN
                The scraped box office revenue (e.g., "$292.6M"), 
                or NaN if unavailable.

    Notes
    -----
    - Uses rotating User-Agent headers to reduce the likelihood of request blocking.
    - If a request fails or box office data is not found, the value for that movie
      is recorded as NaN.
    - This scraper is dependent on Rotten Tomatoes' HTML structure, which may change
      over time and break parsing.
    """
    # User-Agents and headers to rotate through to reduce chance to get blocked by RT
    headers_mod=np.array([{'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36','Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8','Referer': 'https://google.com'}, 
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36','Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8','Referer': 'https://google.com'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15','Accept-Language': ' en-GB,en;q=0.9','Referer': 'https://google.com'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:139.0) Gecko/20100101 Firefox/139.0','Accept-Language': 'en-US,en;q=0.5','Referer': 'https://google.com'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Gecko/20100101 Firefox/128.0','Accept-Language': 'en-US,en;q=0.5','Referer': 'https://google.com'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36','Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8','Referer': 'https://google.com'},
                {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36','Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8','Referer': 'https://google.com'}])

    # Initialize DataFrame to collect scraped box office data for all movies
    movie_box_office = pd.DataFrame(columns=["id", "BoxOffice"])

    for movie in movie_list:
        # Assemble Rotten Tomatoes URL for the movie
        url = "https://www.rottentomatoes.com/m/" + movie

        # Send request using rotating User-Agent headers
        response = requests.get(url, np.random.choice(headers_mod))
        boxOffice = np.nan

        # Check if request was successful, proceed with scraping if True
        if response.status_code == 200:
            html_content = response.text
        else:
            print(movie, "Failed to retrieve content):", {response.status_code})
            continue    # Skip parsing if request failed

        # Use BeautifulSoup to parse HTML page
        soup = BeautifulSoup(html_content, "html.parser")

        # Parse "Movie Info"-table to find Box Office Entry (Table length is variable)
        movie_info_table = soup.find_all("div",class_ = "category-wrap")
        for idx, box in enumerate(movie_info_table):
            if "Box Office" in box.find('rt-text', {'data-qa': 'item-label'}).text:
                # Extrac Box Office value from the matching row
                boxOffice = movie_info_table[idx].find('rt-text', {'data-qa': 'item-value'}).text
                break
        
        # Record movie ID and scraped box office in DataFrame
        movie_box_office.loc[movie_box_office.shape[0]] = [movie, boxOffice]

    return movie_box_office                                                                                                                    
