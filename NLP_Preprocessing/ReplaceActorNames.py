import pandas as pd
from rapidfuzz import fuzz, process
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from NLP_Preprocessing.normalize_text import normalize_text

stop_words = set(stopwords.words("english"))
stop_words_lower = {w.lower() for w in stop_words}


def ReplaceActorNames(MovieReviewDataFrame):
    """
    Mask actor names in movie reviews by replacing them with the placeholder '[actor]'.

    This function iterates over a pandas DataFrame of movie reviews, detects mentions of
    actor names using fuzzy string matching (RapidFuzz), and replaces them with a 
    standardized placeholder. It accounts for multi-word actor names and preserves 
    leading/trailing stopwords around matched names.

    Parameters
    ----------
    MovieReviewDataFrame : pandas.DataFrame
        A DataFrame containing movie reviews. Must include the following columns:
        - 'id': Unique identifier for each movie.
        - 'title': Movie title (used for grouping, not for matching).
        - 'reviewText': Text of the review in which actor names should be masked.

    Returns
    -------
    pandas.Series
        A Series indexed by the original review indices, containing the cleaned review 
        text with actor names replaced by '[actor]'.

    Notes
    -----
    - Actor names are loaded from 'NLP_Preprocessing/Actor_List.csv'. The CSV should
      contain a single column of normalized actor names (lowercased, punctuation removed).
    - Stopwords (from NLTK's English stopwords list) that occur at the start or end of
      matched names are preserved; only the actor name itself is replaced.
    - Full names are matched using fuzzy matching (RapidFuzz's fuzz.ratio). Only n-grams
      up to the length of the longest actor name are considered.
    - The function iterates over each movie individually for efficiency and maintains 
      the original review indices in the output.
    """
    # Initialize Variables used for finetuning
    threshold = 85
    len_tolerance = 0

    # Load actor names (already cleaned and lowercased when list was compiled)       
    actors = pd.read_csv("NLP_Preprocessing/Actor_List.csv", index_col=0)           # Top 2000 Actors from the top 10000 celebrities database off of kaggle (Celebrity.csv) supplemented with IMDb_top_1000_actors.csv (total of 2515 actors)
    actors = actors.iloc[:, 0].tolist()    

    # Compute max n-gram for fuzzy matching
    max_actor_len = max(len(a.split()) for a in actors) + len_tolerance

    # Drop duplicate movies to iterate through each movie once
    movies = MovieReviewDataFrame.drop_duplicates("id")[["id", "title"]]
    
    # Dictionary to store cleaned reviews keyed by DataFrame index
    cleaned_reviews = {}

    # Loop over each unique movie
    for _, row in movies.iterrows():

        # Select reviews corresponding to the current movie
        reviews = MovieReviewDataFrame.loc[MovieReviewDataFrame["id"] == row["id"], "review"].apply(normalize_text)

        # Loop over each review for the current movie
        for review_idx, review in reviews.items():
            # Split review into lowercased words for comparison
            review_words = word_tokenize(review)
            cleaned_words = []
            i = 0

            # Loop through words in review
            while i < len(review_words):
                replaced = False
                    
                # Check n-grams starting from the largest down to 1
                for n in range(min(max_actor_len, len(review_words) - i), 0, -1):
                    ngram_words = review_words[i:i+n]
                    ngram = " ".join(ngram_words)

                    # Compute fuzzy match score for all actors' full names
                    _, score, _ = process.extractOne(ngram, actors, scorer=fuzz.ratio)      # fuzz.ratio was found to perform better than fuzz.partial_ratio, cleaning first- and last-names individually was also found to lead to too many false positives

                    # If score exceeds threshold, replace n-gram
                    if score >= threshold:
                        start, end = 0, len(ngram_words)

                        # Trim stopwords at the start
                        leading_stopwords = []
                        while start < end and ngram_words[start] in stop_words_lower:
                            leading_stopwords.append(ngram_words[start])
                            start += 1
                        
                        # Trim stopwords at the end
                        trailing_stopwords = []
                        while start < end and ngram_words[end-1] in stop_words_lower:
                            end -= 1
                            trailing_stopwords.insert(0, ngram_words[end])      # insert at front to preserve order
                        
                        # Append in correct order: leading, actor placeholder, trailing
                        cleaned_words.extend(leading_stopwords)
                        if start < end:
                            cleaned_words.append("[actor]")
                        cleaned_words.extend(trailing_stopwords)

                        # Skip past matched words
                        i += n
                        replaced = True
                        break

                # If no n-gram was replaced, keep current word
                if not replaced:
                    cleaned_words.append(review_words[i])
                    i += 1
            
            # Save cleaned review while preserving the original index
            cleaned_reviews[review_idx] = " ".join(cleaned_words)
    
    # Return a pandas series, mapping review indices to cleaned review text
    return pd.Series(cleaned_reviews)

