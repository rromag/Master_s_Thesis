import pandas as pd
from NLP_Preprocessing.normalize_text import normalize_text
from rapidfuzz import fuzz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words("english"))
stop_words_lower = {w.lower() for w in stop_words}



def ReplaceMovieTitles(MovieReviewDataFrame):

    """
    Mask mentions of movie titles in reviews by replacing them with a placeholder '[movie]'.

    This function iterates over a pandas DataFrame containing movie reviews, identifies
    mentions of the movie title in each review using fuzzy string matching, and replaces
    the matched phrases with '[movie]'. Leading and trailing stopwords around a matched
    title are preserved. The function supports slight variations in spacing or minor 
    spelling differences due to the fuzzy matching.

    Parameters
    ----------
    MovieReviewDataFrame : pandas.DataFrame
        A DataFrame containing movie reviews. Must contain the following columns:
        - 'id': Unique identifier for each movie.
        - 'title': Movie title (used for matching in reviews).
        - 'review' or 'reviewText': The review text in which movie titles will be masked.

    Returns
    -------
    pandas.Series
        A Series indexed by the original review indices, containing the cleaned review
        text with movie titles replaced by '[movie]'.

    Notes
    -----
    - Review text and movie titles are normalized (lowercased, punctuation removed) 
      before matching.
    - Matching is performed using fuzzy string matching (RapidFuzz's `partial_ratio`) 
      to allow for minor spelling or formatting differences.
    - Stopwords (from NLTK's English stopwords list) at the start or end of matched 
      movie titles are preserved.
    - Processes each movie separately for efficiency and maintains the original review 
      indices in the output.
    - Progress for the outer loop over movies can be displayed using tqdm if desired.
    """

    # Initialize variables used for finetuning
    threshold = 85
    len_tolerance = 0

    # Drop duplicate movies to iterate through each movie once
    movies = MovieReviewDataFrame.drop_duplicates("id")[["id", "title"]]
    
    # Dictionary to store cleaned reviews keyed by DataFrame index
    cleaned_reviews = {}

    # Loop over each unique movie
    for _, row in movies.iterrows():

        # Clean Movie title
        movie_title = normalize_text(row["title"])

        # Select reviews corresponding to the current movie
        reviews = MovieReviewDataFrame.loc[MovieReviewDataFrame["id"] == row["id"], "review"].apply(normalize_text)

        # Determine max n-gram length for fuzzy matching
        max_ngram = len(movie_title.split()) + len_tolerance

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
                for n in range(min(max_ngram, len(review_words) - i), 0, -1):
                    ngram_words = review_words[i:i+n]
                    ngram = " ".join(ngram_words)

                    # Compute fuzzy match score
                    score = fuzz.partial_ratio(ngram, movie_title)          # For movie titles fuzz.partial_ratio performs pretty well, catches more than fuzz.ratio, not too many false positives
            
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
                        while end > start and ngram_words[end-1] in stop_words_lower:
                            end -= 1
                            trailing_stopwords.insert(0, ngram_words[end])  # insert at front to preserve order

                        # Append in correct order: leading, movie placeholder, trailing
                        cleaned_words.extend(leading_stopwords)
                        if start < end:
                            cleaned_words.append("[movie]")
                        cleaned_words.extend(trailing_stopwords)

                        # skip past matched words
                        i += n
                        replaced = True
                        break
        
                # If no n-gram was replaced, keep current word
                if not replaced:
                    cleaned_words.append(review_words[i])
                    i += 1
            
            # Save cleaned review while preserving the original index
            cleaned_reviews[review_idx] = " ".join(cleaned_words)

    # Return a pandas series, mapping review indices to cleaned review text.
    return pd.Series(cleaned_reviews)


