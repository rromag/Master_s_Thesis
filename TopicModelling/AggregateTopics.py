import pandas as pd
from pathlib import Path




def AggregateTopics(Review_Type):
    """
    Aggregate topic counts for a given review type (Audience or Critic).

    This function loads all topic-assigned review files (previously generated 
    with BERTopic inference), merges them into a single DataFrame, and 
    computes the number of documents per topic. It also preserves the 
    first available topic label for each topic.

    Parameters
    ----------
    Review_Type : str
        Must be either "Audience" or "Critic". Determines which dataset 
        (Audience Topic Data or Critic Topic Data) is processed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per topic containing:
        - Topic (int): The numeric topic ID.
        - TopicLabel (str): The assigned human-readable label of the topic.
        - TopicCount (int): The number of documents assigned to that topic.

    Side Effects
    ------------
    - Reads JSON files from:
        NLP Data/{Review_Type} Topic Data/
        (expects filenames of the form rt_{review_type}_reviews_topics_{i}.json)
    - Writes an aggregated JSON file:
        NLP Data/{Review_Type} Topic Data/rt_{review_type}_topics_aggregated.json

    Notes
    -----
    - The number of input files is currently hardcoded:
        * Audience → 25 files
        * Critic → 20 files
    - Topics assigned as `-1` (outliers) will also appear in the aggregation.
    """
    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    num_files = {"Audience": 25, "Critic": 20}

    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Check for valid Review_Type/Analysis_Type argument. 
    if Review_Type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review Type: {Review_Type}. Must be one of {VALID_REVIEW_TYPES}")
    
    # Set up paths for reading / saving the data
    folder = PROJECT_ROOT / "NLP Data" / f"{Review_Type} Topic Data"

    # Read in Data
    topics_data = pd.concat([pd.read_json(folder / f"rt_{Review_Type.lower()}_reviews_topics_{i}.json") for i in range(num_files[Review_Type])], ignore_index=True)

    # Extract Topic Counts
    TopicCounts = topics_data.groupby("topic").agg(TopicLabel = ("topic_label", "first"),
                                                   TopicCount = ("topic", "count")).reset_index()
    TopicCounts.rename(columns={"topic": "Topic"}, inplace=True)

    # Write to file
    TopicCounts.to_json(folder / f"rt_{Review_Type.lower()}_topics_aggregated.json", orient="records", indent=2)

    return TopicCounts
