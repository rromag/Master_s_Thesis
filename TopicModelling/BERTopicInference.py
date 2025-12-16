from bertopic import BERTopic
from TopicModelling.BERTopicLoadModel import BERTopicLoadModel
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re, gc
import time




def BERTopicInference(Review_Type, ModelFile="bertopic_aspects_model_tuned", timing=True):
    """
    Run BERTopic inference on extracted review aspects and save results with topic assignments.

    This function loads all JSON files containing extracted aspects and their metadata 
    (per review) for either Audience or Critic reviews. It then performs the following steps:

    1. Reads each file and explodes aspect/sentiment lists into individual rows.
    2. Runs the trained BERTopic model to assign a topic and topic probability to each aspect.
    3. Maps topics to their human-readable labels (if available in the model).
    4. Writes the enriched data (including topics, labels, and probabilities) back to JSON 
       in a separate output folder, while preserving review metadata.

    Parameters
    ----------
    Review_Type : str
        Must be either "Audience" or "Critic". Determines which dataset to process.
    timing : bool, default=True
        If True, prints per-file runtime information.

    Input
    -----
    - JSON files located in:
      NLP Data/{Review_Type} Aspects Data/
      with filenames of the form: rt_{review_type}_reviews_aspects_{i}.json

    Output
    ------
    - JSON files saved to:
      NLP Data/{Review_Type} Topic Data/
      with filenames of the form: rt_{review_type}_reviews_topics_{i}.json

      Each output file contains the following columns:
      - reviewId: Unique identifier of the review
      - sentence: The review sentence where the aspect was extracted
      - aspect: The extracted aspect text
      - sentiment: Sentiment polarity of the aspect
      - confidence: Confidence score of aspect extraction
      - topic: Numeric topic assigned by BERTopic
      - topic_label: Human-readable label of the topic (if assigned, else None)
      - topic_probability: Probability score of the assigned topic

    Notes
    -----
    - Aspects assigned to the outlier topic (`-1`) will not receive a probability score 
      and will have `topic_probability = None`.
    - Only topics with manually assigned labels (e.g. top-N topics) will show a 
      non-None `topic_label`.

    Returns
    -------
    None
        Results are written to disk.
    """

    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Check for valid Review_Type/Analysis_Type argument. 
    if Review_Type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review Type: {Review_Type}. Must be one of {VALID_REVIEW_TYPES}")
    
    topic_model = BERTopic.load(PROJECT_ROOT / f"TopicModelling/{ModelFile}")

    print(topic_model.get_topic_info().head(50))

    # Set up paths for reading / saving the data
    folder = PROJECT_ROOT / "NLP Data" / f"{Review_Type} Aspects Data"
    output_folder = PROJECT_ROOT / "NLP Data" / f"{Review_Type} Topic Data"
    output_folder.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Looking in folder:", folder)
    print("Glob pattern:", f"rt_{Review_Type.lower()}_reviews_aspects_*.json")

    # List of files to process, sorted to maintain order
    json_files = sorted(folder.glob(f"rt_{Review_Type.lower()}_reviews_aspects_*.json"), key=lambda x: int(re.search(r"_(\d+)\.json$", x.name).group(1)))

    # Get label mapping to write the topic label into the DataFrame
    label_map = topic_model.get_topic_info().set_index("Topic")["CustomName"].to_dict()

    # Process files one by one
    for i, file_path in tqdm(enumerate(json_files), desc=f"Processing {Review_Type} files."):

        # Inizialize output path
        output_path = output_folder / f"rt_{Review_Type.lower()}_reviews_topics_{i}.json"

        # Skip already processed files
        if output_path.exists():
            print(f"[✓] Skipping File {i} — already completed.")
            continue
        
        # Prepare aspect data
        aspect_data = pd.read_json(file_path)

        # Ensure lists are lists
        aspect_data["aspect"] = aspect_data["aspect"].apply(lambda x: x if isinstance(x, list) else [])
        aspect_data["sentiment"] = aspect_data["sentiment"].apply(lambda x: x if isinstance(x, list) else [])
        aspect_data["confidence"] = aspect_data["confidence"].apply(lambda x: x if isinstance(x, list) else [])

        # Replace empty lists with [""], then explode
        aspect_data["aspect"] = aspect_data["aspect"].apply(lambda x: x if x else [""])
        aspect_data["sentiment"] = aspect_data["sentiment"].apply(lambda x: x if x else [""])

        aspect_data = aspect_data.explode(["aspect", "sentiment", "confidence"], ignore_index=True)

        # Drop rows with empty aspects (after exploding)
        aspect_data = aspect_data[aspect_data["aspect"].astype(str).str.strip().astype(bool)]

        print(f"[→] Processing file {i}/{len(json_files)-1}: {file_path.name}, doing Topic Modelling…")
        start_time = time.time()

        # Run inference
        topics, probabilities = topic_model.transform(aspect_data["aspect"].to_list())

        # Write topics, topic labels and topic probability back into DataFrame
        aspect_data["topic"] = topics
        aspect_data["topic_probability"] = [prob if t != -1 else None for prob, t in zip(probabilities, topics)]
        aspect_data["topic_label"] = aspect_data["topic"].map(label_map)

        # Write Data to file
        aspect_data[["reviewId", "sentence", "aspect", "sentiment", "confidence", "topic", "topic_label", "topic_probability"]].to_json(output_path, date_format="iso", orient="records", indent=2)

        # Clear variables
        del aspect_data, topics, probabilities
        gc.collect()

        end_time = time.time()
        # Display file time
        if timing:
            elapsed = end_time-start_time
            print(f"[✓] Done! Runtime File {i}: {int(elapsed//60)}:{elapsed%60:05.2f}")
    
    return None







