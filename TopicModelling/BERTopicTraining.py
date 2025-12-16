from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import time




def BERTopicTraining():
    """
    Train a BERTopic model on movie review aspects extracted from audience and critic reviews.

    The function performs the following steps:
    1. Loads JSON files containing audience and critic review aspects.
    2. Combines the data into a single DataFrame.
    3. Flattens and cleans the "aspect" column by removing empty entries.
    4. Deduplicates aspect strings to improve training efficiency.
    5. Computes embeddings for the unique, non-empty aspect strings using SentenceTransformer.
    6. Trains a BERTopic model with a KeyBERT-inspired representation.
    7. Saves the trained model, training documents, embeddings, and topic assignments to disk.

    Notes
    -----
    - Empty aspect strings are excluded from training.
    - The function is currently hardcoded to load 25 audience and 20 critic JSON files.
    - The "all-MiniLM-L6-v2" SentenceTransformer model is used for embeddings.
    - The following artifacts are saved in the "TopicModelling" folder at the project root:
        * `bertopic_aspects_model/` : The trained BERTopic model.
        * `BERTopic_aspects_model_training_data.csv` : Training documents (aspect strings).
        * `embeddingsTopicModel.npy` : Precomputed embeddings for the training documents.
        * `topicsTopicModel.npy` : Topic assignments for the training documents.

    Returns
    -------
    None
        Results are written to disk.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    start_time = time.time()

    # Load Data
    audience_data = pd.concat([pd.read_json(PROJECT_ROOT / f"NLP Data/Audience Aspects Data/rt_audience_reviews_aspects_{i}.json") for i in range(25)])
    critic_data = pd.concat([pd.read_json(PROJECT_ROOT / f"NLP Data/Critic Aspects Data/rt_critic_reviews_aspects_{i}.json") for i in range(20)])

    # Flatten aspects into a list (faster than explode)
    all_aspects = []

    for aspects in pd.concat([audience_data, critic_data], ignore_index=True)["aspect"]:
        if isinstance(aspects, list):
            all_aspects.extend(aspects)

    # Clean and filter aspects
    train_docs = []
    for aspect in all_aspects:
        text = str(aspect).strip()
        if text:
            train_docs.append(text)

    print(f"Number of Aspects for Training: {len(train_docs)}")

    # Calculate embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)     # device="cuda" for GPU
    
    embeddings = embedding_model.encode(train_docs, batch_size=1024, show_progress_bar=True)

    # Fit BERTopic
    representation_model = KeyBERTInspired()
    topic_model = BERTopic(representation_model=representation_model, embedding_model=embedding_model)

    topics, probs = topic_model.fit_transform(train_docs, embeddings=embeddings)

    # Save the BERTopic Model
    topic_model.save(PROJECT_ROOT / "TopicModelling" / "bertopic_aspects_model")

    # Save training data for future topic reduction
    pd.Series(train_docs, name = "aspectString").to_csv(PROJECT_ROOT / "TopicModelling/BERTopic_aspects_model_training_data.csv", index=False)

    # Save embeddings for later use
    np.save(PROJECT_ROOT / "TopicModelling/aspect_model_embeddings.npy", embeddings)

    # Save topics for later use
    np.save(PROJECT_ROOT / "TopicModelling/aspect_model_topics.npy", topics)

    end_time = time.time()

    print(f"Runtime: {end_time-start_time}")

    return None