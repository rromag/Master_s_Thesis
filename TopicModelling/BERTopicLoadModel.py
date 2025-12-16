from bertopic import BERTopic
import numpy as np
import pandas as pd
from pathlib import Path


def BERTopicLoadModel(ModelFile="bertopic_aspects_model"):
    """
    Reload the full BERTopic setup: trained model, training documents, embeddings, 
    and saved topic assignments.

    This function restores all components necessary to continue working with a 
    previously trained BERTopic model without needing to re-run expensive 
    computations (such as embeddings or topic inference).

    Returns
    -------
    topic_model : BERTopic
        The trained BERTopic model reloaded from disk.

    train_docs : list of str
        The training documents used to fit the model. Loaded from CSV.

    embeddings : np.ndarray of shape (n_docs, embedding_dim)
        The precomputed document embeddings used during training.

    topics : np.ndarray of shape (n_docs,)
        The topic assignments for each training document, as produced by the model 
        when it was last saved.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Reload model
    topic_model = BERTopic.load(PROJECT_ROOT / "TopicModelling" / f"{ModelFile}")

    # Reload training docs
    train_docs = pd.read_csv(PROJECT_ROOT / "TopicModelling" / "BERTopic_aspects_model_training_data.csv")["aspectString"]
    # Typecast as string
    train_docs = train_docs.fillna("").astype(str).to_list()

    # Reload embeddings
    embeddings = np.load(PROJECT_ROOT / "TopicModelling" / "aspect_model_embeddings.npy", allow_pickle=False)

    # Reload topics
    topics = np.load(PROJECT_ROOT / "TopicModelling/aspect_model_topics.npy", allow_pickle=True)

    return topic_model, train_docs, embeddings, topics
