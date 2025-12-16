import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch
import re, gc


def CalculateEmbeddings(Review_Type):


    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Check for valid Review_Type/Analysis_Type argument. 
    if Review_Type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review Type: {Review_Type}. Must be one of {VALID_REVIEW_TYPES}")
    
    # Set up paths for reading / writing data
    folder = PROJECT_ROOT / "Rotten Tomatoes Reviews" / f"{Review_Type} Reviews Preprocessed for NLP"
    output_folder = PROJECT_ROOT / "NLP Data" / f"{Review_Type} Embeddings"
    output_folder.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Looking in folder:", folder)
    print("Glob pattern:", f"rt_{Review_Type.lower()}_reviews_preprocessed_*.json")

    # List of files to process, sorted to maintain order
    json_files = sorted(folder.glob(f"rt_{Review_Type.lower()}_reviews_preprocessed_*.json"), key=lambda x: int(re.search(r"_(\d+)\.json$", x.name).group(1)))

    # Process files
    for i, file_path in enumerate(json_files):

        # Skip already processed files
        output_path = output_folder / f"rt_{Review_Type.lower()}_embeddings_{i}.parquet"

        # Skip already processed files
        if output_path.exists():
            print(f"[✓] Skipping File {i} — already completed.")
            continue

        print(f"[→] Processing file {i}/{len(json_files)-1}: {file_path.name}, calculating embeddings…")

        # Load Data
        data = pd.read_json(file_path)
        docs = data["cleanedReviews"].to_list()

        # Calculate embeddings
        embeddings = embedding_model.encode(docs, batch_size=256, show_progress_bar=True)

        # Store in DataFrame
        data["embeddings"] = embeddings.tolist()
        data = data[["id", "reviewId", "embeddings"]]

        # Write to file
        data.to_parquet(output_path, index=False)

        # Clean up
        del embeddings, data, docs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return None