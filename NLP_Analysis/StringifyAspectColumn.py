import pandas as pd
import re
from pathlib import Path

def StringifyAspectColumn(Review_Type):
    """
    Convert the 'aspect' column (a list of aspects) in movie review JSON files into a 
    single string column called 'aspectString'. 
    
    Each JSON file contains movie reviews with an 'aspect' column, where aspects 
    are stored as a list of strings. This function:
    
    1. Reads all review JSON files in the appropriate folder 
       (based on the provided Review_Type).
    2. Creates a new column 'aspectString' by joining the list of aspects 
       into a space-separated string.
    3. Normalizes terminology in 'aspectString' by replacing all occurrences 
       of the word "film" with "movie" (case-insensitive, whole-word match).
    4. Writes the modified DataFrame back to the same JSON file.

    Parameters
    ----------
    Review_Type : str
        The type of reviews to process. Must be one of {"Audience", "Critic"}.
        Determines which folder of JSON files is processed.

    Raises
    ------
    ValueError
        If Review_Type is not one of the supported values ("Audience" or "Critic").

    Notes
    -----
    - The transformation only affects the new 'aspectString' column.
    - Original 'aspect' lists are preserved in the files.
    - JSON files are overwritten with the modified data.
    """
    VALID_REVIEW_TYPES = {"Audience", "Critic"}

    # Find Workspace folder for relative paths to input/output folders & files
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Check for valid Review_Type argument. 
    if Review_Type not in VALID_REVIEW_TYPES:
        raise ValueError(f"Unsupported Review Type: {Review_Type}. Must be one of {VALID_REVIEW_TYPES}")
    
    # Set up paths for reading / writing the data
    folder = PROJECT_ROOT / "NLP Data" / f"{Review_Type} Aspects Data"

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Looking in folder:", folder)
    print("Glob pattern:", f"rt_{Review_Type.lower()}_reviews_aspects_*.json")

    # List of files to process, sorted to maintain order
    json_files = sorted(folder.glob(f"rt_{Review_Type.lower()}_reviews_aspects_*.json"), key=lambda x: int(re.search(r"_(\d+)\.json$", x.name).group(1)))

    for i, file_path in enumerate(json_files):
        # Read Data
        movie_data = pd.read_json(file_path)

        # Join Aspects for each Review
        movie_data["aspectString"] = movie_data["aspect"].apply(lambda aspects: " ".join(aspects) if isinstance(aspects, list) else "").str.replace(r"\bfilm\b", "movie", case=False, regex=True)

        # Write back to file
        movie_data.to_json(file_path, date_format="iso", orient="records", indent=2)
        print(f"✅ Processed: {file_path.name}")
