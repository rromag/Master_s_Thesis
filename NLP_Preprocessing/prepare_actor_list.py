import pandas as pd
from NLP_Preprocessing.normalize_text import normalize_text

# Load datasets
celebrities = pd.read_csv("NLP_Preprocessing/Celebrity.csv", index_col=0)
top_1000 = pd.read_csv("NLP_Preprocessing/IMDb_top_1000_actors.csv")["Name"]

# Select top 2000 actors from the celebrity dataset
actors = celebrities[celebrities["known_for_department"] == "Acting"].head(2000)["name"].reset_index(drop=True)

# Normalize text (lowercase, remove punctuation, etc.)
actors = actors.apply(normalize_text)
top_1000 = top_1000.apply(normalize_text)

# Remove IMDb actors already in the celebrity list
actor_set = set(actors)
new_actors = top_1000[~top_1000.isin(actor_set)].reset_index(drop=True)

# Merge the two lists
merged_actors = pd.concat([actors, new_actors]).reset_index(drop=True)
merged_actors = merged_actors[merged_actors.str.strip() != ""]  # remove empty strings
merged_actors.dropna(inplace=True)
merged_actors.reset_index(inplace=True, drop=True)

# Save to CSV
merged_actors.to_csv("NLP_Preprocessing/Actor_List.csv")
