import re

def normalize_text(text):
    """
    Lowercase, remove punctuation except numbers, letters, spaces, 
    and square brackets (so placeholders like [actor] survive).
    """
    # Remove all non-alphanumeric characters except spaces
    text = re.sub(r"[^a-z0-9\s\[\]]", "", text.lower())
    # Optionally, remove commas inside numbers: "10,000" -> "10000"
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return text
