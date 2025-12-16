

def split_actor_name(full_name):
    """
    Split an actor's full name into first name and last name, keeping
    important multi-word last names (e.g., 'van Damme', 'De Niro') and
    dropping middle names like 'John' or single-letter middles like 'J.'.
    """
    parts = full_name.split()

    # Regular first name, last name
    if len(parts) == 2:
        return (parts[0],parts[1])

    # Handle single name entries
    if len(parts) == 1:
        return (parts[0], "")
    
    # Check for "jr" suffix
    if parts[-1] == "jr":
        last = " ".join(parts[:1])
        return (parts[0], last)
    
    # Keep common short middle words for multi-word last names
    short_middle_words = {"de", "van", "von", "da", "di"}

    # If middle words are short, include them in last name
    middle = parts[1:-1]
    last_parts = [p for p in middle if p.lower() in short_middle_words] + [parts[-1]]

    return (parts[0], " ".join(last_parts))
