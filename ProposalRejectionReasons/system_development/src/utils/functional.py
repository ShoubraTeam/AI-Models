# --------------------------------------------------------------------------------------
# Utility Functions (if needed) [loading data -- plotting -- ...]
# --------------------------------------------------------------------------------------

import json

def load_json(file_path: str, encoding: str = "utf-8"):
    """
    Loads a JSON file & returns its content

    Args:
        file_path (str): the path to the file
        encoding  (str): the encoding method

    Returns:
        content: the JSON file content
    """
    with open(file = file_path, mode = 'r', encoding = encoding) as f:
        return json.load(f)
# ------------------------------------------------------------------------------------
    
