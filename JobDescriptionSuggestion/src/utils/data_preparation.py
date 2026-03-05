# ---------------------------------------------------------------------------
# Contains the code required to load the cleaned saved .csv files in order
# to build a vector database from them
# 
# Ahmed Ragab
# ---------------------------------------------------------------------------

import pandas as pd

def load_csv(path: str):
    """
    Args:
        path (str): the path to the .csv file

    Returns:
        df (Pandas.DataFrame)
    """
    df = pd.read_csv(path)
    return df


# standardize CSVs
def standardize_csvs(paths: list):
    """
    Standardizes .csv files & concatenate them in one final .csv file
    Args:
        paths (list): list of .csv files of final RAG data
    """
    for path in paths:
        df = load_csv(path = path)

        