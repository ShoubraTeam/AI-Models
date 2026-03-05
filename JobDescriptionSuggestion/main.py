# ---------------------------------------------------------------------------
# Contains the code required
# - define Global Variables [embedding_model, LLMs, ...]
# - Execute the whole system into the main function
# 
# Ahmed Ragab
# ---------------------------------------------------------------------------

# Imports
from pathlib import Path
import pandas as pd

from src.utils.data_preparation import load_csv

def print_separator(n_lines = 3):
    print(3 * '\n')


# CFG
MAIN_DATA_PATH = Path("./data/")
DATA_PATHS = list(MAIN_DATA_PATH.glob("*.csv"))

pd.set_option('display.max_columns', None)
if __name__ == "__main__":
    for path in DATA_PATHS:
        df = load_csv(path)
        print(df.head(n = 5))
        print_separator()