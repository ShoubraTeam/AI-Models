# ------------------------------------------------------------
# contains functions for loading the data (experimental)
# ------------------------------------------------------------

import pandas as pd

def load_csv(csv_path):
    df = pd.read_csv(csv_path, index_col = 0)
    return df

def get_row(df, index = 0):
    row = df.iloc[index]

    return {
        "title": row.get("job_title"),
        "description": row.get("description"),
        "experience": row.get("ex_level_demand"),
        "skills": row.get("categories"),
    }
    