# ---------------------------------------------------------------------------
# Contains the code required to load the cleaned saved .csv files in order
# to build a vector database from them
# ---------------------------------------------------------------------------

import pandas as pd
import ast

def load_csv(path: str):
    """
    Args:
        path (str): the path to the .csv file

    Returns:
        df (Pandas.DataFrame)
    """
    df = pd.read_csv(path)
    return df


def form_document(row):
    """
    Turning a row in a Pandas DataFrame to a Document (concat all the columns)

    Returns:
        document: the whole job document containing (Title - Description - skills (if found) - category (if found))
    """    
    title = row.title
    description = row.description
    skills = row.skills
    category = row.category

    # format skills
    if pd.isna(skills):
        skills_sentence = ""
    else:
        skills_str = ", ".join(ast.literal_eval(skills))
        skills_sentence = f"""
        
Recommended Skills:
{skills_str}
"""
        
    # format category
    if pd.isna(category):
        cat_sentence = ""
    else:
        cat_sentence = f"""
        
The job belong to these categories:
{category}
"""    
        
    document = f"""Job Title: {title}

Job Description:
{description}{cat_sentence}{skills_sentence}
"""
    
    return document.strip()
    


def jobs_to_documents(path: str):
    """
    Turning all the data into documents
    """
    df = load_csv(path = path)

    documents = []
    for row in df.itertuples(index = False):
        doc = form_document(row)

        documents.append({
            'job_document' : doc,
            'year' : int(row.year)
        })

    
    documents_df = pd.DataFrame(documents)
    return documents_df


def get_documents(document_path: str, jobs_path: str):
    """
    Builds / Retrieves the documents_df
    """
    try:
        doc_df = pd.read_parquet(document_path, index_col = 0)
    except:
        doc_df = jobs_to_documents(jobs_path)
        doc_df.to_parquet(document_path)
    finally:
        return doc_df