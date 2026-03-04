# imports
from dotenv import load_dotenv
from src.rephraser import LLMRephraser
from src.data import load_csv, get_row
import pandas as pd

MODEL_1 = "llama-3.1-8b-instant"
MODEL_2 = "llama-3.3-70b-versatile"

DATA_1_PATH = 'llm/data/jobs1_cleaned.csv'
DATA_2_PATH = 'llm/data/jobs2_cleaned.csv'

if __name__ == "__main__":
    load_dotenv()

    df = load_csv(DATA_1_PATH)
    rephraser = LLMRephraser(model_name = MODEL_2)

    user_data = get_row(df, index = 0)
    formatted_data, _ = rephraser.format_job(
        job_title = user_data['title'],
        job_description = user_data['description'],
        experience_required = user_data['experience'],
        skills = user_data['skills']
    )

    print(formatted_data)
    print()

    rephrased = rephraser.rephrase(input = formatted_data, temperature = 0.3)
    print(rephrased)
