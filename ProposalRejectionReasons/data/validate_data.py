# ------------------------------------------------------
# Printing the Json data in order to validate it
# ------------------------------------------------------


import json
def load_json(path: str):
    with open(path, mode = 'r', encoding = "utf-8") as f:
        return json.load(f)

def print_title(title: str):
    title = f" {title} "
    title = title.center(100, "=")
    print(title)

DATA_PATH = "data/eval_data.json"
data = load_json(DATA_PATH)




for idx, sample in enumerate(data, start = 1):


    # -- job requirements
    print(f">> Job Requirements:")
    for req in sample['job_requirements']:
        print(f"\t> {req}")
    print(100 * '-')


    # -- tools
    print(f">> Job Tools:")
    for req in sample['job_tools']:
        print(f"\t> {req}")
    print(100 * '-')


    # -------------------------
    print()

def choose_field_to_print(index: int, fields: list[str]):
    """
    Printing a job & a field to analyze them

    Args:
        index: the data row index
        field: the filed (col)
    """
    data = load_json(DATA_PATH)
    sample = data[index]

    # Job Desc
    print(f">> Job Desc:\n{sample['job_desc']}")

    # field
    if field == "job_requirements":
        print(f">> Job Requirements:")
        for req in sample['job_requirements']:
            print(f"\t> {req}")
       