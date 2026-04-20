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

def print_title(title: str, n_sep = 150):
    print()
    title = " " + title + " "
    title = title.center(n_sep, "=")
    print(title)
    print()


def print_structured_response(response):
    print_title("Structured Response", 70)
    structured_response = response["structured_response"].model_dump()

    for attb, value in structured_response.items():
        print()
        print(50 * '-')
        print(attb.capitalize())

        if isinstance(value, list):
            for item in value:
                print(f">> {item}")
                print()
        else:
            print(value)


def print_dict(dict: dict, title: str):
    print_title(title.capitalize(), 70)

    for attb, value in dict.items():
        print()
        print(50 * '-')
        print(attb.capitalize())

        if isinstance(value, list):
            for item in value:
                print(f">> {item}")
                print()
        else:
            print(value)

def print_response(response):
    print_title("Messages", 70)
    for message in response["messages"]:
        print()
        print(50 * '-')
        print(message.type.capitalize())
        print(message.content)

    
    print_structured_response(response)


def list_to_str(lst: list, title: str):
    full_str = ""
    for idx, elem in enumerate(lst, start = 1):
        full_str += f"\n\n{title.capitalize()}_{idx}\n"
        full_str += elem

    return full_str