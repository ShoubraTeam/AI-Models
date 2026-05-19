# --------------------------------------------
# Import Utility Functions
# --------------------------------------------

from typing import Any

def print_title(title: str, n_sep: int =  100, sep: str = "="):
    """Printing a title in a well-formatted manner"""
    title = f" {title} ".center(n_sep, sep)
    print(title)



def print_structured_response(structured_response):
    """Printing the Agent Structured Response"""
    print_title("Structured Response", 50)
    structured_response = structured_response.model_dump()

    for attb, value in structured_response.items():
        print()
        print(f"{attb.capitalize()}", end = "")

        if isinstance(value, list):
            print(":")
            for item in value:
                if isinstance(item, dict): 
                    for key, val in item.items():
                        print(f"\t{key} => {val}")
                else:
                    print(f"\t{item}")
                
                print()

        else:
            print(f"=> {value}")


def print_agent_response(response: Any):
    """Printing the Agent Response"""

    print_title("Messages", 50)
    for idx, message in enumerate(response["messages"], start = 1):
        print(f"Message #{idx}")
        print(f"Message Type => {message.type.capitalize()}")
        print(f"Message Content:\n{message.content}")
        print()
    
    print_structured_response(response["print_structured_response"])
    

