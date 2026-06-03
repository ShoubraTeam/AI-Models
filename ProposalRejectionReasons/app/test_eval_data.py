

import json
import time

def load_json(file_path: str):
    with open(file_path, mode = "r", encoding = "utf-8") as f:
        return json.load(f)
    


EVAL_DATA_PATH = "/mnt/d/Education/College/______GraduationProject/AI-Models/ProposalRejectionReasons/app/assets/eval_data/eval_data.json"


if __name__ == "__main__":
    data = load_json(EVAL_DATA_PATH)

    print(f">> N.Samples: {len(data)}")

    for idx, sample in enumerate(data, start = 1):
        print(f" Sample #{idx} ".center(50, '='))

        try:
            job_desc = sample["job_desc"]
            print(f">> Job Desc: {job_desc[:100]}")
            print()
        except:
            raise ValueError("Job Desc not Found")
        
        time.sleep(1)
        
        try:
            job_data = sample["job_data"]
            for key, value in job_data.items():
                if isinstance(value, str):
                    print(f"\t- {key} => {value[:100]}")
                else:
                    print(f"\t- {key} => {value}")

                print()
        except:
            raise ValueError("Job Data not Found")
        
        time.sleep(1)

        try:
            proposals = sample["proposals"]
            print(f">> N.Proposals: {len(proposals)}")

            for proposal in proposals:
                for key, value in proposal.items():
                    if isinstance(value, str):
                        print(f"\t- {key} => {value[:100]}")
                    else:
                        print(f"\t- {key} => {value}")
                print()
        
        except:
            raise ValueError("Proposals not Found")

        time.sleep(2)



    