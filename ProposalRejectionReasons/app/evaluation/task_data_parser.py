# ---------------------------------------------------------------
# A utility class used to extract the data specific to the task
# ---------------------------------------------------------------

class EvaluationDataParser:
    """
    A class used to extract the data required to evaluate agents on a specific task
    """

    def __init__(self):
        pass

    @staticmethod
    def get_tools_alignment_data(sample):
        return {
            "job": {
                "desc": sample["job_desc"],
                "tools": sample["job_data"]["tools"]
            },

            "proposals": [
                {
                    "proposal": p["proposal"],
                    "tools"   : p["tools"]
                }

                for p in sample["proposals"]
            ]
        }
    

    @staticmethod
    def get_requirement_coverage_data(sample):
        return {
            "job": {
                "desc": sample["job_desc"],
                "requirements": sample["job_data"]["requirements"]
            },

            "proposals": [
                {
                    "proposal": p["proposal"],
                    ""   : p[""]
                }

                for p in sample["proposals"]
            ]
        }
    

    @staticmethod
    def get_job_understanding_data(sample):
        pass
    
    @staticmethod
    def get_evidence_of_experience_data(sample):
        pass

    @staticmethod
    def get_language_clarity_data(sample):
        pass

    @staticmethod
    def get_super_agent_data(sample):
        pass
