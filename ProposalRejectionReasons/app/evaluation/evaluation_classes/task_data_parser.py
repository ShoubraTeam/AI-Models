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
    def get_job_tools_extractor_data(sample):
        return {
                "job_desc": sample["job_desc"],
                "job_tools": sample["job_data"]["tools"]
            }
    
    @staticmethod
    def normalize_proposal_tool(tool: dict):
        tool_name = tool.get("tool_name", tool.get("tool"))
        if not tool_name:
            raise KeyError("Proposal tool must contain 'tool_name' or 'tool'")

        return {
            "tool_name"      : tool_name,
            "with_confidence": tool.get("with_confidence"),
        }

    @staticmethod
    def get_proposal_tools_analyzer_data(sample):
        return {
            "job_tools": sample["job_data"]["tools"],
            "proposals": [
                {
                    "proposal"      : p["proposal"],
                    "proposal_tools": [
                        EvaluationDataParser.normalize_proposal_tool(tool)
                        for tool in p["tools"]
                    ]
                } for p in sample["proposals"]
            ]
        }
        
    

    @staticmethod
    def get_job_requirements_extractor_data(sample):
        return {
            "desc": sample.get("job_desc", ""),
            "requirements": sample.get("job_data", {}).get("requirements", [])
        }
    

    @staticmethod
    def get_job_requirements_matcher_data(sample):
        return {
            "job_data": {
                "requirements": sample.get("job_data", {}).get("requirements", [])
            },
            "proposals": [
                {
                    "proposal": p.get("proposal", ""),
                    "true_covered_ids": p.get("true_covered_ids", []),
                    "true_missing_ids": p.get("true_missing_ids", [])
                } for p in sample.get("proposals", [])
            ]
        }
    
    @staticmethod
    def get_language_clarity_evaluator_data(sample):
        pass

    @staticmethod
    def get_job_key_points_extractor_data(sample):
        pass

    @staticmethod
    def get_job_understanding_evaluator_data(sample):
        pass

    @staticmethod
    def get_experience_evidence_finder_data(sample):
        return {
            "job_desc": sample.get("job_desc", ""),
            "proposals": [
                {
                    "proposal": p.get("proposal", ""),
                    "has_evidence": p.get("has_evidence", False),
                    "true_projects": p.get("true_projects", [])
                }
                for p in sample.get("proposals", [])
            ]
        }
    @staticmethod
    def get_super_agent_data(sample):
        pass
