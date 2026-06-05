

import os
from pathlib import Path
import json
from typing import Any
from helpers.config import get_settings

from models.error_enums import LoggingErrors
from models.pydantic_schemas import AgentInferenceResult

class FeatureController:
    """
    Controlling the logic of:-
        - Controlling a feature directory
        - Saving feature files / json_logs
        - orchestrate feature workflow [inference]

    Args:
        feature_id (str): feature identifier. Must be one of the following:
            - identity_recognition
            - job_recommendation_system
            - profile_analyzer
            - job_description_enhancement
            - proposal_rejection_reasons 
    """
    def __init__(self, feature_id: str) -> None:
        # setup
        self.feature_id = feature_id
        self.init_feature_paths()


    def init_feature_paths(self) -> None:
        """Init paths related to the feature_controller"""
        results_path = get_settings().RESULTS_PATH
        
        self.json_path = os.path.join(results_path, self.feature_id, "results.json")
        self.visualizations_path = os.path.join(results_path, self.feature_id, "visualizations")
        os.makedirs(self.visualizations_path, exist_ok = True)

    # ------------------------------------ Json Logging Utils -------------------------------------------
    def load_json(self) -> list[dict[str, Any]]:
        path = Path(self.json_path)

        if not path.exists() or path.stat().st_size == 0:
            return []

        with open(path, mode = "r", encoding = "utf-8") as f:
            return json.load(f)
        
        
    def save_json(self, results: list[dict[str, Any]]) -> None:
        path = Path(self.json_path)
        path.parent.mkdir(parents = True, exist_ok = True)

        with open(path, mode = "w", encoding = "utf-8") as f:
            json.dump(results, f, indent = 4, ensure_ascii = False)
        

    def log_result(self, result: AgentInferenceResult) -> dict[str, bool | str | None]:
        """
        Logging the given result.

        Returns:
            dict contains {
                success (bool)     : indicating if it was a successful log operation or not.
                error (str | None) : error returned if success = False
            }
        """
        # load file
        try:
            json_file = self.load_json()
        except Exception as e:
            return {
                "success": False,
                "error"  : f"{LoggingErrors.JSON_FILE_LOADING_ERROR.value} - {e}"
            }

        # parse data
        try:
            result_dict = result.model_dump()
            json_file.append(result_dict)
        except Exception as e:
            return {
                "success": False,
                "error"  : f"{LoggingErrors.JSON_FILE_PROCESSING_ERROR.value} - {e}"
            }

        # save
        try:
            self.save_json(json_file)
        except Exception as e:
            return {
                "success": False,
                "error"  : f"{LoggingErrors.JSON_FILE_SAVING_ERROR.value} - {e}"
            }
        
        return {
            "success": True,
            "error"  : None
        }
        # ------------------------------------ Json Logging Utils -------------------------------------------
        




        
    
    