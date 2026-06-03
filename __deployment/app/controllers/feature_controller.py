

import os
import json
from typing import Any
from helpers.config import get_settings

from models.error_enums import LoggingErrors

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
        with open(self.json_path, mode = "r", encoding = "utf-8") as f:
            return json.load(f)
        
    def save_json(self, results: list[dict[str, Any]]):
        with open(self.json_path, encoding = 'utf-8') as f:
            json.dump(results, f, indent = 4, ensure_ascii = False)
        
    def log_result(self, result) -> bool:
        """
        Logging the given result.

        Returns:
            success (bool): indicating if it was a successful log operation or not.
        """
        # load file
        try:
            results = self.load_json()
        except Exception as e:
            print("error")
            raise 

            
        # parse data

        # save
        try:
            self.save_json(results)
        except:
            print("error")
            raise

        # ------------------------------------ Json Logging Utils -------------------------------------------





        
    
    