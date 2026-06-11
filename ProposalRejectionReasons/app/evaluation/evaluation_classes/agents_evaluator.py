# ----------------------------------------------------------------------------------
# A utility class used as an abstract class to evluate any underlying given model
# ----------------------------------------------------------------------------------
import json
from pprint import pprint
from helpers import config as CFG
import time
from pathlib import Path

from ..handle_errors import parse_groq_error, get_short_error_info
from groq import BadRequestError

from agents import JobToolsExtractor, ProposalToolsAnalyzer
from agents import JobKeyPointsExtractor, JobUnderstandingEvaluator
from agents import JobRequirementsExtractor, JobRequirementsMatcher
from agents import ExperienceEvidenceAgent
from agents import LanguageClarityEvaluator




Agent_Type = JobToolsExtractor | \
    ProposalToolsAnalyzer      | \
    JobKeyPointsExtractor      | \
    JobUnderstandingEvaluator  | \
    JobRequirementsExtractor   | \
    JobRequirementsMatcher     | \
    ExperienceEvidenceAgent    | \
    LanguageClarityEvaluator



class AgentsEvaluator:
    """
    Attbs:
        task_name         : str represents the agent being evaluated | evaluation task
        agent             : the agent to evaluate
        data              : evaluation samples
        run_configurations: the run configuration to log the current run 
    """
    def __init__(
            self, 
            task_name         : str, 
            agent             : Agent_Type, 
            data              : list[dict], 
            run_configurations: dict,
            reset_results     : bool = False
        ):

        self.task_name = task_name
        self.agent     = agent
        self.data      = data

        self.json_file_path = Path(
            CFG.EVAL_RESULTS_PATH,
            "json_results",
            f"{self.task_name}.json"
        )

        self.log_file_path = Path(
            CFG.EVAL_RESULTS_PATH,
            "logs",
            f"{self.task_name}.txt"
        )

        if reset_results:
            self.reset_files()

        self.log_file = open(
            file = self.log_file_path,
            mode = "a",
            encoding = "utf-8"
        )

        self.run_configurations = run_configurations
        self.parse_run_configurations(run_configurations)

    
    def parse_run_configurations(self, run_configuration):
        self.run_id       = run_configuration["run_id"]
        self.model_name   = run_configuration["model_name"]
        self.model_kwargs = run_configuration["model_kwargs"]

    def reset_files(self) -> None:
        with open(self.json_file_path, mode = "w", encoding = "utf-8") as f:
            json.dump([], f, indent = 4, ensure_ascii = False)

        with open(self.log_file_path, mode = "w", encoding = "utf-8") as f:
            f.write("")
    # -------------------------------------------- Evaluation -------------------------------------------------
    def calc_avg_for_multiple_metrics(self, metrics: dict[str, list[float]]) -> dict[str, float]:
        """
        Calculating the average for multiple metrics

        Args:
            metrics: the dict of given metrics

        Returns:
            averages: a dict contains mapping each metric to its corresponding average
        """
        if not isinstance(metrics, dict):
            print(type(metrics))
            raise TypeError("The metrics should be a dict object")
        
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in metrics.items()
        }


    def evaluate(self) -> None:
        """Evaluating Agents - Logging & Saving Results"""
        self.log_event(message = f"Run #{self.run_id}", title = True)
        self.log_event(message = "Run Configurations:")
        self.log_event(message = self.run_configurations, dic = True, identation = 1)

        metrics = {
            metric_name: []
            for metric_name in self.agent.get_metric_names()
        }

        # evaluate
        n_errors = 0
        for sample_idx, sample in enumerate(self.data, start = 1):
            self.log_event(message = f"\n=> Sample #{sample_idx}:")
            try:
                sample_metrics = self.agent.evaluate_sample(sample = sample)

                for metric_name, metric_value in sample_metrics.items():
                    metrics[metric_name].append(metric_value)

                self.log_event(message = f"- Passed", identation = 1)
                self.log_event(message = f"- Results:", identation = 1)
                self.log_event(message = sample_metrics, identation = 2, dic = True)

            except BadRequestError as e:
                n_errors += 1
                error_info = parse_groq_error(e)
                error_info = {
                    "sample_idx"          : sample_idx,
                    "error_type"          : "BadRequestError",
                    "error_code"          : error_info.get("error_code"),
                    "message"             : error_info.get("message"),
                    "summary_length"      : error_info.get("summary_length"),
                    "tool_reviews_count"  : error_info.get("tool_reviews_count"),
                    "has_extra_type_field": error_info.get("has_extra_type_field"),
                }
                pprint(error_info, indent = 2)
                print()

                self.log_event(f"- Groq Bad Request Error", identation = 1)
                self.log_event(f"- Error Info:", identation = 1)
                self.log_event(error_info, dic = True, identation = 2)

            except Exception as e:
                n_errors += 1
                error_info = get_short_error_info(e)

                self.log_event(f"- Unexpected Error", identation = 1)
                self.log_event(f"- Error Info:", identation = 1)
                self.log_event(error_info, dic = True, identation = 2)
            
            finally:
                time.sleep(1)
        

        # return metrics
        total_samples = len(self.data)
        error_rate = n_errors / total_samples if total_samples else 0.0
        metrics = self.calc_avg_for_multiple_metrics(metrics = metrics)

        metrics["error_rate"] = error_rate
        
        
        # output
        self.write_run_result(
            results = metrics,
        )

        self.log_event(message = "", sep = True)
        self.close()


    # -------------------------------------------- Saving Results -------------------------------------------------  
    def load_json(self) -> list[dict]:
        with open(self.json_file_path, mode = "r", encoding = "utf-8") as f:
            return json.load(f)

    def save_json(self, logs: list[dict], mode: str = "w"):
        with open(self.json_file_path, mode = mode, encoding = 'utf-8') as f:
            json.dump(logs, f, indent = 4, ensure_ascii = False)


    def write_run_result(self, results : dict[str, float],):
        """
        Logging a single run

        Args:
            results   : the model results [accuracy - precison - time - ..]
        """
        correct_counts = 0

        try:
            json_results = self.load_json()
            correct_counts += 1

        except Exception as e:
            print(f">> Error while loading results file: {e}")
            raise

        run_results = {
            "run_id"       : self.run_id,
            "model_name"   : self.model_name,
            "model_kwargs" : self.model_kwargs,
            "results"      : results,
        }

        try:
            json_results.append(run_results)
            self.save_json(json_results)
            correct_counts += 1

        except Exception as e:
            print(f">> Error while saving results file: {e}")
            raise
        

        if correct_counts != 2:
            return False
        
        return True
    
    # -------------------------------------------- Logging -------------------------------------------------
    def log_event(self, message, identation: int = 0, title = False, dic = False, sep = False):
        self.log_file.write(f"\n")
        
        identation_str = ""
        if identation > 0:
            identation_str = '\t' * identation
  

        if title:
            message = f" {message} "
            self.log_file.write(message.center(100, "="))
        
        elif dic:
            for key, val in message.items():
                self.log_file.write(f'{identation_str}>> {key} --> {val}\n')
        
        elif sep:
            self.log_file.write(100 * "=")
        
        else:
            self.log_file.write(f"{identation_str}{message}")

        self.log_file.flush()

    def close(self) -> None:
        if not self.log_file.closed:
            self.log_file.close()
