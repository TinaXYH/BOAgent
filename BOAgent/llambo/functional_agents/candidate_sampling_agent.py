# llambo/functional_agents/candidate_sampling_agent.py
import json
from llambo.llm_utils import gpt4o_generate
import logging as py_logging

class CandidateSamplingAgent:
    def __init__(self, hyperparameters):
        self.history = []
        self.hyperparameters = hyperparameters
        self.logger = py_logging.getLogger(self.__class__.__name__)

    def add_to_history(self, candidate, evaluation_result):
        self.history.append((candidate, evaluation_result))
        self.logger.info(f"Added to history: {candidate} -> {evaluation_result}")

    def perform_task(self, prompt, max_retries=3):
        self.logger.info(f"Executing CandidateSamplingAgent task with prompt: {prompt}")
        for attempt in range(1, max_retries + 1):
            response = gpt4o_generate(prompt)
            try:
                candidate = json.loads(response)
                if self.validate_candidate(candidate):
                    self.logger.info(f"Generated Candidate: {candidate}")
                    return candidate
                else:
                    self.logger.warning(f"Attempt {attempt}: Generated candidate {candidate} is invalid.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Attempt {attempt}: Failed to parse candidate: {e}")
        self.logger.error("Failed to generate a valid candidate after multiple attempts.")
        return None

    def validate_candidate(self, candidate):
        expected_keys = set(self.hyperparameters.keys())
        candidate_keys = set(candidate.keys())
        if not expected_keys.issubset(candidate_keys):
            self.logger.error(f"Candidate keys {candidate_keys} do not match expected keys {expected_keys}.")
            return False

        for hp_name, hp_info in self.hyperparameters.items():
            hp_value = candidate.get(hp_name)
            hp_type = hp_info['type']
            hp_range = hp_info['range']

            if hp_type == 'continuous':
                min_val, max_val = hp_range
                if not (min_val <= hp_value <= max_val):
                    self.logger.error(f"Hyperparameter '{hp_name}' value {hp_value} out of range [{min_val}, {max_val}].")
                    return False
            elif hp_type == 'discrete':
                if hp_value not in hp_range:
                    self.logger.error(f"Hyperparameter '{hp_name}' value {hp_value} is invalid. Allowed values: {hp_range}.")
                    return False
            else:
                self.logger.error(f"Unknown hyperparameter type '{hp_type}' for '{hp_name}'.")
                return False

        return True
