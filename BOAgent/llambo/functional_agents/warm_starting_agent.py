# llambo/functional_agents/warm_starting_agent.py
import json
from llambo.llm_utils import gpt4o_generate
import logging as py_logging

class WarmStartingAgent:
    def __init__(self, hyperparameters):
        self.samples = []
        self.hyperparameters = hyperparameters
        self.logger = py_logging.getLogger(self.__class__.__name__)

    def perform_task(self, prompt, num_samples=3, max_retries=3):
        hyperparameters_info = "\n".join([
            f"- {hp}: type={self.hyperparameters[hp]['type']}, range={self.hyperparameters[hp]['range']}"
            for hp in self.hyperparameters
        ])

        modified_prompt = (
            prompt +
            f"\n\nPlease generate {num_samples} initial hyperparameter configurations within the specified ranges for warm-starting. "
            "Each configuration should include the following hyperparameters:\n"
            f"{hyperparameters_info}\n"
            "Provide the output strictly as a JSON array of dictionaries, and nothing else."
        )

        for attempt in range(1, max_retries + 1):
            self.logger.info(f"WarmStartingAgent attempt {attempt} with prompt.")
            response = gpt4o_generate(modified_prompt)
            #self.logger.debug(f"GPT-4 response: {response}")

            try:
                samples = json.loads(response)
                if self.validate_samples(samples, num_samples):
                    self.samples = samples
                    self.logger.info(f"Generated Initial Samples: {self.samples}")
                    return self.samples
                else:
                    self.logger.warning("Generated samples failed validation.")
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding failed: {e}")

        self.logger.error("Failed to generate valid initial samples after multiple attempts.")
        self.samples = None
        return None

    def validate_samples(self, samples, expected_num):
        if not isinstance(samples, list):
            self.logger.error("Samples are not in a list format.")
            return False
        if len(samples) != expected_num:
            self.logger.error(f"Expected {expected_num} samples, got {len(samples)}.")
            return False
        # Validate each sample
        for candidate in samples:
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
