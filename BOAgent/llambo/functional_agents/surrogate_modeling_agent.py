# llambo/functional_agents/surrogate_modeling_agent.py
import re
from llambo.llm_utils import gpt4o_generate
import logging as py_logging

class SurrogateModelingAgent:
    def __init__(self):
        self.observed_data = []
        self.logger = py_logging.getLogger(self.__class__.__name__)

    def add_observation(self, input_data, output_data):
        self.observed_data.append((input_data, output_data))
        #self.logger.info(f"Added observation: {input_data} -> {output_data}")

    def perform_task(self, prompt):
        self.logger.info(f"Executing SurrogateModelingAgent task with prompt: {prompt}")
        response = gpt4o_generate(prompt)
        #self.logger.debug(f"GPT-4 response: {response}")

        # Extract float value from the response
        match = re.search(r"[-+]?\d*\.\d+|\d+", response)
        if match:
            try:
                predicted_output = float(match.group())
                if 0.0 <= predicted_output <= 1.0:
                    self.logger.info(f"Predicted Output: {predicted_output}")
                    return predicted_output
                else:
                    self.logger.warning(f"Predicted output {predicted_output} is out of expected range [0, 1].")
                    return None
            except ValueError as e:
                self.logger.error(f"Error converting predicted output to float: {e}")
                return None
        else:
            self.logger.error("No valid float found in GPT-4 response.")
            return None
