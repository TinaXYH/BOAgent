# system_agent.py
import json
import ast
from llambo.llm_utils import gpt4o_generate
import logging

class SystemAgent:
    def __init__(self, task_context):
        self.task_context = json.loads(task_context)  # Parse task context from JSON string
        self.plan = []
        self.hyperparameters = {}
        self.metric = ""
        self.task_type = ""
        self.model_name = ""
        self.dataset_name = ""
        self.observed_data = []
        self.range_cache = {}  # Initialize range_cache

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_task_details(self):
        # Extract details directly from the parsed task context
        task_details = self.task_context.get("task_description", {})
        self.hyperparameters = task_details.get('hyperparameters', {})
        self.metric = task_details.get('metric', 'f1')
        self.task_type = task_details.get('task_type', 'classification')
        self.model_name = task_details.get('model_name', 'bert-base-uncased')
        self.dataset_name = task_details.get('dataset', 'imdb')

        # Determine ranges for each hyperparameter using GPT-4 only for continuous types
        for hp in self.hyperparameters:
            if hp not in self.range_cache:
                hp_type = self.hyperparameters[hp]['type']
                if hp_type == 'continuous':
                    try:
                        self.hyperparameters[hp]['range'] = self.determine_range(hp)
                        self.range_cache[hp] = self.hyperparameters[hp]['range']
                        self.logger.info(f"Determined range for '{hp}': {self.range_cache[hp]}")
                    except (ValueError, SyntaxError) as e:
                        self.logger.error(f"Error determining range for hyperparameter '{hp}': {e}")
                        # Fallback to predefined range or default
                        self.hyperparameters[hp]['range'] = self.hyperparameters[hp].get('range', [0, 1])
                elif hp_type == 'discrete':
                    # Keep the range as specified in the task description
                    self.logger.info(f"Using predefined range for discrete hyperparameter '{hp}': {self.hyperparameters[hp]['range']}")
                else:
                    self.logger.error(f"Unknown hyperparameter type '{hp_type}' for '{hp}'.")
                    # Optionally, set a default range or handle accordingly
                    self.hyperparameters[hp]['range'] = self.hyperparameters[hp].get('range', [0, 1])

    def determine_range(self, hyperparameter_name):
        """
        Uses GPT-4 to determine the suitable range for a given hyperparameter based on the task type.

        Args:
            hyperparameter_name (str): The name of the hyperparameter.

        Returns:
            tuple: A tuple containing the minimum and maximum values for the hyperparameter.

        Raises:
            ValueError: If the returned range is not a valid tuple of two numbers.
        """
        range_prompt = (
            f"Given the task of {self.task_type} using the model '{self.model_name}', "
            f"what would be a suitable range for the hyperparameter '{hyperparameter_name}'? "
            "Please return the range as a Python tuple of two numbers, e.g., (min_value, max_value), and nothing else."
        )
        response = gpt4o_generate(range_prompt)
        try:
            range_tuple = ast.literal_eval(response)
            if isinstance(range_tuple, tuple) and len(range_tuple) == 2:
                min_val, max_val = range_tuple
                if min_val < max_val:
                    return range_tuple
                else:
                    raise ValueError("Minimum value is not less than maximum value.")
            else:
                raise ValueError("Returned range is not a valid tuple of two numbers.")
        except (ValueError, SyntaxError) as e:
            self.logger.error(f"Failed to parse range for '{hyperparameter_name}': {e}")
            # Handle error, e.g., return a default range or re-raise
            return (0, 1)  # Example default

    def generate_plan(self):
        """
        Generates a detailed plan for performing Bayesian Optimization using GPT-4.

        Returns:
            list: A list of steps outlining the Bayesian Optimization process.
        """
        plan_prompt = (
            f"Based on the following task description: {self.task_context}, "
            "generate a detailed plan for performing Bayesian Optimization, "
            "including steps like warm-starting, surrogate modeling, candidate sampling, and evaluation."
        )
        response = gpt4o_generate(plan_prompt)
        self.plan = [step.strip() for step in response.split('\n') if step.strip()]
        #self.logger.info(f"Generated Plan: {self.plan}")
        return self.plan

    def generate_prompt(self, step_description):
        historical_data_str = "\n".join([f"Input: {data['input']}, Output: {data['output']}" for data in self.observed_data])

        if step_description.lower() == "warm-starting":
            prompt = (
                f"We are performing warm-starting for Bayesian Optimization on the task: {self.task_type} "
                f"with model '{self.model_name}' on dataset '{self.dataset_name}'. "
                f"The hyperparameters to optimize are: {list(self.hyperparameters.keys())}.\n"
                "Please generate initial hyperparameter configurations within the specified ranges for warm-starting. "
                "Provide the output strictly as a JSON array of dictionaries."
            )
        elif step_description.lower() == "surrogate modeling":
            prompt = (
                f"Based on the task description, we are working on a {self.task_type} task using the '{self.model_name}' model on the '{self.dataset_name}' dataset. "
                f"The metric to be optimized is '{self.metric}'. The hyperparameters to be tuned are {list(self.hyperparameters.keys())}.\n\n"
                f"Given the historical data:\n{historical_data_str}\n\n"
                "Please predict the metric value for the given hyperparameters. "
                "Provide only the predicted metric value as a float between 0 and 1 (e.g., 0.85), and nothing else."
            )
        elif step_description.lower() == "candidate sampling":
            hyperparameters_info = "\n".join([
                f"- {hp}: type={self.hyperparameters[hp]['type']}, range={self.hyperparameters[hp]['range']}"
                for hp in self.hyperparameters
            ])

            prompt = (
                f"Based on the historical data:\n{historical_data_str}\n\n"
                f"The hyperparameters to be tuned are:\n{hyperparameters_info}\n\n"
                f"Please propose a new set of hyperparameters within the specified ranges that are likely to improve the {self.metric} score.\n"
                "Provide the output strictly as a JSON dictionary."
            )
        elif step_description.lower() == "evaluation":
            # Get the last candidate hyperparameters
            candidate = self.observed_data[-1]['input']
            prompt = f"Evaluating the following hyperparameters: {candidate}"
        else:
            prompt = f"Unknown step description: {step_description}"
        return prompt

    def execute_plan(self, agents):
        """
        Executes each step in the generated plan by invoking the appropriate agent.

        Args:
            agents (dict): A dictionary mapping agent names to their instances.
        """
        for step in self.plan:
            step = step.strip()
            if "warm-starting" in step.lower():
                prompt = self.generate_prompt(step)
                agents["warm_starting_agent"].perform_task(prompt)
            elif "surrogate modeling" in step.lower():
                prompt = self.generate_prompt(step)
                agents["surrogate_modeling_agent"].perform_task(prompt)
            elif "candidate sampling" in step.lower():
                prompt = self.generate_prompt(step)
                agents["candidate_sampling_agent"].perform_task(prompt)
            elif "evaluation" in step.lower():
                prompt = self.generate_prompt(step)
                evaluation_result = agents["evaluation_agent"].perform_task(prompt, self.hyperparameters, self.metric)
                # Store observed data for future use
                self.store_observation(prompt, evaluation_result)

    def store_observation(self, input_data, output_data):
        """
        Stores the observed data for future use in the optimization process.

        Args:
            input_data (str): The input prompt or configuration.
            output_data (float): The evaluation result.
        """
        self.observed_data.append({'input': input_data, 'output': output_data})
        self.logger.info(f"Stored observation: Input='{input_data}', Output='{output_data}'")
