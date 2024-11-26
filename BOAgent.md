

### Step 1: Task Description in JSON Format
Here's an example of the task description in JSON format, which includes all the required information:
- **Model Name**: BERT
- **Dataset**: IMDB
- **Task Type**: Sentiment Analysis
- **Hyperparameters**: Learning rate, batch size, and other key hyperparameters with their ranges.

```json
{
    "task_description": {
        "model_name": "bert-base-uncased",
        "dataset": "imdb",
        "task_type": "classification",
        "metric": "f1",
        "hyperparameters": {
            "learning_rate": {
                "type": "continuous",
                "range": [1e-6, 1e-3]
            },
            "batch_size": {
                "type": "discrete",
                "range": [16, 32, 64]
            },
            "num_epochs": {
                "type": "discrete",
                "range": [1, 2, 3, 4, 5]
            }
        }
    }
}
```

This JSON description contains all the relevant data for the task, which is dynamically extracted by the **System Agent** and used throughout the Bayesian Optimization process.

### Step 2: Utility for GPT-4 Integration

**File: `llambo/llm_utils.py`**

```python
import openai

def gpt4o_generate(prompt, model="gpt-4", max_tokens=500):
    """
    Utility function to generate a response from GPT-4.
    
    Args:
        prompt (str): The prompt to send to GPT-4.
        model (str): The GPT model to use (default is "gpt-4").
        max_tokens (int): The maximum number of tokens in the response.
    
    Returns:
        str: The response from GPT-4.
    """
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].text.strip()
```

### Step 3: System Agent - Task Analysis, Planning, and Prompt Generation

The **System Agent** now extracts the model name, dataset, and hyperparameters from the task description.

**System Agent (`system_agent.py`)**

```python
import json
from llambo.llm_utils import gpt4o_generate

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

    def extract_task_details(self):
        # Extract details directly from the parsed task context
        task_details = self.task_context["task_description"]
        self.hyperparameters = task_details.get('hyperparameters', {})
        self.metric = task_details.get('metric', 'f1')
        self.task_type = task_details.get('task_type', 'classification')
        self.model_name = task_details.get('model_name', 'bert-base-uncased')
        self.dataset_name = task_details.get('dataset', 'imdb')

        # Determine ranges for each hyperparameter using GPT-4
        for hp in self.hyperparameters:
            self.hyperparameters[hp]['range'] = self.determine_range(hp)

    def determine_range(self, hyperparameter_name):
        # Use GPT-4 to determine the range for a hyperparameter based on the task type
        range_prompt = (
            f"Given the task of {self.task_type} using the model '{self.model_name}', "
            f"what would be a suitable range for the hyperparameter '{hyperparameter_name}'?"
        )
        response = gpt4o_generate(range_prompt)
        return eval(response)  # Assuming GPT-4 returns the range in a valid Python format

    def generate_plan(self):
        # Use GPT-4 to generate a plan for Bayesian Optimization
        plan_prompt = (
            f"Based on the following task description: {self.task_context}, "
            "generate a detailed plan for performing Bayesian Optimization, "
            "including steps like warm-starting, surrogate modeling, candidate sampling, and evaluation."
        )
        response = gpt4o_generate(plan_prompt)
        self.plan = response.split('\n')  # Each step is a new line in the response
        return self.plan

    def generate_prompt(self, step_description):
        # Construct historical data string
        historical_data_str = "\n".join([f"Input: {data['input']}, Output: {data['output']}" for data in self.observed_data])

        # Use GPT-4 to generate a prompt for executing each step, including historical data
        prompt_generation_prompt = (
            f"Task Description: {self.task_context}\n"
            f"Step Description: {step_description}\n"
            f"Hyperparameters to Optimize: {self.hyperparameters}\n"
            f"Historical Data:\n{historical_data_str}\n"
            "Generate the proper prompt to guide the execution of this step, and decide which parts of the historical data are relevant."
        )
        response = gpt4o_generate(prompt_generation_prompt)
        return response

    def execute_plan(self, agents):
        # Execute each step in the plan by calling the appropriate agent
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
        self.observed_data.append({'input': input_data, 'output': output_data})
```

### Step 4: Functional Agents

#### WarmStarting Agent

**File: `warm_starting_agent.py`**

```python
class WarmStartingAgent:
    def __init__(self):
        self.samples = []

    def perform_task(self, prompt):
        # Execute the warm-starting task based on the given prompt
        print(f"Executing WarmStartingAgent task based on prompt: {prompt}")
        response = gpt4o_generate(prompt)
        self.samples = eval(response)  # Assuming GPT-4 returns a properly formatted list of hyperparameter configurations
        print(f"Generated Initial Samples: {self.samples}")
        return self.samples
```

#### SurrogateModeling Agent

**File: `surrogate_modeling_agent.py`**

```python
class SurrogateModelingAgent:
    def __init__(self):
        self.observed_data = []

    def add_observation(self, input_data, output_data):
        # Add a new observation
        self.observed_data.append((input_data, output_data))

    def perform_task(self, prompt):
        # Execute the surrogate modeling task based on the given prompt
        print(f"Executing SurrogateModelingAgent task based on prompt: {prompt}")
        response = gpt4o_generate(prompt)
        predicted_output = float(response)  # Assuming GPT-4 returns the predicted output as a float
        print(f"Predicted Output: {predicted_output}")
        return predicted_output
```

#### CandidateSampling Agent

**File: `candidate_sampling_agent.py`**

```python
class CandidateSamplingAgent:
    def __init__(self):
        self.history = []

    def add_to_history(self, candidate, evaluation_result):
        # Add the candidate and evaluation result to history
        self.history.append((candidate, evaluation_result))

    def perform_task(self, prompt):
        # Execute candidate sampling task based on the given prompt
        print(f"Executing CandidateSamplingAgent task based on prompt: {prompt}")
        response = gpt4o_generate(prompt)
        candidate = eval(response)  # Assume GPT-4 returns a properly formatted candidate configuration
        print(f"Generated Candidate: {candidate}")
        return candidate
```

#### Evaluation Agent

**File: `evaluation_agent.py`**

```python
from transformers import TrainingArguments, Trainer, BertForSequenceClassification, BertTokenizer
from datasets import load_dataset, load_metric

class EvaluationAgent:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset = load_dataset(self.dataset_name)

    def perform_task(self, prompt, hyperparameters, metric_name):
        # Use the hyperparameters from the prompt for evaluation
        print(f"Executing EvaluationAgent task based on prompt: {prompt}")
        hyperparameters = eval(prompt)  # Assuming the prompt contains hyperparameter configuration

        # Unpack hyperparameters dynamically
        training_args_dict = {key: hyperparameters[key] for key in hyperparameters}

        # Load the BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertForSequenceClassification.from_pretrained(self.model_name)

        # Tokenize the dataset
        def tokenize_function(example):
            return tokenizer(example["text"], padding="max_length", truncation=True)
        tokenized_datasets = self.dataset.map(tokenize_function, batched=True)

        # Prepare the training arguments dynamically
        training_args = TrainingArguments(
            output_dir="./results",
            **training_args_dict,
            logging_dir='./logs'
        )

        # Create the Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"].shuffle().select(range(1000)),  # Use subset for faster training
            eval_dataset=tokenized_datasets["test"].shuffle().select(range(1000)),   # Use subset for faster evaluation
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics(metric_name)
        )

        # Train and evaluate
        trainer.train()
        eval_result = trainer.evaluate()
        metric_score = eval_result[f'eval_{metric_name}']
        print(f"Evaluation {metric_name.upper()} Score: {metric_score}")
        return metric_score

    def compute_metrics(self, metric_name):
        metric = load_metric(metric_name)
        def metrics_fn(p):
            predictions, labels = p
            predictions = predictions.argmax(axis=1)
            return metric.compute(predictions=predictions, references=labels)
        return metrics_fn
```

### Step 5: BOAgent - Full Orchestration

The **BOAgent** class coordinates the entire Bayesian Optimization loop.

**File: `boagent.py`**

```python
from llambo.system_agent import SystemAgent
from llambo.functional_agents.warm_starting_agent import WarmStartingAgent
from llambo.functional_agents.surrogate_modeling_agent import SurrogateModelingAgent
from llambo.functional_agents.candidate_sampling_agent import CandidateSamplingAgent
from llambo.functional_agents.evaluation_agent import EvaluationAgent

class BOAgent:
    def __init__(self, task_context, n_initial_samples, n_trials):
        self.system_agent = SystemAgent(task_context)
        self.system_agent.extract_task_details()

        self.warm_starting_agent = WarmStartingAgent()
        self.surrogate_modeling_agent = SurrogateModelingAgent()
        self.candidate_sampling_agent = CandidateSamplingAgent()
        self.evaluation_agent = EvaluationAgent(
            model_name=self.system_agent.model_name,
            dataset_name=self.system_agent.dataset_name
        )

    def run(self):
        # Generate plan
        plan = self.system_agent.generate_plan()
        print(f"Generated Plan: {plan}")

        # Create a dictionary of functional agents
        agents = {
            "warm_starting_agent": self.warm_starting_agent,
            "surrogate_modeling_agent": self.surrogate_modeling_agent,
            "candidate_sampling_agent": self.candidate_sampling_agent,
            "evaluation_agent": self.evaluation_agent,
        }

        # Execute the plan
        self.system_agent.execute_plan(agents)
```

### Example Usage

**Main Script (`main.py`)**

```python
from boagent import BOAgent
import json

if __name__ == "__main__":
    # Example task description JSON string
    task_description_json = json.dumps({
        "task_description": {
            "model_name": "bert-base-uncased",
            "dataset": "imdb",
            "task_type": "classification",
            "metric": "f1",
            "hyperparameters": {
                "learning_rate": {
                    "type": "continuous",
                    "range": [1e-6, 1e-3]
                },
                "batch_size": {
                    "type": "discrete",
                    "range": [16, 32, 64]
                },
                "num_epochs": {
                    "type": "discrete",
                    "range": [1, 2, 3, 4, 5]
                }
            }
        }
    })

    # Initialize BOAgent with the task description, number of initial samples, and total trials
    bo_agent = BOAgent(task_context=task_description_json, n_initial_samples=3, n_trials=10)

    # Run the Bayesian Optimization process
    bo_agent.run()
```

### Summary of Changes and Explanations

1. **Task Description as JSON**:
   - The task description is provided as a JSON format, which includes the model name, dataset, task type, hyperparameters, and their ranges.

2. **System Agent**:
   - The **System Agent** extracts the model name, dataset, hyperparameters, and other details from the JSON input.
   - Hyperparameters' ranges are dynamically determined using GPT-4 based on the task type.

3. **Evaluation Agent**:
   - The **Evaluation Agent** uses the extracted model name and dataset for loading the required resources, rather than using hardcoded values.

4. **Full Dynamic Prompt Generation**:
   - Prompts generated for each step include the task context, hyperparameters, and historical data, allowing GPT-4 to decide the most relevant information for each step.