# main.py
from boagent import BOAgent
import json
import logging

def setup_logging():
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,  # Set the global logging level to INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Suppress DEBUG logs from external libraries
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    

if __name__ == "__main__":
    setup_logging()
    
    # Example task description JSON string
    task_description_json = json.dumps({
        "task_description": {
            "model_name": "distilbert-base-uncased",  # Updated Model
            "dataset": "imdb",
            "task_type": "classification",
            "metric": "f1",
            "hyperparameters": {
                "learning_rate": {
                    "type": "continuous",
                    "range": [1e-5, 5e-5]  # Adjusted range for better convergence
                },
                "batch_size": {
                    "type": "discrete",
                    "range": [16, 32]  # Reduced options to match computational constraints
                },
                "num_epochs": {
                    "type": "discrete",
                    "range": [2, 3, 4]  # Slightly adjusted for efficient training
                }
            }
        }
    })

    # Initialize BOAgent with the task description, number of initial samples, and total trials
    bo_agent = BOAgent(task_context=task_description_json, n_initial_samples=2, n_trials=5)

    # Run the Bayesian Optimization process
    bo_agent.run()
