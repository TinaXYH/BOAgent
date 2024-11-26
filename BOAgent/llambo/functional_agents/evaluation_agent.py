# llambo/functional_agents/evaluation_agent.py
import logging as py_logging
from transformers import (
    TrainingArguments,
    Trainer,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    logging
)
from datasets import load_dataset
import evaluate
import torch

logging.set_verbosity_error()

class EvaluationAgent:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.logger = py_logging.getLogger(self.__class__.__name__)
        self.dataset = self.load_dataset()

    def load_dataset(self):
        try:
            # Load only 5% of the dataset to reduce memory usage
            dataset = load_dataset(
                self.dataset_name,
                split={'train': 'train[:5%]', 'test': 'test[:5%]'}
            )
            return dataset
        except Exception as e:
           # self.logger.error(f"Failed to load dataset '{self.dataset_name}': {e}")
            raise

    def perform_task(self, hyperparameters, metric_name, subset_size=200):
        self.logger.info(f"Starting EvaluationAgent task with hyperparameters: {hyperparameters}")
        if not isinstance(hyperparameters, dict):
            self.logger.error("Hyperparameters should be a dictionary.")
            return None

        # Prepare training arguments
        training_args_dict = {
            'learning_rate': hyperparameters.get('learning_rate', 5e-5),
            'per_device_train_batch_size': hyperparameters.get('batch_size', 4),
            'per_device_eval_batch_size': hyperparameters.get('batch_size', 4),
            'num_train_epochs': hyperparameters.get('num_epochs', 1)
        }

        try:
            # Load the DistilBERT model and tokenizer
            tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
            model = DistilBertForSequenceClassification.from_pretrained(self.model_name)

            # Limit the maximum sequence length
            max_seq_length = 128  # Reduced from 512 to 128

            # Tokenize the dataset
            def tokenize_function(example):
                return tokenizer(
                    example["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length
                )

            tokenized_datasets = self.dataset.map(
                tokenize_function,
                batched=True,
                num_proc=1,
                remove_columns=["text"],  # Remove unnecessary columns
                load_from_cache_file=True,
                keep_in_memory=False
            )

            # Reduce dataset size further if needed
            train_dataset = tokenized_datasets["train"].shuffle(seed=42)
            eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

            # Prepare the training arguments dynamically
            training_args = TrainingArguments(
                output_dir="./results",
                **training_args_dict,
                logging_dir='./logs',
                evaluation_strategy="no",  # Disable evaluation during training
                save_strategy="no",        # Disable checkpoint saving
                logging_steps=10,
                disable_tqdm=False,
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=0,  # Set to 0 to reduce memory
            )

            # Create the Trainer instance
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics(metric_name)
            )

            # Train and evaluate
            trainer.train()
            eval_result = trainer.evaluate()
            metric_score = eval_result.get(f'eval_{metric_name}', None)
            if metric_score is not None:
                self.logger.info(f"Evaluation {metric_name.upper()} Score: {metric_score}")
                return metric_score
            else:
                self.logger.warning(f"Metric '{metric_name}' not found in evaluation results.")
                return None
        except Exception as e:
            self.logger.error(f"An error occurred during training or evaluation: {e}")
            return None

    def compute_metrics(self, metric_name):
        try:
            if metric_name.lower() == 'f1':
                metric = evaluate.load('f1')
            else:
                metric = evaluate.load(metric_name)
        except Exception as e:
            self.logger.error(f"Failed to load metric '{metric_name}': {e}")
            raise

        def metrics_fn(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            return metric.compute(predictions=predictions, references=labels, average='macro')
        return metrics_fn
