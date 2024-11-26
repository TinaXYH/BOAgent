from system_agent import SystemAgent
from llambo.functional_agents.warm_starting_agent import WarmStartingAgent
from llambo.functional_agents.surrogate_modeling_agent import SurrogateModelingAgent
from llambo.functional_agents.candidate_sampling_agent import CandidateSamplingAgent
from llambo.functional_agents.evaluation_agent import EvaluationAgent
import logging

class BOAgent:
    def __init__(self, task_context, n_initial_samples, n_trials):
        
        self.system_agent = SystemAgent(task_context)
        self.system_agent.extract_task_details()

        self.n_initial_samples = n_initial_samples
        self.n_trials = n_trials
        self.current_trial = 0
        

         # Pass hyperparameters to agents that need them
        self.warm_starting_agent = WarmStartingAgent(self.system_agent.hyperparameters)
        self.surrogate_modeling_agent = SurrogateModelingAgent()
        self.candidate_sampling_agent = CandidateSamplingAgent(self.system_agent.hyperparameters)
        self.evaluation_agent = EvaluationAgent(
            model_name=self.system_agent.model_name,
            dataset_name=self.system_agent.dataset_name
        )

        # Setup logging
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        try:
            # Generate plan
            plan = self.system_agent.generate_plan()
            self.logger.info(f"Generated Plan: {plan}")

            # Create a dictionary of functional agents
            agents = {
                "warm_starting_agent": self.warm_starting_agent,
                "surrogate_modeling_agent": self.surrogate_modeling_agent,
                "candidate_sampling_agent": self.candidate_sampling_agent,
                "evaluation_agent": self.evaluation_agent,
            }

            # Warm-starting phase
            initial_samples = self.warm_starting_agent.perform_task(self.system_agent.plan[0])
            for sample in initial_samples:
                if self.current_trial >= self.n_trials:
                    break
                eval_result = self.evaluation_agent.perform_task(sample, self.system_agent.metric)
                self.system_agent.store_observation(sample, eval_result)
                self.current_trial += 1

            # Optimization phase
            while self.current_trial < self.n_trials:
                # Surrogate Modeling
                surrogate_prompt = self.system_agent.generate_prompt("surrogate modeling")
                surrogate_pred = self.surrogate_modeling_agent.perform_task(surrogate_prompt)
                self.system_agent.store_observation("surrogate_modeling", surrogate_pred)

                # Candidate Sampling
                candidate_prompt = self.system_agent.generate_prompt("candidate sampling")
                candidate = self.candidate_sampling_agent.perform_task(candidate_prompt)
                if candidate is None:
                    self.logger.error("CandidateSamplingAgent failed to generate a valid candidate.")
                    break  # or continue to next trial
                self.system_agent.store_observation(candidate, None)

                # Evaluation
                eval_result = self.evaluation_agent.perform_task(candidate, self.system_agent.metric)
                self.system_agent.store_observation(candidate, eval_result)

                self.current_trial += 1
                self.logger.info(f"Completed trial {self.current_trial}/{self.n_trials}")

        except Exception as e:
            self.logger.error(f"An error occurred during BOAgent run: {e}")
