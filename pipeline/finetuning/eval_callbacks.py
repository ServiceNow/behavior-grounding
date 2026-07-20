import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch
import logging
import omegaconf
logger = logging.getLogger(__name__)
from tqdm import tqdm
import pickle as pkl

class LLMRecEvalCallback(TrainerCallback):
    """
    Custom evaluation callback that performs generation-based evaluation
    similar to the llmrec function during training.
    """
    
    def __init__(
        self,
        eval_dataset_path: str,
        sample_number: int = 100,
        task: str = "direct",
        few_zero: str = "zero",
        max_new_tokens: int = 4096,
        debug_print: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        tokenizer=None,
        output_dir: str = "./output",
        eval_batch_size: int = 8,
        tag: str = "eval"
    ):
        """
        Initialize the generation evaluation callback.
        
        Args:
            eval_dataset_path: Path to the evaluation dataset (JSONL format)
            sample_number: Number of samples to evaluate
            task: Task type to filter from dataset
            few_zero: Few-shot or zero-shot setting to filter
            output_dir: Directory to save evaluation results
            debug_print: Whether to print debug information
        """
        self.eval_dataset_path = eval_dataset_path
        self.sample_number = sample_number
        self.task = task
        self.few_zero = few_zero
        self.debug_print = debug_print
        self.eval_results = []
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.df = self.load_eval_dataset()
        self.sampled_df = self.df.sample(n=self.sample_number, random_state=42)
        self.eval_batch_size = eval_batch_size
        self.tag = tag
    def load_eval_dataset(self) -> pd.DataFrame:
        """Load and filter the evaluation dataset"""
        logger.info(f"Loading evaluation dataset from {self.eval_dataset_path}")
        
        list_of_json = []
        with open(self.eval_dataset_path, "r") as f:
            for line in f:
                list_of_json.append(json.loads(line))
        
        df = pd.DataFrame(list_of_json)

        df = df[df["few_zero"] == self.few_zero]
        df = df[df["task"] == self.task]
        
        logger.info(f"Loaded {len(df)} examples for evaluation")
        return df
    
    def generate_responses(self, model, tokenizer, prompts: List[str]) -> List[str]:
        """Generate responses for the given prompts"""
        model.eval()
        responses = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), self.eval_batch_size), desc="Generating eval responses"):
                batch_prompts = prompts[i:i + self.eval_batch_size]
                logger.info(f"batch_prompts: {batch_prompts}")
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=32768,  # Max input length (separate from max_new_tokens for generation)
                    padding=True
                ).to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
                # Outputs may be a 2D tensor: [batch_size, seq_length]
                # For each prompt, decode the generated output past the input prompt length
                input_lengths = [len(input_ids) for input_ids in inputs['input_ids']]
                for j, output in enumerate(outputs):
                    start_pos = input_lengths[j]
                    response = tokenizer.decode(
                        output[start_pos:],
                        skip_special_tokens=True
                    )
                    print(f"prompt: {batch_prompts[j]}")
                    print("*" * 100)
                    print(f"response: {response}")
                    print("-" * 100)
                    responses.append(response.strip())
        return responses
    
    def evaluate_step(self, model, tokenizer, eval_df: pd.DataFrame, current_epoch: int=0, is_main_process: bool=True) -> Dict[str, Any]:
        """Perform evaluation on a sample of the dataset"""
        prompts = eval_df["prompt"].tolist()
        targets = eval_df["target"].tolist()

        logger.info("Generating responses...")
        responses = self.generate_responses(model, tokenizer, prompts)

        eval_results = {
            "num_samples": len(prompts),
            "avg_response_length": np.mean([len(r) for r in responses]),
            "responses": responses,
            "targets": targets,
            "prompts": prompts
        }
        
        if is_main_process:
            try:    
                pkl.dump(eval_results, open(f"{self.output_dir}/eval_results_{self.tag}_step_{current_epoch}.pkl", "wb"))
                logger.info(f"Saved eval results to {self.output_dir}/eval_results_{self.tag}_step_{current_epoch}.pkl")
            except Exception as e:
                logger.error(f"Failed to save eval results: {str(e)}")
                logger.exception("Full traceback:")

        if self.task == "direct" or self.task == "sequential":
            def process_response_sequential(response):
                try:
                    if ">" in response:
                        response = response.replace(">", "").strip()
                    if "assistantfinal" in response:
                        response = response.split("assistantfinal")[-1]
                        response = response.strip()
                    
                    if "[" in response and "]" in response:
                        #remove evrthing before the last "[" and after the last "]"
                        response = response.split("[")[1].split("]")[0]
                        response = "[" + response + "]"
                        response = eval(response)
                        return response    
                except Exception as e:
                    print(e)
                    return None
            tmp_df = pd.DataFrame({"response": responses, "target": targets})
            tmp_df["processed_response"] = tmp_df["response"].apply(process_response_sequential)
            
            for index, row in tmp_df.iterrows():
                print(f"response: {row['response']}")
                print(f"target: {row['target']}")
                print(f"processed_response: {row['processed_response']}")
                print("-" * 100)
            # Filter out None values and keep targets aligned
            valid_df = tmp_df[tmp_df["processed_response"].notna()].copy()
            
            if len(valid_df) > 0:
                eval_results["hit_rate_at_top_5"] = np.mean([
                    1 if row['target'] in row['processed_response'][:5] else 0 
                    for _, row in valid_df.iterrows()
                ])
                eval_results["hit_rate_at_top_10"] = np.mean([
                    1 if row['target'] in row['processed_response'][:10] else 0 
                    for _, row in valid_df.iterrows()
                ])
                eval_results["valid_response_rate"] = len(valid_df) / len(tmp_df)
                
                logger.info(f"hit_rate_at_top_5: {eval_results['hit_rate_at_top_5']:.4f}")
                logger.info(f"hit_rate_at_top_10: {eval_results['hit_rate_at_top_10']:.4f}")
                logger.info(f"valid_response_rate: {eval_results['valid_response_rate']:.4f}")
            else:
                logger.warning("No valid processed responses found")
                eval_results["hit_rate_at_top_5"] = 0.0
                eval_results["hit_rate_at_top_10"] = 0.0
                eval_results["valid_response_rate"] = 0.0
        elif self.task == "rating":
            import json
            def process_response_rating(response):
                #handle format of (x stars, xx%)
                if "assistantfinal" in response:
                    response = response.split("assistantfinal")[-1]
                    if "(" in response and ")" in response:
                        response = response.split("(")[1]
                        response = "(" + response
                    response = response.strip()
                    return response
                if "(" in response and ")" in response:
                    response = response.split("(")[-1]
                    response = "(" + response
                    response = response.strip()
                    return response
                return None
            
            def convert_rating_string_to_tuple(rating_string):
                try:
                    if rating_string is None:
                        return None, None
                    rating_string = rating_string.strip("()")
                    stars, percentage = rating_string.split(",")
                    #remove the % sign and unnecessary strings like "stars"
                    percentage = percentage.replace("%", "").strip()
                    stars = stars.replace("stars", "").replace("star", "").strip()
                    return (float(stars.strip()), float(percentage.strip()))
                except Exception as e:
                    return None, None
            tmp_df = pd.DataFrame({"response": responses, "target": targets})
            tmp_df["processed_response"] = tmp_df["response"].apply(process_response_rating)
            tmp_df["processed_response"] = tmp_df["processed_response"].apply(convert_rating_string_to_tuple)
            non_none_processed_response = tmp_df["processed_response"].dropna()
            non_none_processed_response["rating"] = non_none_processed_response["processed_response"].apply(lambda x: x[0])
            non_none_processed_response["confidence"] = non_none_processed_response["processed_response"].apply(lambda x: x[1])
            non_none_processed_response["rating_difference"] = non_none_processed_response["target"] - non_none_processed_response["rating"]
            non_none_processed_response["absolute_rating_difference"] = non_none_processed_response["rating_difference"].abs()
            eval_results["average_prediction_error"] = non_none_processed_response["absolute_rating_difference"].mean()
            eval_results["weighted_average_prediction_error"] = non_none_processed_response["absolute_rating_difference"].mean() * non_none_processed_response["confidence"]

        
        return eval_results
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training to establish baseline"""
        logger.info(f"[{self.tag}] Starting initial baseline evaluation before training...")
        model = kwargs.get('model')
        model.eval()
        
        try:
            # Use stored tokenizer
            if self.tokenizer is None:
                logger.error("Tokenizer not available for evaluation callback")
                return
            
            # Load evaluation dataset
            if len(self.df) == 0:
                logger.warning("No evaluation data found, skipping generation evaluation")
                return
            
            # Perform evaluation at step 0
            eval_results = self.evaluate_step(model, self.tokenizer, self.sampled_df, current_epoch=0, is_main_process=state.is_world_process_zero)
            
            # Store results
            eval_results["step"] = 0
            eval_results["epoch"] = 0
            self.eval_results.append(eval_results)
            
            # Log to wandb if available and main process
            if state.is_world_process_zero and hasattr(args, 'report_to') and 'wandb' in args.report_to:
                import wandb
                # log everything in eval_results with tag prefix
                for key, value in eval_results.items():
                    # Skip logging non-numeric values like responses, targets, prompts
                    if key not in ['responses', 'targets', 'prompts']:
                        wandb.log({f"{self.tag}/{key}": value}, step=0)
            
            logger.info(f"[{self.tag}] Baseline evaluation completed")
            logger.info(f"[{self.tag}] Evaluated {eval_results['num_samples']} samples")
            logger.info(f"[{self.tag}] Average response length: {eval_results['avg_response_length']:.2f}")
            
            # Save baseline results
            if state.is_world_process_zero:
                self.save_eval_results(0)
                
        except Exception as e:
            logger.error(f"Error during baseline evaluation: {str(e)}")
            logger.exception("Full traceback:")
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each epoch"""
        logger.info("Starting epoch end generation-based evaluation...")
        model = kwargs.get('model')
        model.eval()
        current_epoch = state.epoch
        try:
            # Use stored tokenizer
            if self.tokenizer is None:
                logger.error("Tokenizer not available for evaluation callback")
                return
            
            # Load evaluation dataset
            if len(self.df) == 0:
                logger.warning("No evaluation data found, skipping generation evaluation")
                return
            
            # Perform evaluation
            eval_results = self.evaluate_step(model, self.tokenizer, self.sampled_df, current_epoch, is_main_process=state.is_world_process_zero)
            
            # Store results
            eval_results["step"] = state.global_step
            eval_results["epoch"] = state.epoch
            self.eval_results.append(eval_results)
            
            
            # Log to wandb if available
            if state.is_world_process_zero and hasattr(args, 'report_to') and 'wandb' in args.report_to:
                import wandb
                # log everything in eval_results with tag prefix
                for key, value in eval_results.items():
                    # Skip logging non-numeric values like responses, targets, prompts
                    if key not in ['responses', 'targets', 'prompts']:
                        wandb.log({f"{self.tag}/{key}": value}, step=state.global_step)
            
            # IMPORTANT: Add metrics to trainer logs for early stopping
            # This makes the metrics available to EarlyStoppingCallback
            metrics_to_log = {}
            for key, value in eval_results.items():
                if key not in ['responses', 'targets', 'prompts', 'step', 'epoch']:
                    # Add prefix to distinguish different callbacks
                    metrics_to_log[f"{self.tag}_{key}"] = value
            
            # Update the trainer's log history
            if metrics_to_log:
                # Add to the trainer's state
                for key, value in metrics_to_log.items():
                    state.log_history[-1][key] = value if state.log_history else None
            
            logger.info(f"[{self.tag}] Generation evaluation completed at step {state.global_step}")
            logger.info(f"[{self.tag}] Evaluated {eval_results['num_samples']} samples")
            logger.info(f"[{self.tag}] Average response length: {eval_results['avg_response_length']:.2f}")
            
            if state.is_world_process_zero and state.global_step % (args.save_steps * 2) == 0:
                self.save_eval_results(state.global_step)
                
        except Exception as e:
            logger.error(f"Error during generation evaluation: {str(e)}")
            logger.exception("Full traceback:")

    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def save_eval_results(self, step: int):
        """Save evaluation results to file"""
        if not self.eval_results:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
            
        # Save latest results
        latest_results = self.eval_results[-1]
        output_file = os.path.join(self.output_dir, f"generation_eval_{self.tag}_step_{step}.json")
        
        # Prepare data for saving (remove non-serializable items)
        save_data = {
            "step": latest_results["step"],
            "epoch": latest_results["epoch"],
            "num_samples": latest_results["num_samples"],
            "avg_response_length": latest_results["avg_response_length"],
            "sample_responses": latest_results["responses"],  # Save first 5 responses
            "sample_targets": latest_results["targets"],  # Save first 5 targets
            "sample_prompts": latest_results["prompts"]  # Save first 5 prompts
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved generation evaluation results to {output_file}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        if self.eval_results and state.is_world_process_zero:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save final evaluation summary
            final_output = os.path.join(self.output_dir, f"generation_eval_{self.tag}_summary.json")
            
            summary = {
                "total_evaluations": len(self.eval_results),
                "final_step": state.global_step,
                "final_epoch": state.epoch,
                "evaluation_history": [
                    {
                        "step": result["step"],
                        "epoch": result["epoch"],
                        "num_samples": result["num_samples"],
                        "avg_response_length": result["avg_response_length"]
                    }
                    for result in self.eval_results
                ]
            }
            
            with open(final_output, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved final generation evaluation summary to {final_output}")


class IterativeRecEvalCallback(TrainerCallback):
    """
    Custom evaluation callback for iterative recommendation tasks.
    For each entry, asks the model 10 times for ONE item recommendation,
    removing recommended items from candidates after each iteration.
    """
    
    def __init__(
        self,
        eval_dataset_path: str,
        sample_number: int = 100,
        max_new_tokens: int = 512,
        debug_print: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        tokenizer=None,
        output_dir: str = "./output",
        eval_batch_size: int = 1,  # Usually 1 for this task
        tag: str = "eval_iterative",
        num_iterations: int = 1,
        enable_thinking: bool = False
    ):
        """
        Initialize the iterative recommendation evaluation callback.
        
        Args:
            eval_dataset_path: Path to the evaluation dataset (CSV format)
            sample_number: Number of samples to evaluate
            max_new_tokens: Maximum tokens to generate
            debug_print: Whether to print debug information
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            tokenizer: Tokenizer instance
            output_dir: Directory to save evaluation results
            eval_batch_size: Batch size for evaluation
            tag: Tag for logging
            num_iterations: Number of iterations per entry (default 10)
        """
        self.eval_dataset_path = eval_dataset_path
        self.sample_number = sample_number
        self.debug_print = debug_print
        self.eval_results = []
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.df = self.load_eval_dataset()
        self.sampled_df = self.df.sample(n=min(self.sample_number, len(self.df)), random_state=42)
        self.eval_batch_size = eval_batch_size
        self.tag = tag
        self.num_iterations = num_iterations
        self.enable_thinking = enable_thinking
    
    def load_eval_dataset(self) -> pd.DataFrame:
        """Load the evaluation dataset from CSV"""
        logger.info(f"Loading evaluation dataset from {self.eval_dataset_path}")
        
        df = pd.read_csv(self.eval_dataset_path)
        
        # Validate required columns
        if "target_interaction" in df.columns:
            df["target"] = df["target_interaction"]
        if "candidate_pool" in df.columns:
            df["candidates"] = df["candidate_pool"]
        required_columns = ['target', 'candidates', 'interaction_history']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Required columns not found: {missing_columns}")
        
        logger.info(f"Loaded {len(df)} examples for evaluation")
        return df
    
    def format_prompt(self, interaction_history: str, candidate_pool: str) -> str:
        """Format the prompt for recommendation"""
        prompt = f"""Requirements: you must choose one item for recommendation and sort them in order of priority, from highest to lowest. Only output the item name. Do not explain the reason or include any other words. 
The user has interacted with the following items (in no particular order):
{interaction_history}. 
From the candidates listed below, choose the top 1 item to recommend to the user and rank it in order of priority from highest to lowest. Candidates: 
{candidate_pool}"""
        return prompt
    
    def generate_single_response(self, model, tokenizer, prompt: str) -> str:
        """Generate a single response for the given prompt"""
        model.eval()
        
        with torch.no_grad():
            messages = [{"role": "user", "content": prompt}]
            
            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True
            }
            if not tokenizer.__class__.__name__.endswith("MistralCommonBackend"):
                chat_template_kwargs["enable_thinking"] = self.enable_thinking
                
            formatted_prompt = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
            
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=32768,
                padding=False
            ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Decode only the generated part
            input_length = inputs['input_ids'].shape[1]
            response = tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            return response.strip()
    
    def evaluate_iterative_recommendation(self, model, tokenizer, row) -> List[str]:
        """
        Perform iterative recommendation for a single row.
        Returns a list of recommended items in order.
        """
        interaction_history = row['interaction_history']
        candidates = eval(row['candidates']) if isinstance(row['candidates'], str) else row['candidates']
        
        # Make a copy of candidates to modify
        remaining_candidates = candidates.copy()
        recommendations = []
        all_responses = []
        all_prompts = []
        
        for iteration in range(self.num_iterations):
            if not remaining_candidates:
                logger.warning(f"No more candidates remaining at iteration {iteration}")
                break
            
            # Format candidate pool as string
            candidate_pool = ", ".join(remaining_candidates)
            
            # Create prompt
            prompt = self.format_prompt(interaction_history, candidate_pool)
            
            # Generate response
            response = self.generate_single_response(model, tokenizer, prompt)
            all_responses.append(response)
            all_prompts.append(prompt)
            if self.debug_print:
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Iteration {iteration + 1}/{self.num_iterations}")
                logger.info(f"Response: {response}")
            
            # Extract recommended item from response
            recommended_item = self.extract_item_from_response(response, remaining_candidates)
            
            if recommended_item:
                recommendations.append(recommended_item)
                remaining_candidates.remove(recommended_item)
            else:
                logger.warning(f"Could not extract valid item from response: {response}")
        
        return all_prompts, all_responses, recommendations
    
    def extract_item_from_response(self, response: str, candidates: List[str]) -> Optional[str]:
        """
        Extract the recommended item from the response.
        Try to match against candidate list.
        """
        if "</think>" in response:
            #strip the thinking part
            response = response.split("</think>")[1].strip()

            print(f"Response after stripping thinking part: {response}")


        response_lower = response.lower().strip()
        
        # Try exact match first
        for candidate in candidates:
            if candidate.lower() == response_lower:
                return candidate
        
        # Try substring match
        for candidate in candidates:
            if candidate.lower() in response_lower:
                return candidate
        
        # Try reverse - if response is in candidate
        for candidate in candidates:
            if response_lower in candidate.lower():
                return candidate
        
        return None
    
    def evaluate_step(self, model, tokenizer, eval_df: pd.DataFrame, current_epoch: int = 0, is_main_process: bool = True) -> Dict[str, Any]:
        """Perform evaluation on the sampled dataset"""
        all_recommendations = []
        all_targets = []
        
        
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating samples"):
            target = row['target']
            all_prompts, all_responses, recommendations = self.evaluate_iterative_recommendation(model, tokenizer, row)
            
            all_recommendations.append(recommendations)
            all_targets.append(target)
            
            if self.debug_print:
                logger.info(f"Sample {idx}")
                logger.info(f"Target: {target}")
                logger.info(f"Recommendations: {recommendations}")
                logger.info("-" * 100)
        
        hit_at_5 = []
        hit_at_10 = []
        
        for recommendations, target in zip(all_recommendations, all_targets):
            # Check if target is in top 5
            is_hit_5 = target in recommendations[:5] if len(recommendations) >= 5 else target in recommendations
            hit_at_5.append(1 if is_hit_5 else 0)
            
            # Check if target is in top 10
            is_hit_10 = target in recommendations[:10] if len(recommendations) >= 10 else target in recommendations
            hit_at_10.append(1 if is_hit_10 else 0)
        
        eval_results = {
            "num_samples": len(eval_df),
            "hit_rate_at_top_5": np.mean(hit_at_5),
            "hit_rate_at_top_10": np.mean(hit_at_10),
            "avg_recommendations_per_sample": np.mean([len(r) for r in all_recommendations]),
            "recommendations": all_recommendations,
            "targets": all_targets,
            "responses": all_responses,
            "prompts": all_prompts,
        }

        if is_main_process:
            try:
                pkl.dump(eval_results, open(f"{self.output_dir}/eval_results_{self.tag}_step_{current_epoch}.pkl", "wb"))
                logger.info(f"Saved eval results to {self.output_dir}/eval_results_{self.tag}_step_{current_epoch}.pkl")
            except Exception as e:
                logger.error(f"Failed to save eval results: {str(e)}")
                logger.exception("Full traceback:")
        
        logger.info(f"[{self.tag}] Hit rate at top 5: {eval_results['hit_rate_at_top_5']:.4f}")
        logger.info(f"[{self.tag}] Hit rate at top 10: {eval_results['hit_rate_at_top_10']:.4f}")
        
        return eval_results
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training to establish baseline"""
        logger.info(f"[{self.tag}] Starting initial baseline iterative recommendation evaluation...")
        model = kwargs.get('model')
        model.eval()
        
        try:
            if self.tokenizer is None:
                logger.error("Tokenizer not available for evaluation callback")
                return
            
            if len(self.df) == 0:
                logger.warning("No evaluation data found, skipping evaluation")
                return
            
            # Perform evaluation at step 0
            eval_results = self.evaluate_step(model, self.tokenizer, self.sampled_df, current_epoch=0, is_main_process=state.is_world_process_zero)
            
            # Store results
            eval_results["step"] = 0
            eval_results["epoch"] = 0
            self.eval_results.append(eval_results)
            
            # Log to wandb if available
            if state.is_world_process_zero and hasattr(args, 'report_to') and 'wandb' in args.report_to:
                import wandb
                for key, value in eval_results.items():
                    # Skip logging non-numeric values
                    if key not in ['recommendations', 'targets', 'responses', 'prompts']:
                        wandb.log({f"{self.tag}/{key}": value}, step=0)
            
            logger.info(f"[{self.tag}] Baseline evaluation completed")
            logger.info(f"[{self.tag}] Hit rate at top 5: {eval_results['hit_rate_at_top_5']:.4f}")
            logger.info(f"[{self.tag}] Hit rate at top 10: {eval_results['hit_rate_at_top_10']:.4f}")
            
        except Exception as e:
            logger.error(f"Error during baseline iterative recommendation evaluation: {str(e)}")
            logger.exception("Full traceback:")
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each epoch"""
        logger.info(f"[{self.tag}] Starting epoch end iterative recommendation evaluation...")
        model = kwargs.get('model')
        model.eval()
        current_epoch = state.epoch
        
        try:
            if self.tokenizer is None:
                logger.error("Tokenizer not available for evaluation callback")
                return
            
            if len(self.df) == 0:
                logger.warning("No evaluation data found, skipping evaluation")
                return
            
            # Perform evaluation
            eval_results = self.evaluate_step(model, self.tokenizer, self.sampled_df, current_epoch, is_main_process=state.is_world_process_zero)
            
            # Store results
            eval_results["step"] = state.global_step
            eval_results["epoch"] = state.epoch
            self.eval_results.append(eval_results)
            
            # Log to wandb if available
            if state.is_world_process_zero and hasattr(args, 'report_to') and 'wandb' in args.report_to:
                import wandb
                for key, value in eval_results.items():
                    # Skip logging non-numeric values
                    if key not in ['recommendations', 'targets', 'responses', 'prompts']:
                        wandb.log({f"{self.tag}/{key}": value}, step=state.global_step)
            
            # IMPORTANT: Add metrics to trainer logs for early stopping
            # This makes the metrics available to EarlyStoppingCallback
            metrics_to_log = {}
            for key, value in eval_results.items():
                if key not in ['recommendations', 'targets', 'responses', 'prompts', 'step', 'epoch']:
                    # Add prefix to distinguish different callbacks
                    metrics_to_log[f"{self.tag}_{key}"] = value
            
            # Update the trainer's log history
            if metrics_to_log:
                # Add to the trainer's state
                for key, value in metrics_to_log.items():
                    state.log_history[-1][key] = value if state.log_history else None
            
            logger.info(f"[{self.tag}] Evaluation completed at step {state.global_step}")
            
        except Exception as e:
            logger.error(f"Error during iterative recommendation evaluation: {str(e)}")
            logger.exception("Full traceback:")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        if self.eval_results and state.is_world_process_zero:
            os.makedirs(self.output_dir, exist_ok=True)
            
            final_output = os.path.join(self.output_dir, f"iterative_eval_{self.tag}_summary.json")
            
            summary = {
                "total_evaluations": len(self.eval_results),
                "final_step": state.global_step,
                "final_epoch": state.epoch,
                "evaluation_history": [
                    {
                        "step": result["step"],
                        "epoch": result["epoch"],
                        "num_samples": result["num_samples"],
                        "hit_rate_at_top_5": result["hit_rate_at_top_5"],
                        "hit_rate_at_top_10": result["hit_rate_at_top_10"]
                    }
                    for result in self.eval_results
                ]
            }
            
            with open(final_output, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved final iterative evaluation summary to {final_output}")


def generation_eval_callback(
    tokenizer,
    cfg: omegaconf.DictConfig,
    tag: str = "eval", data_path: str = None
) -> LLMRecEvalCallback:
    return LLMRecEvalCallback(
        tokenizer=tokenizer,
        eval_dataset_path=data_path if data_path is not None else cfg.task.eval.eval_dataset_path,
        sample_number=cfg.task.eval.sample_number,
        task=cfg.task.eval.task,
        few_zero=cfg.task.eval.few_zero,
        max_new_tokens=cfg.task.eval.max_new_tokens,
        debug_print=cfg.task.eval.debug_print,
        temperature=cfg.task.eval.temperature,
        top_p=cfg.task.eval.top_p,
        top_k=cfg.task.eval.top_k,
        output_dir=cfg.experiment.output_dir,
        eval_batch_size=cfg.task.eval.eval_batch_size,
        tag=tag
    )


def iterative_rec_eval_callback(
    tokenizer,
    cfg: omegaconf.DictConfig,
    tag: str = "eval_iterative",
    data_path: str = None,
    sample_number: int = None,
    num_iterations: int = 1
) -> IterativeRecEvalCallback:
    """
    Factory function to create IterativeRecEvalCallback from config.
    
    Args:
        tokenizer: Tokenizer instance
        cfg: Hydra config
        tag: Tag for logging
        data_path: Optional override for dataset path
        sample_number: Optional override for sample number
        num_iterations: Number of iterations per entry (default 1)
    """
    return IterativeRecEvalCallback(
        tokenizer=tokenizer,
        eval_dataset_path=data_path if data_path is not None else cfg.task.eval.eval_dataset_path,
        sample_number=sample_number if sample_number is not None else cfg.task.eval.sample_number,
        max_new_tokens=cfg.task.eval.max_new_tokens,
        debug_print=cfg.task.eval.debug_print,
        temperature=cfg.task.eval.temperature,
        top_p=cfg.task.eval.top_p,
        top_k=cfg.task.eval.top_k,
        output_dir=cfg.experiment.output_dir,
        eval_batch_size=1,  # Typically 1 for iterative tasks
        tag=tag,
        num_iterations=num_iterations,
        enable_thinking=cfg.task.training.enable_thinking
    )


class YesNoNewsRecEvalCallback(TrainerCallback):
    """
    Custom evaluation callback for yes/no news recommendation tasks.
    Evaluates binary classification accuracy for news recommendations.
    """
    
    def __init__(
        self,
        eval_dataset_path: str,
        sample_number: int = 100,
        max_new_tokens: int = 10,  # Short responses for yes/no
        debug_print: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        tokenizer=None,
        output_dir: str = "./output",
        eval_batch_size: int = 8,
        tag: str = "eval_yesno_news",
        enable_thinking: bool = False,
        df: pd.DataFrame = None
    ):
        """
        Initialize the yes/no news recommendation evaluation callback.
        
        Args:
            eval_dataset_path: Path to the evaluation dataset (CSV format)
            sample_number: Number of samples to evaluate
            max_new_tokens: Maximum tokens to generate (should be small for yes/no)
            debug_print: Whether to print debug information
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            tokenizer: Tokenizer instance
            output_dir: Directory to save evaluation results
            eval_batch_size: Batch size for evaluation
            tag: Tag for logging
            enable_thinking: Whether to enable thinking mode
        """
        self.eval_dataset_path = eval_dataset_path
        self.sample_number = sample_number
        self.debug_print = debug_print
        self.eval_results = []
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.df = df if df is not None else self.load_eval_dataset()
        self.sampled_df = self.df.sample(n=min(self.sample_number, len(self.df)), random_state=42)
        self.eval_batch_size = eval_batch_size
        self.tag = tag
        self.enable_thinking = enable_thinking

        self.tokenizer.padding_side = "left"
        print("Eval: Tokeniser padding side: ", self.tokenizer.padding_side)
    def load_eval_dataset(self) -> pd.DataFrame:
        """Load the evaluation dataset from CSV"""
        logger.info(f"Loading evaluation dataset from {self.eval_dataset_path}")
        
        df = pd.read_csv(self.eval_dataset_path)
        
        # Validate required columns
        required_columns = ['question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Required columns not found: {missing_columns}")
        
        # Check if system_prompt column exists
        if 'system_prompt' in df.columns:
            logger.info("Found 'system_prompt' column - will include system messages in evaluation")
        
        logger.info(f"Loaded {len(df)} examples for evaluation")
        return df
    
    def generate_responses(self, model, tokenizer, prompts: List[str], system_prompts: Optional[List[str]] = None) -> List[str]:
        """Generate responses for the given prompts
        
        Args:
            model: The model to use for generation
            tokenizer: The tokenizer to use
            prompts: List of user prompts/questions
            system_prompts: Optional list of system prompts (same length as prompts)
        """
        model.eval()
        responses = []
        print("Eval: Tokeniser padding side: ", tokenizer.padding_side)
        tokenizer.padding_side = "left"
        print("Eval: Tokeniser after padding side: ", tokenizer.padding_side)
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), self.eval_batch_size), desc="Generating yes/no responses"):
                batch_prompts = prompts[i:i + self.eval_batch_size]
                batch_system_prompts = system_prompts[i:i + self.eval_batch_size] if system_prompts else None
                
                # Format prompts with chat template
                formatted_prompts = []
                for j, prompt in enumerate(batch_prompts):
                    messages = []
                    
                    # Add system prompt if available and not empty
                    if batch_system_prompts and j < len(batch_system_prompts):
                        system_prompt = batch_system_prompts[j]
                        if pd.notna(system_prompt) and system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                    
                    # Add user prompt
                    messages.append({"role": "user", "content": prompt})
                    
                    chat_template_kwargs = {
                        "tokenize": False,
                        "add_generation_prompt": True
                    }
                    if not tokenizer.__class__.__name__.endswith("MistralCommonBackend"):
                         chat_template_kwargs["enable_thinking"] = self.enable_thinking

                    formatted_prompt = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
                    formatted_prompts.append(formatted_prompt)
                
                inputs = tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=32768,
                    padding=True
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                
                # Decode only the generated part
                input_lengths = [len(input_ids) for input_ids in inputs['input_ids']]
                for j, output in enumerate(outputs):
                    start_pos = input_lengths[j]
                    response = tokenizer.decode(
                        output[start_pos:],
                        skip_special_tokens=True
                    )
                    
                    if self.debug_print:
                        logger.info(f"Prompt: {batch_prompts[j]}")
                        logger.info(f"Formatted Prompt: {formatted_prompts[j]}")
                        logger.info(f"Response: {response}")
                        logger.info("-" * 50)
                    
                    responses.append(response.strip())
                
                # Free GPU memory after each batch
                del inputs
                del outputs
                torch.cuda.empty_cache()
        
        return responses
    
    def extract_yes_no(self, response: str) -> Optional[str]:
        """
        Extract YES or NO from the response.
        Returns 'YES', 'NO', or None if unable to parse.
        """
        # Strip thinking tags if present
        if "</think>" in response:
            logger.warning(f"Thinking tags found in response: {response}")
            response = response.split("</think>")[1].strip()
        
        response_upper = response.upper().strip()
        
        # Direct match
        if response_upper == "YES":
            return "YES"
        elif response_upper == "NO":
            return "NO"
        
        # Check if YES or NO appears in the response
        if "YES" in response_upper and "NO" not in response_upper:
            return "YES"
        elif "NO" in response_upper and "YES" not in response_upper:
            return "NO"
        
        # If both or neither, return None (ambiguous)
        return None
    
    def evaluate_step(self, model, tokenizer, eval_df: pd.DataFrame, current_epoch: int = 0, is_main_process: bool = True) -> Dict[str, Any]:
        """Perform evaluation on the sampled dataset"""
        prompts = eval_df["question"].tolist()
        targets = eval_df["answer"].tolist()
        
        # Extract system prompts if available
        system_prompts = None
        if 'system_prompt' in eval_df.columns:
            system_prompts = eval_df["system_prompt"].tolist()
        
        # Normalize targets to uppercase YES/NO
        normalized_targets = []
        for target in targets:
            target_str = str(target).upper().strip()
            if "YES" in target_str:
                normalized_targets.append("YES")
            elif "NO" in target_str:
                normalized_targets.append("NO")
            else:
                normalized_targets.append(target_str)
        
        logger.info("Generating yes/no responses...")

        # Use simple generation loop instead of generate_responses to have better control over memory
        responses = []
        
        # Prepare batches
        batch_size = self.eval_batch_size
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating yes/no responses", disable=not is_main_process):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(prompts))
            
            batch_prompts = prompts[start_idx:end_idx]
            batch_system_prompts = system_prompts[start_idx:end_idx] if system_prompts else None
            
            # Format prompts
            formatted_prompts = []
            for j, prompt in enumerate(batch_prompts):
                messages = []
                if batch_system_prompts and j < len(batch_system_prompts):
                    system_prompt = batch_system_prompts[j]
                    if pd.notna(system_prompt) and system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                chat_template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True
                }
                if not tokenizer.__class__.__name__.endswith("MistralCommonBackend"):
                        chat_template_kwargs["enable_thinking"] = self.enable_thinking

                formatted_prompt = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
                formatted_prompts.append(formatted_prompt)
            
            # Tokenize
            inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=4096, # Reduce max length to save memory
                padding=True
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                try:
                    print("torch.distributed.is_initialized(): ", torch.distributed.is_initialized())
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,  # Disable cache to avoid potential shape mismatches in distributed
                        synced_gpus=True if torch.distributed.is_initialized() else False
                    )
                    
                    # Decode
                    input_lengths = [len(input_ids) for input_ids in inputs['input_ids']]
                    for j, output in enumerate(outputs):
                        start_pos = input_lengths[j]
                        response = tokenizer.decode(output[start_pos:], skip_special_tokens=True)
                        responses.append(response.strip())
                        
                except RuntimeError as e:
                    # If we still get the specific "not allocated" error or OOM, log and skip
                    if "not allocated" in str(e) or "out of memory" in str(e):
                        logger.warning(f"Generation error in batch {i}: {str(e)}. Skipping batch.")
                        responses.extend(["ERROR_GEN"] * len(batch_prompts))
                        torch.cuda.empty_cache()
                    else:
                        raise e
            
            # Clean up
            del inputs
            if 'outputs' in locals(): del outputs
            torch.cuda.empty_cache()

        # Extract yes/no predictions
        predictions = [self.extract_yes_no(resp) for resp in responses]

        correct = 0
        valid_predictions = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for pred, target in zip(predictions, normalized_targets):
            if pred is not None:
                valid_predictions += 1
                if pred == target:
                    correct += 1
                    if pred == "YES":
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if pred == "YES":
                        false_positives += 1
                    else:
                        false_negatives += 1
        
        accuracy = correct / valid_predictions if valid_predictions > 0 else 0.0
        valid_rate = valid_predictions / len(predictions) if len(predictions) > 0 else 0.0
        
        # Precision, Recall, F1 for YES class
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        eval_results = {
            "num_samples": len(prompts),
            "accuracy": accuracy,
            "valid_response_rate": valid_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "responses": responses,
            "predictions": predictions,
            "targets": normalized_targets,
            "prompts": prompts
        }

        if is_main_process:
            try:
                pkl.dump(eval_results, open(f"{self.output_dir}/eval_results_{self.tag}_step_{current_epoch}.pkl", "wb"))
                logger.info(f"Saved eval results to {self.output_dir}/eval_results_{self.tag}_step_{current_epoch}.pkl")
            except Exception as e:
                logger.error(f"Failed to save eval results: {str(e)}")
                logger.exception("Full traceback:")
        
        logger.info(f"[{self.tag}] Accuracy: {accuracy:.4f}")
        logger.info(f"[{self.tag}] Valid response rate: {valid_rate:.4f}")
        logger.info(f"[{self.tag}] Precision: {precision:.4f}")
        logger.info(f"[{self.tag}] Recall: {recall:.4f}")
        logger.info(f"[{self.tag}] F1 Score: {f1:.4f}")
        
        # Clear large data from eval_results after saving to reduce memory usage
        # Keep only the metrics for return value
        eval_results_summary = {
            "num_samples": eval_results["num_samples"],
            "accuracy": accuracy,
            "valid_response_rate": valid_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        }
        
        # Free memory
        del responses
        del predictions
        del normalized_targets
        del prompts
        
        return eval_results_summary
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training to establish baseline"""
        logger.info(f"[{self.tag}] Starting initial baseline yes/no evaluation...")
        model = kwargs.get('model')
        model.eval()
        
        try:
            if self.tokenizer is None:
                logger.error("Tokenizer not available for evaluation callback")
                return
            
            if len(self.df) == 0:
                logger.warning("No evaluation data found, skipping evaluation")
                return
            
            # Perform evaluation at step 0
            eval_results = self.evaluate_step(model, self.tokenizer, self.sampled_df, current_epoch=0, is_main_process=state.is_world_process_zero)
            
            # Store results
            eval_results["step"] = 0
            eval_results["epoch"] = 0
            self.eval_results.append(eval_results)
            
            # Log to wandb if available
            if state.is_world_process_zero and hasattr(args, 'report_to') and 'wandb' in args.report_to:
                import wandb
                for key, value in eval_results.items():
                    # Skip logging non-numeric values
                    if key not in ['responses', 'predictions', 'targets', 'prompts']:
                        wandb.log({f"{self.tag}/{key}": value}, step=0)
            
            logger.info(f"[{self.tag}] Baseline evaluation completed")
            
        except Exception as e:
            logger.error(f"Error during baseline yes/no evaluation: {str(e)}")
            logger.exception("Full traceback:")
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each epoch"""
        logger.info(f"[{self.tag}] Starting epoch end yes/no evaluation...")
        model = kwargs.get('model')
        model.eval()
        current_epoch = state.epoch
        
        try:
            if self.tokenizer is None:
                logger.error("Tokenizer not available for evaluation callback")
                return
            
            if len(self.df) == 0:
                logger.warning("No evaluation data found, skipping evaluation")
                return
            
            # Perform evaluation
            eval_results = self.evaluate_step(model, self.tokenizer, self.sampled_df, current_epoch, is_main_process=state.is_world_process_zero)
            
            # Store results
            eval_results["step"] = state.global_step
            eval_results["epoch"] = state.epoch
            self.eval_results.append(eval_results)
            
            # Log to wandb if available
            if state.is_world_process_zero and hasattr(args, 'report_to') and 'wandb' in args.report_to:
                import wandb
                for key, value in eval_results.items():
                    # Skip logging non-numeric values
                    if key not in ['responses', 'predictions', 'targets', 'prompts']:
                        wandb.log({f"{self.tag}/{key}": value}, step=state.global_step)
            
            # IMPORTANT: Add metrics to trainer logs for early stopping
            # This makes the metrics available to EarlyStoppingCallback
            metrics_to_log = {}
            for key, value in eval_results.items():
                if key not in ['responses', 'predictions', 'targets', 'prompts', 'step', 'epoch']:
                    # Add prefix to distinguish different callbacks
                    metrics_to_log[f"{self.tag}_{key}"] = value
            
            # Update the trainer's log history
            if metrics_to_log:
                # Add to the trainer's state
                for key, value in metrics_to_log.items():
                    state.log_history[-1][key] = value if state.log_history else None
            
            logger.info(f"[{self.tag}] Evaluation completed at step {state.global_step}")
            
        except Exception as e:
            logger.error(f"Error during yes/no evaluation: {str(e)}")
            logger.exception("Full traceback:")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        if self.eval_results and state.is_world_process_zero:
            os.makedirs(self.output_dir, exist_ok=True)
            
            final_output = os.path.join(self.output_dir, f"yesno_eval_{self.tag}_summary.json")
            
            summary = {
                "total_evaluations": len(self.eval_results),
                "final_step": state.global_step,
                "final_epoch": state.epoch,
                "evaluation_history": [
                    {
                        "step": result["step"],
                        "epoch": result["epoch"],
                        "num_samples": result["num_samples"],
                        "accuracy": result["accuracy"],
                        "f1_score": result["f1_score"]
                    }
                    for result in self.eval_results
                ]
            }
            
            with open(final_output, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved final yes/no evaluation summary to {final_output}")


def yesno_news_rec_eval_callback(
    tokenizer,
    cfg: omegaconf.DictConfig,
    tag: str = "eval_yesno_news",
    data_path: str = None,
    sample_number: int = None
) -> YesNoNewsRecEvalCallback:
    """
    Factory function to create YesNoNewsRecEvalCallback from config.
    
    Args:
        tokenizer: Tokenizer instance
        cfg: Hydra config
        tag: Tag for logging
        data_path: Optional override for dataset path
        sample_number: Optional override for sample number
    """
    return YesNoNewsRecEvalCallback(
        tokenizer=tokenizer,
        eval_dataset_path=data_path if data_path is not None else cfg.task.eval.eval_dataset_path,
        sample_number=sample_number if sample_number is not None else cfg.task.eval.sample_number,
        max_new_tokens=cfg.task.eval.get('max_new_tokens', 10),
        debug_print=cfg.task.eval.debug_print,
        temperature=cfg.task.eval.temperature,
        top_p=cfg.task.eval.top_p,
        top_k=cfg.task.eval.top_k,
        output_dir=cfg.experiment.output_dir,
        eval_batch_size=cfg.task.eval.eval_batch_size,
        tag=tag,
        enable_thinking=cfg.task.training.enable_thinking
    )

