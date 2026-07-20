import json
import os
import time
import hydra
import omegaconf
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import logging
import wandb
import glob
import re
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, TrainerCallback
from accelerate import Accelerator

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Extract step numbers and find the latest
    def extract_step(checkpoint_path):
        match = re.search(r'checkpoint-(\d+)', checkpoint_path)
        return int(match.group(1)) if match else 0
    
    latest_checkpoint = max(checkpoints, key=extract_step)
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def get_wandb_run_id(output_dir):
    """Get the wandb run ID from the training state if it exists"""
    training_state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(training_state_path):
        try:
            with open(training_state_path, 'r') as f:
                state = json.load(f)
                return state.get('wandb_run_id')
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Also check for wandb run ID in the logs directory
    logs_dir = os.path.join(output_dir, "logs")
    if os.path.exists(logs_dir):
        # Look for wandb run files
        wandb_files = glob.glob(os.path.join(logs_dir, "wandb", "run-*"))
        if wandb_files:
            # Extract run ID from the most recent wandb run directory
            latest_wandb_dir = max(wandb_files, key=os.path.getmtime)
            run_id = os.path.basename(latest_wandb_dir).replace("run-", "")
            logger.info(f"Found wandb run ID from logs: {run_id}")
            return run_id
    
    return None


def load_qa_dataset(cfg: DictConfig, tokenizer, path=None):
    """Load and preprocess the QA dataset for SFTTrainer with chat template or prompt-completion format"""
    logger.info(f"Loading dataset from {path if path else cfg.task.dataset.path}")
    
    df = pd.read_csv(path if path else cfg.task.dataset.path)
    logger.info(f"Loaded {len(df)} examples")
    
    # Validate dataset
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    
    required_columns = ['question', 'answer']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns not found: {missing_columns}")
    
    # Check if system_prompt column exists
    has_system_prompt = 'system_prompt' in df.columns
    if has_system_prompt:
        logger.info("Found 'system_prompt' column - will include system messages")
    
    # Check if completion_only_loss is enabled
    completion_only_loss = getattr(cfg.task.training, 'completion_only_loss', False)
    if completion_only_loss:
        logger.info("Using prompt-completion format (completion_only_loss=True)")
    else:
        logger.info("Using chat template format (completion_only_loss=False)")
    
    # Check if using models that need special tokenization handling
    # Gemma and OLMo models have BPE tokenization issues with trailing space
    model_name_lower = cfg.model.name.lower()
    needs_newline_suffix = "gemma" in model_name_lower or "olmo" in model_name_lower
    
    # Convert to chat template format or prompt-completion format for SFTTrainer
    def format_qa_example(row):
        # Ensure question and answer are strings, handling NaN/None values
        question = str(row['question']) if pd.notna(row['question']) else ""
        answer = str(row['answer']) if pd.notna(row['answer']) else ""
        
        # Skip examples with empty question or answer
        if not question or not answer:
            logger.warning(f"Skipping example with empty question or answer. Question: {question[:50] if question else 'None'}, Answer: {answer[:50] if answer else 'None'}")
            return None
        
        if completion_only_loss:
            # Prompt-completion format: separate prompt and completion fields
            # Build prompt from system_prompt (if exists) + question
            # Important: Ensure prompt ends with a space to avoid tokenization mismatch
            # when SFTTrainer concatenates prompt + completion
            prompt_parts = []
            
            # Add system prompt if available and not empty
            if has_system_prompt and pd.notna(row.get('system_prompt')) and row.get('system_prompt'):
                system_prompt = str(row['system_prompt']) if pd.notna(row.get('system_prompt')) else ""
                if system_prompt:
                    prompt_parts.append(system_prompt.strip())
            
            # Add question to prompt
            prompt_parts.append(question.strip())
            
            # Combine prompt parts with newlines
            # For Gemma/OLMo models: Use "\n\n" as suffix to create clean token boundary.
            # BPE tokenizers can merge characters differently when tokenizing prompt alone
            # vs prompt+completion together. Using "\n\n" ensures newlines act as clean
            # token boundaries, preventing TRL's completion_only_loss from failing to
            # align the prompt boundary correctly.
            # For other models: Keep trailing space for reproducibility of previous experiments.
            prompt_suffix = "\n\n" if needs_newline_suffix else " "
            if len(prompt_parts) > 1:
                prompt = "\n\n".join(prompt_parts) + prompt_suffix
            else:
                prompt = prompt_parts[0] + prompt_suffix
            
            # Ensure completion doesn't start with whitespace that could cause issues
            completion = answer.strip()
            
            # Return as separate prompt and completion fields for SFTTrainer
            # SFTTrainer will automatically mask prompt tokens when completion_only_loss=True
            return {"prompt": prompt, "completion": completion}
        else:
            # Chat template format (original behavior)
            # Create a conversation with optional system message, user message and assistant response
            messages = []
            
            # Add system prompt if available and not empty
            if has_system_prompt and pd.notna(row.get('system_prompt')) and row.get('system_prompt'):
                system_prompt = str(row['system_prompt']) if pd.notna(row.get('system_prompt')) else ""
                if system_prompt:  # Only add if not empty after conversion
                    messages.append({"role": "system", "content": system_prompt})
            
            # Add user question and assistant answer
            messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
            
            # Apply the chat template with tokenizer
            # add_generation_prompt=False because we want to include the assistant's response
            
            # Check if tokenizer supports enable_thinking
            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": False
            }
            
            # Handle MistralCommonBackend special requirements
            # MistralCommonBackend requires continue_final_message=True when last message is assistant
            is_mistral_backend = tokenizer.__class__.__name__.endswith("MistralCommonBackend")
            if is_mistral_backend:
                # For MistralCommonBackend, we need continue_final_message=True when training with assistant messages
                chat_template_kwargs["continue_final_message"] = True
            else:
                # Only add enable_thinking if not using MistralCommonBackend
                # MistralCommonBackend does not support enable_thinking
                enable_thinking = getattr(cfg.task.training, 'enable_thinking', False)
                chat_template_kwargs["enable_thinking"] = enable_thinking
            
            try:
                text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
            except Exception as e:
                # Provide more context about the error
                logger.error(f"Error applying chat template. Message types: {[type(m.get('content', 'N/A')) for m in messages]}")
                logger.error(f"Message contents (first 100 chars): {[str(m.get('content', 'N/A'))[:100] for m in messages]}")
                raise ValueError(f"Failed to apply chat template: {e}") from e
            
            return {"text": text}
    
    # Apply formatting to all examples, filtering out None values
    formatted_data = []
    for idx, row in df.iterrows():
        try:
            formatted_example = format_qa_example(row)
            if formatted_example is not None:
                formatted_data.append(formatted_example)
        except Exception as e:
            logger.error(f"Error formatting example at index {idx}: {e}")
            logger.error(f"Row data: question={row.get('question', 'N/A')[:100]}, answer={row.get('answer', 'N/A')[:100]}")
            continue
    
    dataset = Dataset.from_list(formatted_data)

    if cfg.experiment.debug_print:
        # Print column names and first example
        logger.info(f"Dataset column names: {dataset.column_names}")
        if completion_only_loss:
            # Prompt-completion format
            first_example = dataset[0]
            logger.info(f"First formatted example - Prompt (first 50 chars):\n{first_example['prompt'][:50]}...")
            logger.info(f"First formatted example - Completion (first 50 chars):\n{first_example['completion'][:50]}...")
        else:
            # Chat template format
            logger.info(f"First formatted example (first 50 chars):\n{dataset[0]['text'][:50]}...")
    
    return dataset


def sft_v2(cfg: DictConfig):
    """Main SFT function using trl.SFTTrainer"""
    logger.info(f"Starting SFT v2 for {cfg.task.name}")
    log_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    
    # Check if we should continue from checkpoint
    resume_from_checkpoint = None
    wandb_run_id = None
    
    if cfg.wandb.resume:
        logger.info("Continue mode enabled")
        
        # Check if manual resume path is specified
        if cfg.wandb.resume_path is not None:
            resume_from_checkpoint = cfg.wandb.resume_path
            logger.info(f"Using manual resume path: {resume_from_checkpoint}")
            
            # Verify the checkpoint exists and is a directory
            if not os.path.exists(resume_from_checkpoint):
                logger.error(f"Manual resume path does not exist: {resume_from_checkpoint}")
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_from_checkpoint}")
            elif not os.path.isdir(resume_from_checkpoint):
                logger.error(f"Manual resume path is not a directory: {resume_from_checkpoint}")
                raise ValueError(f"Resume path must be a directory: {resume_from_checkpoint}")
            else:
                logger.info(f"Verified manual resume path exists: {resume_from_checkpoint}")
        else:
            # Fall back to automatic checkpoint detection
            logger.info("No manual resume path specified - looking for latest checkpoint")
            resume_from_checkpoint = find_latest_checkpoint(cfg.experiment.output_dir)
            if not resume_from_checkpoint:
                logger.warning("Continue mode enabled but no checkpoint found. Starting fresh training.")
                resume_from_checkpoint = None
        
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            # Try to get wandb run ID for resuming wandb logging
            wandb_run_id = get_wandb_run_id(resume_from_checkpoint)
            
            # If we couldn't find wandb run ID from checkpoint, use manual wandb_id if provided
            if wandb_run_id is None:
                if cfg.wandb.wandb_id is not None and cfg.wandb.wandb_id != "None":
                    wandb_run_id = cfg.wandb.wandb_id
                    logger.info(f"Could not find wandb run ID from checkpoint, using manual wandb_id: {wandb_run_id}")
                else:
                    logger.info("No wandb run ID found from checkpoint and no manual wandb_id provided. Starting new wandb run.")
            else:
                logger.info(f"Found wandb run ID from checkpoint: {wandb_run_id}")
    
    # Set up device and distributed training (must be done before wandb init)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Print and log number of GPUs available
    logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # Determine device_map based on distributed status
    # For FSDP, we don't set device_map and use low_cpu_mem_usage for efficient loading
    # Accelerate's FSDP handles the sharding and device placement
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp_enabled = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0
    
    # Initialize wandb only on main process to avoid multiple runs
    if is_main_process:
        wandb_kwargs = {
            "project": cfg.wandb.project,
            "config": log_config
        }
        
        # Handle wandb run ID properly
        if wandb_run_id and wandb_run_id != "None" and wandb_run_id is not None:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"
            logger.info(f"Resuming wandb run with ID: {wandb_run_id}")
        else:
            wandb_kwargs["resume"] = "allow"
            logger.info("Starting new wandb run")
        
        wandb.init(**wandb_kwargs)
        logger.info(f"Initialized wandb project: {cfg.wandb.project}")
    
    # Check if quantization (QLoRA) or LoRA is enabled
    use_quantization = hasattr(cfg.task.training, 'quantization') and cfg.task.training.quantization
    use_lora = hasattr(cfg.task.training, 'use_lora') and cfg.task.training.use_lora
    
    # If quantization is enabled, LoRA is automatically enabled too
    if use_quantization:
        use_lora = True
    
    if use_quantization:
        logger.info("=" * 60)
        logger.info("QLoRA MODE ENABLED - Using 4-bit quantization + LoRA")
        logger.info("Note: FSDP will be disabled, using device_map='auto' instead")
        logger.info("=" * 60)
        # Force single-process mode for QLoRA (incompatible with FSDP)
        ddp_enabled = False
    elif use_lora:
        logger.info("=" * 60)
        logger.info("LoRA MODE ENABLED - Using full precision (bf16) + LoRA adapters")
        logger.info("Compatible with FSDP distributed training")
        logger.info("=" * 60)
    
    if ddp_enabled:
        logger.info(f"Distributed training detected (world_size={world_size}, local_rank={local_rank}).")
    else:
        logger.info("Single process training detected. Setting device_map='auto' for model parallelism.")

    # Configure quantization (4-bit QLoRA)
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load model and tokenizer
    if resume_from_checkpoint:
        logger.info(f"Loading model from checkpoint: {resume_from_checkpoint}")
        if use_quantization:
            model = AutoModelForCausalLM.from_pretrained(
                resume_from_checkpoint,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        elif ddp_enabled:
            # For FSDP: load to CPU, FSDP will shard
            model = AutoModelForCausalLM.from_pretrained(
                resume_from_checkpoint,
                torch_dtype=torch.bfloat16 if cfg.task.training.bf16 else (torch.float16 if cfg.task.training.fp16 else torch.float32),
                low_cpu_mem_usage=True,  # Required for FSDP
            )
        else:
            # For single GPU / no FSDP: use device_map="auto"
            model = AutoModelForCausalLM.from_pretrained(
                resume_from_checkpoint,
                torch_dtype=torch.bfloat16 if cfg.task.training.bf16 else (torch.float16 if cfg.task.training.fp16 else torch.float32),
                device_map="auto",
            )
        tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint)
    else:
        logger.info(f"Loading model: {cfg.model.name}")
        
        # Check if using Mistral/Mixtral models that need specific backend
        is_mistral = "mistral" in cfg.model.name.lower() or "mixtral" in cfg.model.name.lower() or "ministral" in cfg.model.name.lower()
        
        if is_mistral:
            try:
                from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
                logger.info("Using MistralCommonBackend for tokenizer")
                tokenizer = MistralCommonBackend.from_pretrained(cfg.model.path + "/" + cfg.model.name)
            except ImportError:
                logger.warning("Could not import MistralCommonBackend, falling back to AutoTokenizer")
                tokenizer = AutoTokenizer.from_pretrained(cfg.model.path + "/" + cfg.model.name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.path + "/" + cfg.model.name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if is_mistral:
            try:
                from transformers import Mistral3ForConditionalGeneration
                logger.info("Using Mistral3ForConditionalGeneration for model")
                if use_quantization:
                    model = Mistral3ForConditionalGeneration.from_pretrained(
                        cfg.model.path + "/" + cfg.model.name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                    )
                elif ddp_enabled:
                    model = Mistral3ForConditionalGeneration.from_pretrained(
                        cfg.model.path + "/" + cfg.model.name,
                        torch_dtype=torch.bfloat16 if cfg.task.training.bf16 else (torch.float16 if cfg.task.training.fp16 else torch.float32),
                        low_cpu_mem_usage=True,  # Required for FSDP
                    )
                else:
                    model = Mistral3ForConditionalGeneration.from_pretrained(
                        cfg.model.path + "/" + cfg.model.name,
                        torch_dtype=torch.bfloat16 if cfg.task.training.bf16 else (torch.float16 if cfg.task.training.fp16 else torch.float32),
                        device_map="auto",
                    )
            except ImportError:
                logger.warning("Could not import Mistral3ForConditionalGeneration, falling back to AutoModelForCausalLM")
                if use_quantization:
                    model = AutoModelForCausalLM.from_pretrained(
                        cfg.model.path + "/" + cfg.model.name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                    )
                elif ddp_enabled:
                    model = AutoModelForCausalLM.from_pretrained(
                        cfg.model.path + "/" + cfg.model.name,
                        torch_dtype=torch.bfloat16 if cfg.task.training.bf16 else (torch.float16 if cfg.task.training.fp16 else torch.float32),
                        low_cpu_mem_usage=True,  # Required for FSDP
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        cfg.model.path + "/" + cfg.model.name,
                        torch_dtype=torch.bfloat16 if cfg.task.training.bf16 else (torch.float16 if cfg.task.training.fp16 else torch.float32),
                        device_map="auto",
                    )
        else:
            if use_quantization:
                # For QLoRA: use device_map="auto" for automatic multi-GPU placement
                logger.info("Loading model with 4-bit quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model.path + "/" + cfg.model.name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            elif ddp_enabled:
                # For FSDP: use low_cpu_mem_usage to load to CPU, FSDP will shard to GPUs
                logger.info("Loading model for FSDP (CPU loading, FSDP will shard)...")
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model.path + "/" + cfg.model.name,
                    torch_dtype=torch.bfloat16 if cfg.task.training.bf16 else (torch.float16 if cfg.task.training.fp16 else torch.float32),
                    low_cpu_mem_usage=True,  # Critical for FSDP - keeps model on CPU until sharding
                )
            else:
                # For single GPU / no FSDP: use device_map="auto" for automatic GPU placement
                logger.info("Loading model with device_map='auto' (single GPU / no FSDP)...")
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model.path + "/" + cfg.model.name,
                    torch_dtype=torch.bfloat16 if cfg.task.training.bf16 else (torch.float16 if cfg.task.training.fp16 else torch.float32),
                    device_map="auto",
                )
    
    # Apply LoRA adapters if using LoRA (with or without quantization)
    if use_lora:
        logger.info("Adding LoRA adapters to the model...")
        
        # Only prepare for k-bit training if using quantization
        if use_quantization:
            logger.info("Preparing model for k-bit (quantized) training...")
            model = prepare_model_for_kbit_training(model)
        
        # Get LoRA config from task config or use defaults
        lora_r = getattr(cfg.task.training, 'lora_r', 64)
        lora_alpha = getattr(cfg.task.training, 'lora_alpha', 16)
        lora_dropout = getattr(cfg.task.training, 'lora_dropout', 0.1)
        
        # Target modules for common architectures (Qwen, Llama, Mistral, etc.)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"LoRA target modules: {target_modules}")
        logger.info(f"LoRA mode: {'QLoRA (4-bit quantized)' if use_quantization else 'Full precision (bf16)'}")

    # DEBUG: Check model device and memory after loading
    def debug_model_state(model, stage=""):
        logger.info(f"=== DEBUG {stage} (rank {local_rank}) ===")
        # Check device of first parameter
        first_param = next(model.parameters())
        logger.info(f"  First param device: {first_param.device}")
        logger.info(f"  First param dtype: {first_param.dtype}")
        logger.info(f"  Model type: {type(model).__name__}")
        # Check GPU memory
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            logger.info(f"  GPU {i}: {alloc:.2f} GB allocated")
    
    debug_model_state(model, "AFTER MODEL LOAD")
    
    # Load and preprocess dataset (must be done after tokenizer is loaded)
    # In distributed mode, stagger dataset loading to avoid OOM from all processes loading simultaneously
    is_main_process = local_rank == 0
    
    if ddp_enabled:
        # Initialize process group if not already initialized (accelerate usually does this)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        
        # Only main process loads and processes dataset first
        if is_main_process:
            logger.info("Main process loading and preprocessing dataset with chat template")
            dataset = load_qa_dataset(cfg, tokenizer)
            dataset = dataset.shuffle(seed=cfg.experiment.seed)
            train_dataset = dataset.select(range(min(cfg.task.dataset.down_sample_size, len(dataset))))
            eval_dataset = load_qa_dataset(cfg, tokenizer, path=cfg.task.eval.eval_synthetic_dataset_path)
            eval_dataset = eval_dataset.shuffle(seed=cfg.experiment.seed)
            eval_dataset = eval_dataset.select(range(min(cfg.task.eval.sample_number, len(eval_dataset))))
        
        # Wait for main process to finish
        torch.distributed.barrier()
        
        # Other processes now load (will benefit from any OS-level caching)
        if not is_main_process:
            logger.info(f"Process {local_rank} loading dataset after main process")
            dataset = load_qa_dataset(cfg, tokenizer)
            dataset = dataset.shuffle(seed=cfg.experiment.seed)
            train_dataset = dataset.select(range(min(cfg.task.dataset.down_sample_size, len(dataset))))
            eval_dataset = load_qa_dataset(cfg, tokenizer, path=cfg.task.eval.eval_synthetic_dataset_path)
            eval_dataset = eval_dataset.shuffle(seed=cfg.experiment.seed)
            eval_dataset = eval_dataset.select(range(min(cfg.task.eval.sample_number, len(eval_dataset))))
        
        # Final barrier to ensure all processes have loaded
        torch.distributed.barrier()
    else:
        logger.info("Loading and preprocessing dataset with chat template")
        dataset = load_qa_dataset(cfg, tokenizer)
        dataset = dataset.shuffle(seed=cfg.experiment.seed)
        train_dataset = dataset.select(range(min(cfg.task.dataset.down_sample_size, len(dataset))))
        eval_dataset = load_qa_dataset(cfg, tokenizer, path=cfg.task.eval.eval_synthetic_dataset_path)
        eval_dataset = eval_dataset.shuffle(seed=cfg.experiment.seed)
        eval_dataset = eval_dataset.select(range(min(cfg.task.eval.sample_number, len(eval_dataset))))

    if cfg.experiment.debug_print and is_main_process:
        logger.info(f"First train example: {train_dataset[0]}")
        logger.info(f"First eval example: {eval_dataset[0]}")
    
    #add early stopping callback
    callbacks = []
    if cfg.task.training.use_early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg.task.training.early_stopping_patience,
            early_stopping_threshold=cfg.task.training.early_stopping_threshold
        ))
    # Load custom evaluation callback if specified
    if hasattr(cfg.task, 'eval') and hasattr(cfg.task.eval, 'entry_point') and cfg.task.eval.entry_point and cfg.task.eval.entry_point != "None":
        custom_eval_callback_instantializer = hydra.utils.get_method(cfg.task.eval.entry_point)

        train_eval_path = getattr(cfg.task.eval, "eval_synthetic_train_dataset_path", None)
        if train_eval_path and str(train_eval_path) != "None":
            custom_eval_callback = custom_eval_callback_instantializer(
                tokenizer=tokenizer,
                cfg=cfg,
                tag="eval_synthetic_train_split",
                data_path=train_eval_path,
            )

            callbacks.append(custom_eval_callback)


        custom_eval_callback2 = custom_eval_callback_instantializer(
            tokenizer=tokenizer,
            cfg=cfg,
            tag="eval_synthetic_eval_split",
            data_path=cfg.task.eval.eval_synthetic_dataset_path,
        )

        callbacks.append(custom_eval_callback2)


        custom_eval_callback3 = custom_eval_callback_instantializer(
            tokenizer=tokenizer,
            cfg=cfg,
            tag="eval_downstream"
        )
        callbacks.append(custom_eval_callback3)


    else:
        print("No eval entry_point defined in config")
        pass
    

    
    # SFT configuration with training arguments
    # Note: torch_compile is disabled when using FSDP as they have compatibility issues
    
    # FSDP configuration for Trainer (explicit settings to ensure proper sharding)
    # Note: FSDP is disabled when using quantization (QLoRA)
    fsdp_config = None
    if ddp_enabled and not use_quantization:
        fsdp_config = {
            "transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "use_orig_params": True,
            "sync_module_states": True,
            "cpu_ram_efficient_loading": True,
            "activation_checkpointing": True,
        }
    
    # Check if completion_only_loss is enabled
    completion_only_loss = getattr(cfg.task.training, 'completion_only_loss', False)
    
    training_args = SFTConfig(
        output_dir=cfg.experiment.output_dir,
        num_train_epochs=cfg.task.training.num_train_epochs,
        per_device_train_batch_size=cfg.task.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.task.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.task.training.gradient_accumulation_steps,
        learning_rate=cfg.task.training.learning_rate,
        weight_decay=cfg.task.training.weight_decay,
        warmup_ratio=cfg.task.training.warmup_ratio,
        logging_steps=cfg.task.training.logging_steps,
        save_steps=cfg.task.training.save_steps,
        eval_steps=cfg.task.training.eval_steps,
        save_total_limit=cfg.task.training.save_total_limit,
        load_best_model_at_end=cfg.task.training.load_best_model_at_end,
        metric_for_best_model=cfg.task.training.metric_for_best_model,
        greater_is_better=cfg.task.training.greater_is_better,
        fp16=cfg.task.training.fp16,
        bf16=cfg.task.training.bf16,
        dataloader_num_workers=cfg.task.training.dataloader_num_workers,
        remove_unused_columns=False,  # SFTTrainer needs this
        eval_strategy=getattr(cfg.task.training, "eval_strategy", "steps"),
        save_strategy=getattr(cfg.task.training, "save_strategy", "epoch"),
        logging_dir=f"{cfg.experiment.output_dir}/logs",
        report_to="wandb",  # Enable wandb logging
        dataset_text_field="text" if not completion_only_loss else None,  # Use text field for chat template, None for prompt/completion
        dataset_num_proc=None,  # Optional: for parallel processing
        packing=False,  # Set to True for more efficient training if supported
        torch_compile=False,  # Disable torch_compile - not compatible with FSDP
        gradient_checkpointing=True if use_quantization else (False if ddp_enabled else True),  # Enable for QLoRA, use FSDP's activation_checkpointing otherwise
        max_length=cfg.task.training.max_length,  # Truncate sequences to prevent OOM
        completion_only_loss=completion_only_loss,  # Enable completion-only loss for prompt-completion format
        # FSDP settings for Trainer (disabled for QLoRA)
        fsdp="full_shard auto_wrap" if (ddp_enabled and not use_quantization) else "",
        fsdp_config=fsdp_config,
    )
    
    # Initialize SFTTrainer with chat-templated dataset or prompt-completion format
    # When completion_only_loss=False: uses "text" field with chat template
    # When completion_only_loss=True: uses "prompt" and "completion" fields
    # SFTTrainer automatically detects the format based on dataset fields and completion_only_loss config
    if completion_only_loss:
        logger.info("Using prompt-completion format with fields: 'prompt' and 'completion'")
    else:
        logger.info("Using chat template format with field: 'text'")
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks
    )
    print("Trainer: Tokeniser padding side: ", tokenizer.padding_side)
    
    # DEBUG: Check model after trainer creation (before FSDP prepare)
    debug_model_state(trainer.model, "AFTER TRAINER CREATION")
    
    # DEBUG: Check if model is FSDP wrapped
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    is_fsdp = isinstance(trainer.model, FSDP)
    logger.info(f"=== DEBUG: Is model FSDP wrapped? {is_fsdp} ===")
    if is_fsdp:
        logger.info(f"  FSDP sharding strategy: {trainer.model.sharding_strategy}")
    
    # Create output directory
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    
    # Add callback to debug FSDP state after prepare()
    class FSDPDebugCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            logger.info("=== DEBUG ON_TRAIN_BEGIN (after accelerator.prepare) ===")
            is_fsdp = isinstance(model, FSDP)
            logger.info(f"  Is model FSDP wrapped? {is_fsdp}")
            if model is not None:
                try:
                    first_param = next(model.parameters())
                    logger.info(f"  First param device: {first_param.device}")
                except StopIteration:
                    logger.info("  No parameters found")
            self._log_gpu_memory("TRAIN_BEGIN")
        
        def on_step_end(self, args, state, control, **kwargs):
            # Log GPU memory every step
            self._log_gpu_memory(f"Step {state.global_step}")
        
        def _log_gpu_memory(self, stage):
            """Log GPU memory usage for all available GPUs"""
            mem_info = []
            for i in range(min(torch.cuda.device_count(), 8)):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                mem_info.append(f"GPU{i}:{alloc:.1f}/{reserved:.1f}GB")
            logger.info(f"[{stage}] Memory (alloc/reserved): {' | '.join(mem_info)}")
    
    trainer.add_callback(FSDPDebugCallback())
    
    # Start training
    logger.info("Starting training...")
    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        logger.info("Starting fresh training")
        trainer.train()
    
    # Log final metrics to wandb (only on main process)
    final_metrics = trainer.evaluate()
    if is_main_process:
        wandb.log(final_metrics)
    logger.info(f"Final evaluation metrics: {final_metrics}")
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(cfg.experiment.output_dir)

    # Finish wandb run (only on main process)
    if is_main_process:
        wandb.finish()
    
    logger.info(f"Training completed! Model saved to {cfg.experiment.output_dir}")


if __name__ == "__main__":
    sft_v2()
