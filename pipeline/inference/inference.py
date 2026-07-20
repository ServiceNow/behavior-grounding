
import os
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.distributed import DistributedConfig
except Exception:
    DistributedConfig = None
import pickle as pkl
import json
import time
import requests
import re
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from vllm.lora.request import LoRARequest
except Exception:
    LoRARequest = None


def _cfg_get(cfg, key: str, default=None):
    """
    Safe nested-ish access for OmegaConf/DictConfig without hard dependency.
    Supports dotted keys like "model.repetition_penalty".
    """
    cur = cfg
    for part in key.split("."):
        try:
            if cur is None:
                return default
            cur = cur.get(part)
        except Exception:
            try:
                cur = getattr(cur, part)
            except Exception:
                return default
    return default if cur is None else cur


def _normalize_optional_list(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _build_chat_messages_from_prompt(prompt: str, use_system_prompt: bool):
    """
    Build chat messages for apply_chat_template.
    When use_system_prompt is enabled, split instruction vs query so the model
    receives system guidance in the system role instead of as user text.
    """
    if not use_system_prompt:
        return [{"role": "user", "content": prompt}]

    split_markers = [
        "\nUser Profile:",
        "\nQuestion:",
    ]
    for marker in split_markers:
        idx = prompt.find(marker)
        if idx > 0:
            system_text = prompt[:idx].strip()
            user_text = prompt[idx + 1 :].strip()
            if system_text and user_text:
                return [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ]

    # Fallback: first line as system instruction, rest as user request.
    if "\n" in prompt:
        first_line, rest = prompt.split("\n", 1)
        first_line = first_line.strip()
        rest = rest.strip()
        if first_line and rest:
            return [
                {"role": "system", "content": first_line},
                {"role": "user", "content": rest},
            ]

    # Final fallback preserves prior behavior.
    return [{"role": "user", "content": prompt}]


def _is_lora_adapter_checkpoint(path: str) -> bool:
    if not path:
        return False
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "adapter_config.json"))
        and os.path.exists(os.path.join(path, "adapter_model.safetensors"))
    )


def _load_adapter_metadata(adapter_path: str):
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model = adapter_cfg.get("base_model_name_or_path")
    lora_rank = adapter_cfg.get("r", 16)
    return base_model, lora_rank


def _vllm_generate_with_optional_lora(model, prompts, sampling_params):
    lora_request = getattr(model, "_default_lora_request", None)
    if lora_request is not None:
        return model.generate(prompts, sampling_params, lora_request=lora_request)
    return model.generate(prompts, sampling_params)


def _strip_known_prefix_markers(response: str) -> str:
    text = response
    lower_text = text.lower()
    for marker in ("assistantfinal", "assistantanalysis"):
        marker_idx = lower_text.rfind(marker)
        if marker_idx != -1:
            text = text[marker_idx + len(marker):].strip()
            lower_text = text.lower()
    return text.strip()


def _extract_code_fence_payloads(text: str):
    pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    return [m.group(1).strip() for m in pattern.finditer(text) if m.group(1).strip()]


def _extract_balanced_json_candidates(text: str):
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start_positions = [i for i, ch in enumerate(text) if ch == opener]
        for start in start_positions:
            depth = 0
            in_str = False
            escape = False
            quote_char = ""
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == quote_char:
                        in_str = False
                    continue
                if ch in ('"', "'"):
                    in_str = True
                    quote_char = ch
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:idx + 1].strip())
                        break
    # Prefer longer candidates first so we keep the fullest JSON blob.
    candidates.sort(key=len, reverse=True)
    return candidates


def _try_parse_json_candidate(candidate: str):
    candidate = candidate.strip()
    if not candidate:
        return None

    # First attempt: strict JSON.
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Second attempt: remove common trailing comma issues.
    try:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(cleaned)
    except Exception:
        pass

    # Third attempt: Python-literal style dict/list with single quotes.
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass
    return None


def robust_json_response_parser(response):
    """
    Extract and parse JSON from noisy model outputs.
    Handles:
    - reasoning + assistantfinal / assistantanalysis wrappers
    - markdown code fences
    - embedded JSON inside larger text
    - single-quoted Python-literal dict/list outputs
    """
    if response is None:
        return response
    if isinstance(response, (dict, list)):
        return response

    text = str(response).strip()
    if not text:
        return response

    text = _strip_known_prefix_markers(text)

    # Try parsing the whole string first.
    parsed = _try_parse_json_candidate(text)
    if parsed is not None:
        return parsed

    # Then try markdown fenced blocks.
    for payload in _extract_code_fence_payloads(text):
        parsed = _try_parse_json_candidate(payload)
        if parsed is not None:
            return parsed

    # Finally, try balanced JSON-like substrings.
    for candidate in _extract_balanced_json_candidates(text):
        parsed = _try_parse_json_candidate(candidate)
        if parsed is not None:
            return parsed

    return response


def qwen_model_response_parser(response):
    try:
        output = response.split("{")[-1]
        output = json.loads("{" + output)
    except Exception as e:
        output = response
    return output


def initialise_llm_transformers(cfg):
    device_map = {
        # Enable Tensor Parallelism
        "tp_plan": "auto",
    }
    if DistributedConfig is not None:
        # Enable Expert Parallelism when the installed transformers version supports it.
        device_map["distributed_config"] = DistributedConfig(enable_expert_parallel=1)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.path + "/" + cfg.model.name,
            torch_dtype="auto",
            **device_map
        )

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.path + "/" + cfg.model.name
        )
    except Exception as e:
        print(f"Error initialising model: {e} with path: ", cfg.model.path + "/" + cfg.model.name)
        print("Trying to initialise model with path: " + cfg.model.path)
        try: 
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.path,
                torch_dtype="auto",
                **device_map
            )
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.path
            )
        except Exception as e:
            print(f"Error initialising model: {e}")
            raise e

    return model, tokenizer

def initialise_llm_vllm(cfg):
    candidate_paths = [
        cfg.model.path + "/" + cfg.model.name,
        cfg.model.path,
    ]
    tried = []

    for candidate in candidate_paths:
        try:
            is_adapter = _is_lora_adapter_checkpoint(candidate)
            model_id = candidate
            tokenizer_id = candidate
            llm_kwargs = {
                "tensor_parallel_size": torch.cuda.device_count()
            }

            if is_adapter:
                if LoRARequest is None:
                    raise ImportError(
                        "LoRA adapter checkpoint detected but vLLM LoRA support is unavailable."
                    )
                base_model, lora_rank = _load_adapter_metadata(candidate)
                if not base_model:
                    raise ValueError(
                        f"LoRA adapter config missing base_model_name_or_path: {candidate}"
                    )
                model_id = base_model
                llm_kwargs["enable_lora"] = True
                llm_kwargs["max_lora_rank"] = max(16, int(lora_rank))
                # Use adapter tokenizer when available; otherwise fall back to base model tokenizer.
                if not os.path.exists(os.path.join(candidate, "tokenizer_config.json")):
                    tokenizer_id = base_model
                print(
                    f"Detected LoRA adapter checkpoint at {candidate}. "
                    f"Loading base model {base_model} with rank {llm_kwargs['max_lora_rank']}."
                )

            model = LLM(
                model=model_id,
                **llm_kwargs
            )

            if is_adapter:
                model._default_lora_request = LoRARequest("default_adapter", 1, candidate)

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            return model, tokenizer
        except Exception as e:
            tried.append(f"{candidate}: {str(e)}")
            print(f"Error initialising model candidate {candidate}: {e}")

    raise RuntimeError("Failed to initialise vLLM model. Errors: " + " | ".join(tried))


def vllm_run_inference(model, tokenizer, cfg, prompts, final_return_list=None, self_incremented_seed=False):
    if self_incremented_seed:
        assert cfg.inference.batch_size == 1, "Batch size must be 1 when using self_incremented_seed"
    task_use_chat_template = _cfg_get(cfg, "task.vllm_use_chat_template", None)
    if task_use_chat_template is None:
        use_chat_template = bool(_cfg_get(cfg, "inference.vllm_use_chat_template", False))
    else:
        use_chat_template = bool(task_use_chat_template)

    task_chat_template_enable_thinking = _cfg_get(
        cfg, "task.vllm_chat_template_enable_thinking", None
    )
    if task_chat_template_enable_thinking is None:
        chat_template_enable_thinking = bool(
            _cfg_get(cfg, "inference.vllm_chat_template_enable_thinking", False)
        )
    else:
        chat_template_enable_thinking = bool(task_chat_template_enable_thinking)

    task_chat_template_use_system_prompt = _cfg_get(
        cfg, "task.vllm_chat_template_use_system_prompt", None
    )
    if task_chat_template_use_system_prompt is None:
        chat_template_use_system_prompt = bool(
            _cfg_get(cfg, "inference.vllm_chat_template_use_system_prompt", False)
        )
    else:
        chat_template_use_system_prompt = bool(task_chat_template_use_system_prompt)
    chat_template_warning_printed = False

    if use_chat_template:
        print(
            "vLLM chat template mode is enabled "
            f"(enable_thinking={chat_template_enable_thinking}, "
            f"use_system_prompt={chat_template_use_system_prompt})."
        )
        if tokenizer is None:
            print("Warning: tokenizer is None, disabling vLLM chat template mode.")
            use_chat_template = False

    max_context_length = _cfg_get(cfg, "model.max_model_len", None) or _cfg_get(cfg, "model.max_length", None)
    configured_max_tokens = _cfg_get(cfg, "model.max_tokens", None)
    if max_context_length and configured_max_tokens is not None:
        effective_max_tokens = min(configured_max_tokens, max(1, max_context_length - 1))
    else:
        effective_max_tokens = configured_max_tokens

    generation_budget_tokens = effective_max_tokens
    if max_context_length and configured_max_tokens is not None and configured_max_tokens >= max_context_length:
        # Guardrail: reserving an entire context window for generation truncates prompts to ~1 token.
        generation_budget_tokens = min(4096, max(256, max_context_length // 8))
        print(
            "Warning: model.max_tokens >= model.max_model_len. "
            f"Using a safer generation budget ({generation_budget_tokens}) to preserve prompt context."
        )

    # Set sampling parameters
    if not self_incremented_seed:
        stop = _normalize_optional_list(_cfg_get(cfg, "model.stop", None))
        stop_token_ids = _normalize_optional_list(_cfg_get(cfg, "model.stop_token_ids", None))
        sampling_params = SamplingParams(
            temperature=cfg.model.temperature,
            top_p=cfg.model.top_p,
            top_k=cfg.model.top_k,
            max_tokens=generation_budget_tokens,
            repetition_penalty=_cfg_get(cfg, "model.repetition_penalty", 1.0),
            presence_penalty=_cfg_get(cfg, "model.presence_penalty", 0.0),
            frequency_penalty=_cfg_get(cfg, "model.frequency_penalty", 0.0),
            stop=stop,
            stop_token_ids=stop_token_ids,
        )

    # Run inference
    print("Running VLLM inference...")
    print(f"Number of prompts: {len(prompts)}")
    middle_results = []
    results = []
    
    # Get max context length for truncation
    max_generation_tokens = generation_budget_tokens
    
    for i in tqdm(range(0, len(prompts), cfg.inference.batch_size)):
        original_batch_prompts = prompts[i:i + cfg.inference.batch_size]
        batch_outputs = None
        batch_results = [None] * len(original_batch_prompts)
        batch_prompts = []
        batch_prompt_indices = []
        
        # Reserve space for generation budget to avoid prompt+generation overflow.
        allowed_input_tokens = max_context_length
        if max_context_length and max_generation_tokens is not None:
            allowed_input_tokens = max(1, max_context_length - max_generation_tokens)

        # Skip overly long prompts to avoid judging on partial/truncated inputs.
        if max_context_length and tokenizer:
            for local_idx, prompt in enumerate(original_batch_prompts):
                prompt_for_generation = prompt
                if use_chat_template:
                    try:
                        messages = _build_chat_messages_from_prompt(
                            prompt,
                            use_system_prompt=chat_template_use_system_prompt,
                        )
                        prompt_for_generation = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=chat_template_enable_thinking,
                        )
                    except TypeError:
                        # Some tokenizers do not support enable_thinking; fall back gracefully.
                        messages = _build_chat_messages_from_prompt(
                            prompt,
                            use_system_prompt=chat_template_use_system_prompt,
                        )
                        prompt_for_generation = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception as e:
                        if not chat_template_warning_printed:
                            print(
                                "Warning: failed to apply chat template in vLLM path; "
                                f"falling back to raw prompts. Error: {e}"
                            )
                            chat_template_warning_printed = True
                        prompt_for_generation = prompt

                tokens = tokenizer.encode(prompt_for_generation, add_special_tokens=False)
                if len(tokens) > allowed_input_tokens:
                    skip_msg = (
                        f"[SKIPPED: Prompt too long for configured context window "
                        f"(prompt_tokens={len(tokens)}, allowed_input_tokens={allowed_input_tokens})]"
                    )
                    batch_results[local_idx] = skip_msg
                    print(
                        f"Warning: Skipping prompt at index {i + local_idx} "
                        f"(tokens={len(tokens)}, allowed={allowed_input_tokens})."
                    )
                    continue
                batch_prompts.append(prompt_for_generation)
                batch_prompt_indices.append(local_idx)
        else:
            if use_chat_template and tokenizer:
                batch_prompts = []
                for prompt in original_batch_prompts:
                    try:
                        messages = _build_chat_messages_from_prompt(
                            prompt,
                            use_system_prompt=chat_template_use_system_prompt,
                        )
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=chat_template_enable_thinking,
                        )
                    except TypeError:
                        messages = _build_chat_messages_from_prompt(
                            prompt,
                            use_system_prompt=chat_template_use_system_prompt,
                        )
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception as e:
                        if not chat_template_warning_printed:
                            print(
                                "Warning: failed to apply chat template in vLLM path; "
                                f"falling back to raw prompts. Error: {e}"
                            )
                            chat_template_warning_printed = True
                        formatted_prompt = prompt
                    batch_prompts.append(formatted_prompt)
            else:
                batch_prompts = original_batch_prompts
            batch_prompt_indices = list(range(len(original_batch_prompts)))

        if self_incremented_seed:
            sampling_params = [
                SamplingParams(
                    temperature=cfg.model.temperature,
                    top_p=cfg.model.top_p,
                    top_k=cfg.model.top_k,
                    max_tokens=generation_budget_tokens,
                    repetition_penalty=_cfg_get(cfg, "model.repetition_penalty", 1.0),
                    presence_penalty=_cfg_get(cfg, "model.presence_penalty", 0.0),
                    frequency_penalty=_cfg_get(cfg, "model.frequency_penalty", 0.0),
                    stop=_normalize_optional_list(_cfg_get(cfg, "model.stop", None)),
                    stop_token_ids=_normalize_optional_list(_cfg_get(cfg, "model.stop_token_ids", None)),
                    seed=cfg.experiment.seed + i + batch_prompt_indices[k]
                ) for k in range(len(batch_prompts))
            ]
            
        
        if batch_prompts:
            try:
                batch_outputs = _vllm_generate_with_optional_lora(model, batch_prompts, sampling_params)
                # Process successful batch outputs
                for j, output in enumerate(batch_outputs):
                    response = output.outputs[0].text.strip()
                    batch_results[batch_prompt_indices[j]] = response
            except Exception as e:
                # Handle recoverable batch errors by retrying each prompt individually.
                err_str = str(e)
                recoverable_error = (
                    "Invalid prefix encountered while decoding" in err_str
                    or "Invalid prefix" in err_str
                    or "Sampled token IDs exceed the max model length" in err_str
                    or "exceed the max model length" in err_str
                )
                if recoverable_error:
                    print(f"Warning: Recoverable vLLM error encountered for batch starting at index {i}. Processing prompts individually...\n{err_str[:400]}")
                    for prompt_idx, prompt in enumerate(batch_prompts):
                        local_idx = batch_prompt_indices[prompt_idx]
                        try:
                            if self_incremented_seed:
                                single_sampling_params = sampling_params[prompt_idx]
                            else:
                                single_sampling_params = sampling_params
                            single_output = _vllm_generate_with_optional_lora(model, [prompt], single_sampling_params)
                            response = single_output[0].outputs[0].text.strip()
                            batch_results[local_idx] = response
                        except Exception as inner_e:
                            inner_err_str = str(inner_e)
                            inner_recoverable_error = (
                                "Invalid prefix encountered while decoding" in inner_err_str
                                or "Invalid prefix" in inner_err_str
                                or "Sampled token IDs exceed the max model length" in inner_err_str
                                or "exceed the max model length" in inner_err_str
                            )
                            if inner_recoverable_error:
                                print(f"Warning: Skipping prompt at index {i + local_idx} due to recoverable generation error: {inner_err_str[:200]}")
                                # Add error placeholder to maintain indexing.
                                batch_results[local_idx] = "[ERROR: Generation skipped due to recoverable vLLM error]"
                            else:
                                # Re-raise if it's a different error
                                raise inner_e
                else:
                    # Re-raise if it's a different error
                    raise e
                
        if cfg.experiment.debug_print and batch_outputs is not None:
            for i, prompt in enumerate(batch_prompts):
                print("Batch Prompt:")
                print(prompt)
                print("--------------------------------")
                print("Model Output:")
                print(batch_outputs[i].outputs[0].text.strip())
            print("--------------------------------")

        for local_idx, response in enumerate(batch_results):
            if response is None:
                response = "[ERROR: Empty response after inference]"
            results.append(response)

            try:
                if final_return_list is not None:
                    index_in_prompts = i + local_idx
                    middle_results.append({
                        "prompt": original_batch_prompts[local_idx],
                        "response": response,
                        "final_return_list": final_return_list[index_in_prompts]
                    })
                else:
                    middle_results.append({
                        "prompt": original_batch_prompts[local_idx],
                        "response": response
                    })
            except:
                print(f"Error constructing intermediate results.")
    
        if cfg.experiment.saving_frequency > 0 and (i // cfg.inference.batch_size + 1) % cfg.experiment.saving_frequency == 0:
            # Save intermediate results
            if os.path.exists(cfg.experiment.output_dir) is False:
                os.makedirs(cfg.experiment.output_dir)
            try:
                pkl.dump(middle_results, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
            except:
                print(f"Error saving intermediate results.")
                continue

    return results

def transformers_run_inference(model, tokenizer, cfg, prompts):
    results = []

    for i in tqdm(range(0, len(prompts), cfg.inference.batch_size)):
        batch_prompts = prompts[i:i + cfg.inference.batch_size]
        
        # Process each prompt separately to avoid merging
        batch_outputs = []
        for prompt in batch_prompts:
            message = {"role": "user", "content": prompt}
            inputs = tokenizer.apply_chat_template(
                [message],  # Single message in a list
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            
            output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask if hasattr(inputs, 'attention_mask') else None,
                max_new_tokens=cfg.model.max_tokens,
                temperature=cfg.model.temperature,
                top_p=cfg.model.top_p,
                top_k=cfg.model.top_k,
                do_sample=bool(cfg.model.temperature and cfg.model.temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=_cfg_get(cfg, "model.repetition_penalty", None),
                no_repeat_ngram_size=_cfg_get(cfg, "model.no_repeat_ngram_size", None),
            )


            decoded_output = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            batch_outputs.append(decoded_output)
            if cfg.experiment.debug_print:
                print("Model Output:")
                print(decoded_output)

            # Clear memory for next prompt
            torch.cuda.empty_cache()
        
        results.extend(batch_outputs)

        torch.cuda.empty_cache()

    return results



def gptoss_model_response_parser(response):
    # Handle the analysis assistantfinal format
    reasoning = response.split("assistantfinal")[0]
    output = response.split("assistantfinal")[-1]
    return output


def initialise_openrouter(cfg):
    """
    Initialize OpenRouter API client.
    Returns None for model (API-based) and None for tokenizer.
    API key is read from OPENROUTER_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    # Return a dict with config info instead of model/tokenizer
    client_info = {
        "api_key": api_key,
        "base_url": _cfg_get(cfg, "model.base_url", "https://openrouter.ai/api/v1"),
        "model_name": _cfg_get(cfg, "model.name", "openai/gpt-5.2"),
    }
    return client_info, None


def _openrouter_single_request(client_info, cfg, prompt, max_retries=3, retry_delay=1.0):
    """
    Make a single request to OpenRouter API with retry logic.
    """
    headers = {
        "Authorization": f"Bearer {client_info['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": _cfg_get(cfg, "model.http_referer", "https://github.com"),
        "X-Title": _cfg_get(cfg, "model.x_title", "URS Scoring"),
    }
    
    payload = {
        "model": client_info["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": _cfg_get(cfg, "model.temperature", 0.7),
        "top_p": _cfg_get(cfg, "model.top_p", 0.9),
        "max_tokens": _cfg_get(cfg, "model.max_tokens", 4096),
    }
    
    # Add optional parameters if specified
    if _cfg_get(cfg, "model.frequency_penalty") is not None:
        payload["frequency_penalty"] = cfg.model.frequency_penalty
    if _cfg_get(cfg, "model.presence_penalty") is not None:
        payload["presence_penalty"] = cfg.model.presence_penalty
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{client_info['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=_cfg_get(cfg, "model.timeout", 120)
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limit
                wait_time = retry_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            elif response.status_code >= 500:  # Server error
                wait_time = retry_delay * (2 ** attempt)
                print(f"Server error {response.status_code}. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error: {e}")
                raise e
        except requests.exceptions.Timeout:
            wait_time = retry_delay * (2 ** attempt)
            print(f"Request timeout. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(retry_delay)
    
    return "[ERROR: Max retries exceeded]"


def openrouter_run_inference(model, tokenizer, cfg, prompts, final_return_list=None, self_incremented_seed=False):
    """
    Run inference using OpenRouter API.
    
    Args:
        model: Client info dict from initialise_openrouter
        tokenizer: Not used (None)
        cfg: Hydra config
        prompts: List of prompts to process
        final_return_list: Optional list to track results
        self_incremented_seed: Not used for API-based inference
    """
    client_info = model  # model is actually client_info from initialise_openrouter
    
    print("Running OpenRouter API inference...")
    print(f"Model: {client_info['model_name']}")
    print(f"Number of prompts: {len(prompts)}")
    
    results = []
    middle_results = []
    
    # Get concurrency settings
    max_workers = _cfg_get(cfg, "inference.max_workers", 8)
    batch_size = _cfg_get(cfg, "inference.batch_size", 32)
    
    # Process in batches with concurrent requests
    for batch_start in tqdm(range(0, len(prompts), batch_size)):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        batch_results = [None] * len(batch_prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _openrouter_single_request,
                    client_info,
                    cfg,
                    prompt
                ): idx for idx, prompt in enumerate(batch_prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    response = future.result()
                    batch_results[idx] = response
                    # Real-time debug print as responses arrive
                    if _cfg_get(cfg, "experiment.debug_print", False):
                        print(f"[DEBUG] Received response for index {batch_start + idx} (len={len(response)})")
                except Exception as e:
                    print(f"Error processing prompt at index {batch_start + idx}: {e}")
                    batch_results[idx] = f"[ERROR: {str(e)}]"
        
        # Add batch results to overall results
        for j, response in enumerate(batch_results):
            results.append(response)
            
            try:
                if final_return_list is not None:
                    index_in_prompts = batch_start + j
                    middle_results.append({
                        "prompt": batch_prompts[j],
                        "response": response,
                        "final_return_list": final_return_list[index_in_prompts]
                    })
                else:
                    middle_results.append({
                        "prompt": batch_prompts[j],
                        "response": response
                    })
            except:
                print(f"Error constructing intermediate results.")
        
        if _cfg_get(cfg, "experiment.debug_print", False):
            for j, prompt in enumerate(batch_prompts):
                print(f"\n{'='*60}")
                print(f"[DEBUG] Index: {batch_start + j}")
                print(f"{'='*60}")
                print("Prompt:")
                print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                print(f"{'─'*60}")
                print("OpenRouter Response:")
                print(batch_results[j])
                print(f"{'='*60}\n")
        
        # Save intermediate results
        if cfg.experiment.saving_frequency > 0 and (batch_start // batch_size + 1) % cfg.experiment.saving_frequency == 0:
            if os.path.exists(cfg.experiment.output_dir) is False:
                os.makedirs(cfg.experiment.output_dir)
            try:
                pkl.dump(middle_results, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
            except:
                print(f"Error saving intermediate results.")
    
    return results



