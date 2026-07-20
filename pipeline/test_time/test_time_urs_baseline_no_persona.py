import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl


def _resolve_sampling_temperature(cfg: DictConfig) -> float:
    sampling_temperature = cfg.task.get("sampling_temperature", None)
    if sampling_temperature is not None:
        cfg.model.temperature = float(sampling_temperature)
    return float(cfg.model.temperature)


def test_time_urs_baseline_no_persona(cfg: DictConfig):
    print(f"Testing URS for {cfg.task.name}")
    effective_temperature = _resolve_sampling_temperature(cfg)
    print(f"Sampling temperature: {effective_temperature}")
    df = pd.read_csv(cfg.task.dataset.path)
    prompt_list = []
    final_result_list = []

    for sample_idx in range(cfg.task.number_of_answers_per_query):
        for _, row_df in df.iterrows():
            intent = row_df["user_intent"]
            question = row_df["question"]
            reference_answer = row_df["reference_ans"]
            prompt = cfg.task.prompt.base.replace("TEMPLATE_QUESTION", question)
            prompt_list.append(prompt)
            final_result_list.append({
                "question": question,
                "prompt": prompt,
                "reference_answer": reference_answer,
                "intent": intent,
                "sample_idx": sample_idx,
                "sampling_temperature": effective_temperature,
            })
    
    print(f"Total number of prompts: {len(prompt_list)}")
    print(f"First 5 prompts:")
    for prompt in prompt_list[:5]:
        print(prompt)
        print("-"*100)

    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, prompt_list, final_result_list, self_incremented_seed=True)
    for i, result in enumerate(results):
        final_result_list[i]["Response"] = result
        try:
            response_parser = hydra.utils.get_method(cfg.inference.response_parser)
            final_result_list[i]["Response"] = response_parser(result)
        except:
            pass
    if os.path.exists(cfg.experiment.output_dir) is False:
        os.makedirs(cfg.experiment.output_dir)
    print("Inference Results Dumped to", cfg.experiment.output_dir + "/inference_results.csv")
    pkl.dump(final_result_list, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
    pd.DataFrame(final_result_list).to_csv(cfg.experiment.output_dir + "/inference_results.csv", index=False)
