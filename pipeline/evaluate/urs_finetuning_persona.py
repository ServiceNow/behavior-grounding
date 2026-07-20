import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np






def urs_finetuning_persona(cfg: DictConfig):

    print(f"Evaluating URS for {cfg.task.name}")


    df = pd.read_csv(cfg.task.dataset.path)
    print(f"Total number of data points: {df.shape[0]}")

    if cfg.task.dataset.down_sample_size > 0:
        down_sampled_df = df.sample(n=cfg.task.dataset.down_sample_size)
    else:
        down_sampled_df = df
    final_result_list = []
    prompt_list = []
    for index, row in down_sampled_df.iterrows():
        question = row["question"]
        intent = row["intent"]
        reference_answer = row["reference_answer"]
        user_profile = row["user_profile"]
        prompt = row["system_prompt"] + "\n" + row["question"]
        prompt_list.append(prompt)
        final_result_list.append({
            "question": question,
            "reference_answer": reference_answer,
            "prompt": prompt,
            "intent": intent,
            "user_profile": user_profile,
        })



    print(f"Total number of prompts: {len(prompt_list)}")
    print(f"First 5 prompts:")
    for prompt in prompt_list[:5]:
        print(prompt)
        print("-"*100)


    print(f"Inferring {len(prompt_list)} prompts")
    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, prompt_list, final_result_list)

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

