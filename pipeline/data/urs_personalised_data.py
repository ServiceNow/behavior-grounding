





import json
import os
from typing import Any
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np

def generate_prompt(cfg: DictConfig, user_profile: str, question: str, reference_answer: str):
    prompt = cfg.task.prompt.base.replace("TEMPLATE_USER_PROFILE", user_profile)
    prompt = prompt.replace("TEMPLATE_QUESTION", question)
    prompt = prompt.replace("TEMPLATE_REFERENCE_ANSWER", reference_answer)
    return prompt


def urs_personalised_data(cfg: DictConfig):
    print("Task: URS Personalised Data")
    dataset = pd.read_csv(cfg.task.dataset.path)
    persona_data = pd.read_csv(cfg.task.dataset.persona_path)
    persona_data = persona_data[cfg.task.persona_start_index: min(cfg.task.persona_start_index + cfg.task.maximum_persona_number, len(persona_data))]
    prompt_list = []
    final_result_list = []
    for index, row_persona in persona_data.iterrows():
        user_profile = row_persona["user_profile"]
        counter = 0
        for index, row_dataset in dataset.iterrows():
            question = row_dataset["question"]
            reference_answer = row_dataset["reference_ans"]
            prompt = generate_prompt(cfg, user_profile, question, reference_answer)
            prompt_list.append(prompt)
            final_result_list.append(
                {
                    "user_profile": user_profile, 
                    "question": question,
                    "reference_answer": reference_answer,
                    "prompt": prompt,
                })
            counter += 1
            if counter >= cfg.task.maximum_pairs_per_person:
                break

    print("First 5 prompts:")
    for prompt in prompt_list[:5]:
        print(prompt)
        print("-"*100)
    print("Number of prompts:", len(prompt_list))

    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, prompt_list, final_result_list)

    for i, result in enumerate(results):
        final_result_list[i]["Response"] = result
    if os.path.exists(cfg.experiment.output_dir) is False:
        os.makedirs(cfg.experiment.output_dir)
    print("Inference Results Dumped to " + cfg.experiment.output_dir + "/inference_results.csv")
    pkl.dump(final_result_list, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
    pd.DataFrame(final_result_list).to_csv(cfg.experiment.output_dir + "/inference_results.csv", index=False)
    return final_result_list

