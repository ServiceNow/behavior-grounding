import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np
import random

def test_time_urs_related_persona(cfg: DictConfig):
    print(f"Testing URS for {cfg.task.name}")

    annotation_df = pd.read_csv(cfg.task.dataset.related_persona_annotation_path)

    
    df = pd.read_csv(cfg.task.dataset.path)
    prompt_list = []
    final_result_list = []

    for index_df, row_df in df.iterrows():


        intent = row_df["user_intent"]
        question = row_df["question"]
        reference_answer = row_df["reference_ans"]

        selected_annotation = annotation_df[annotation_df["question"] == question]
        list_of_related_persona = selected_annotation["user_profile"].unique().tolist()
        number_of_related_persona = len(list_of_related_persona)
        if number_of_related_persona > cfg.task.number_of_answers_per_query:
            sampled_list_of_related_persona = random.sample(list_of_related_persona, cfg.task.number_of_answers_per_query)
            for user_profile in list_of_related_persona:
                prompt = cfg.task.prompt.base.replace("TEMPLATE_USER_PROFILE", user_profile).replace("TEMPLATE_QUESTION", question)
                prompt_list.append(prompt)
                final_result_list.append({
                    "user_profile": user_profile,
                    "question": question,
                    "prompt": prompt,
                    "reference_answer": reference_answer,
                    "intent": intent,
                })
        else:
            print(f"Not enough related persona found for question: {question}")
            print(f"Number of related persona: {number_of_related_persona}")
        print("Current number of prompts:", len(prompt_list))

    
    print(f"Total number of prompts: {len(prompt_list)}")
    print(f"First 5 prompts:")
    for prompt in prompt_list[:5]:
        print(prompt)
        print("-"*100)

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