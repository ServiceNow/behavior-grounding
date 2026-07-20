import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np



def generate_qa_prompt(cfg: DictConfig, user_name: str, user_profile: str, question: str):
    prompt_template = cfg.task.prompt.base
    prompt = prompt_template.replace("TEMPLATE_USER_NAME", user_name).replace("TEMPLATE_USER_PROFILE", user_profile).replace("TEMPLATE_QUESTION", question)
    return prompt





def twitter_user_profile_qa(cfg: DictConfig):
    print(f"Evaluating twitter user profile qa for {cfg.task.name}")


    dataset = pd.read_csv(cfg.task.dataset.path)
    print(f"Number of questions: {len(dataset)} in the dataset")

    dataset = dataset.sample(min(len(dataset), cfg.task.maximum_number_of_questions))
    print(f"Number of questions: {len(dataset)} after sampling")

    all_prompts = []        
    final_results = []
    for index, row in dataset.iterrows():
        user_name = row["User Name"]
        user_profile = row["User Profile"]
        question = row["Question"]
        answer = row["Answer"]
        prompt = generate_qa_prompt(cfg, user_name, user_profile, question)
        all_prompts.append(prompt)
        final_results.append({
            "User Name": user_name,
            "User Profile": user_profile,
            "Question": question,
            "Answer": answer
        })
        if cfg.experiment.debug_print:
            print(prompt)
            print("--------------------------------")

    print(f"Number of prompts: {len(all_prompts)}")
    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, all_prompts)

    for i, result in enumerate(results):
        final_results[i]["Response"] = result
        try:
            response_parser = hydra.utils.get_method(cfg.inference.response_parser)
            final_results[i]["Response"] = response_parser(result)
        except:
            pass
    if os.path.exists(cfg.experiment.output_dir) is False:
        os.makedirs(cfg.experiment.output_dir)
    print("Inference Results Dumped to", cfg.experiment.output_dir + "/inference_results.csv")
    pkl.dump(final_results, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
    pd.DataFrame(final_results).to_csv(cfg.experiment.output_dir + "/inference_results.csv", index=False)


    
    