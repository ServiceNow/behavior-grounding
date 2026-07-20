import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np






def urs(cfg: DictConfig):

    print(f"Evaluating URS for {cfg.task.name}")


    df = pd.read_csv(cfg.task.dataset.path)
    print(f"Total number of data points: {df.shape[0]}")

    df = df[df["user_intent"] == cfg.task.dataset.user_intention]
    df = df[df["language"] == cfg.task.dataset.language]

    print(f"Total number of data points after filtering: {df.shape[0]}")

    sample_number = min(cfg.task.maximum_question_number, df.shape[0])
    down_sampled_df = df.sample(n=sample_number)

    print(f"Total number of data points after downsampling: {down_sampled_df.shape[0]}")
    
    if cfg.task.prompt.use_user_profile:
        persona_df = pd.read_csv(cfg.task.dataset.user_profile_path)
        down_sampled_persona_df = persona_df.sample(n=cfg.task.maximum_persona_number)
    else:
        down_sampled_persona_df = None

    final_result_list = []
    prompt_list = []
    for index, row in down_sampled_df.iterrows():
        question = row["question"]
        reference_answer = row["reference_ans"]
        if down_sampled_persona_df is not None:
            for index_persona, row_persona in down_sampled_persona_df.iterrows():
                user_name = row_persona["UserName"]
                user_profile = row_persona["UserProfile"]
                prompt = cfg.task.prompt.base.replace("TEMPLATE_USER_NAME", user_name).replace("TEMPLATE_USER_PROFILE", user_profile).replace("TEMPLATE_QUESTION", question)
                prompt_list.append(prompt)
                final_result_list.append({
                    "user_name": user_name,
                    "user_profile": user_profile,
                    "question": question,
                    "prompt": prompt,
                    "reference_answer": reference_answer
                })

                if cfg.experiment.debug_print:
                    print(prompt)
                    print("--------------------------------")
        else:
            prompt = cfg.task.prompt.base.replace("TEMPLATE_QUESTION", question)
            prompt_list.append(prompt)
            final_result_list.append({
                "question": question,
                "prompt": prompt,
                "reference_answer": reference_answer
            })
            if cfg.experiment.debug_print:
                print(prompt)
                print("--------------------------------")


    print(f"Total number of prompts: {len(prompt_list)}")


    print(f"Inferring {len(prompt_list)} prompts")
    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, prompt_list)

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

