
import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np




def baseline_fake_userprofile(cfg: DictConfig):
    print(f"Generating fake user profile for {cfg.task.name}")
    final_return_list = []
    prompt_list = []

    for index in range(cfg.task.maximum_number):
        prompt = cfg.task.prompt.base
        prompt_list.append(prompt)
        final_return_list.append({
            "prompt": prompt,
        })

    print(f"Total number of prompts: {len(prompt_list)}")
    print("First 5 prompts:")
    for i, prompt in enumerate(prompt_list[:5]):
        print(prompt)
        print("--------------------------------")

    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, prompt_list, self_incremented_seed=True)

    for i, result in enumerate(results):
        final_return_list[i]["response"] = result

    if os.path.exists(cfg.experiment.output_dir) is False:
        os.makedirs(cfg.experiment.output_dir)
    print("Inference Results Dumped to", cfg.experiment.output_dir + "/inference_results.csv")
    pkl.dump(final_return_list, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
    pd.DataFrame(final_return_list).to_csv(cfg.experiment.output_dir + "/inference_results.csv", index=False)

    return final_return_list
