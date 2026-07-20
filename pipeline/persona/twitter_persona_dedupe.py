import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl




def load_user_descriptions(cfg: DictConfig):
    path = cfg.task.dataset.path
    df = pd.read_csv(path)
    return df


def summary_user_descriptions_prompt(cfg: DictConfig, user_id: str, user_description_list: str):
    prompt_template = cfg.task.prompt.base
    prompt = prompt_template.replace("TEMPLATE_USER_ID", user_id).replace("TEMPLATE_USER_DESCRIPTION_LIST", user_description_list)
    return prompt

def dedupe_social_media_twitter(cfg: DictConfig):
    df = load_user_descriptions(cfg)
    all_prompts = []
    final_result_list = []
    for index, row in df.iterrows():
        user_id = row["User"]
        user_description_list = row["UserInformation"]
        prompt = summary_user_descriptions_prompt(cfg, user_id, user_description_list)
        all_prompts.append(prompt)
        final_result_list.append(
            {
                "User": user_id,
                "UserInformation": user_description_list,
                "Prompt": prompt
            }
        )
    if cfg.experiment.debug_print:
        print("First 5 prompts:")
        print(all_prompts[:5])
        print("-"*100)
    print("Number of prompts:", len(all_prompts))
    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, all_prompts, final_result_list)

    for i, result in enumerate(results):
        append_dict = {"User": final_result_list[i]["User"], "UserInformation": final_result_list[i]["UserInformation"], "Prompt": final_result_list[i]["Prompt"], "Response": result}
        try:
            response_parser = hydra.utils.get_method(cfg.inference.response_parser)
            append_dict["Response"] = response_parser(result)
            final_result_list.append(append_dict)
        except:
            final_result_list.append(append_dict)
        
    if os.path.exists(cfg.experiment.output_dir) is False:
        os.makedirs(cfg.experiment.output_dir)
    print("Inference Results Dumped to", cfg.experiment.output_dir + "/inference_results.csv")
    pkl.dump(final_result_list, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
    pd.DataFrame(final_result_list).to_csv(cfg.experiment.output_dir + "/inference_results.csv", index=False)

    return final_result_list

    
