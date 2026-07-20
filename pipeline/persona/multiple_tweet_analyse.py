import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
from datasets import load_dataset
from datasets import Dataset



def check_tweet_useful_prompt_multiple_tweet_analyse(cfg: DictConfig, user: str, tweet_text_list: list) -> str:
    prompt_template = cfg.task.prompt.base
    list_of_tweet = ""
    for i, tweet in enumerate(tweet_text_list):
        list_of_tweet += f"User Tweet {i+1}: {tweet}\n"
    prompt = prompt_template.replace("TEMPLATE_USER_NAME", user).replace("TEMPLATE_USER_TWEETS", list_of_tweet)
    return prompt

def multiple_tweet_analyse(cfg: DictConfig):
    print(f"Synthesizing persona data for {cfg.task.name}")

    dataset_loader = hydra.utils.get_method(cfg.task.dataset.loader)
    dataset = dataset_loader(cfg)

    user_list = dataset.unique("user")
    print(f"Number of users: {len(user_list)}")
    persona_counter = 0
    all_prompts = []
    for index, user in enumerate(user_list):
            
        print(f"Processing user {index+1} of {len(user_list)}")

        user_tweets = dataset.filter(lambda x: x["user"] == user, num_proc=os.cpu_count())
        tweet_number = user_tweets.shape[0]
        print(f"Number of tweets for user {user}: {tweet_number}")
        if tweet_number < cfg.task.min_tweet_number or tweet_number < cfg.task.number_of_tweets_per_batch:
            print(f"User {user} has less than {cfg.task.min_tweet_number} tweets, skipping")
            continue
        
        tweet_counter = 0
        for i in range(0, tweet_number, cfg.task.number_of_tweets_per_batch):
            tweet_text_list = user_tweets[i:i+cfg.task.number_of_tweets_per_batch]["tweet"]
            prompt = check_tweet_useful_prompt_multiple_tweet_analyse(cfg, user, tweet_text_list)
            all_prompts.append({"User": user, "Tweet": tweet_text_list, "Prompt": prompt})
            tweet_counter += cfg.task.number_of_tweets_per_batch
            if tweet_counter >= cfg.task.max_tweet_number:
                print(f"Reached the maximum number of tweets to use for user {user}: {cfg.task.max_tweet_number}")
                break
            if cfg.experiment.debug_print:
                print(f"Prompt for user {user} and tweet {tweet_text_list}", user, tweet_text_list)
                print(prompt)
                print("-"*100)
        persona_counter += 1
        if persona_counter >= cfg.task.max_persona_number:
            print(f"Reached the maximum number of personas to construct: {cfg.task.max_persona_number}")
            break


    
    prompt_list = [prompt["Prompt"] for prompt in all_prompts]
    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, prompt_list)

    final_result_list = []
    for i, result in enumerate(results):
        final_result_list.append({"User": all_prompts[i]["User"], "Tweet": all_prompts[i]["Tweet"], "Prompt": all_prompts[i]["Prompt"], "Response": result})
    
    if os.path.exists(cfg.experiment.output_dir) is False:
        os.makedirs(cfg.experiment.output_dir)
    print("Inference Results Dumped to", cfg.experiment.output_dir + "/inference_results.csv")
    pkl.dump(final_result_list, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
    
    try:
        response_parser = hydra.utils.get_method(cfg.model.response_parser)
        for i in range(len(final_result_list)):
            try:
                json_response = json.loads(final_result_list[i]["Response"])
                final_result_list[i]["UsefulInformation"] = response_parser(json_response["UsefulInformation"])
                final_result_list[i]["Justificantion"] = response_parser(json_response["Justificantion"])
                final_result_list[i]["UserInformation"] = response_parser(json_response["UserInformation"])
            except Exception as e:
                final_result_list[i]["UsefulInformation"] = "N/A"
                final_result_list[i]["Justificantion"] = "N/A"
                final_result_list[i]["UserInformation"] = "N/A"
    except Exception as e:
        print("Parsing response failed, skipping", e)
    
    pd.DataFrame(final_result_list).to_csv(cfg.experiment.output_dir + "/inference_results.csv", index=False)

    return final_result_list
