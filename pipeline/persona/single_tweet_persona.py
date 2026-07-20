import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
from datasets import load_dataset
from datasets import Dataset
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import logging
from datetime import datetime
import gc



def load_twitter_data_dask(cfg: DictConfig, logger=None) -> dd.DataFrame:
    path = cfg.task.dataset.path
    final_path = path + "*.parquet"
    if cfg.experiment.debug_print and logger:
        logger.info("Loading dataset from " + final_path)
        logger.info("-"*100)
    ddf = dd.read_parquet(final_path)
    return ddf

def load_bluesky_data_dask(cfg: DictConfig, logger=None) -> dd.DataFrame:
    path = cfg.task.dataset.path
    ds = load_dataset(path)
    ds = ds["train"].to_pandas()
    ddf = dd.from_pandas(ds, npartitions=10)
    ddf["user"] = ddf["author"]
    ddf["tweet"] = ddf["text"]
    if cfg.experiment.debug_print and logger:
        logger.info("Loading dataset from " + path)
        logger.info("-"*100)
    return ddf

def check_tweet_useful_prompt(cfg: DictConfig, user_name: str, tweet_text: str) -> str:
    promt_template = cfg.task.prompt.base
    prompt = promt_template.replace("TEMPLATE_USER_NAME", user_name).replace("TEMPLATE_TWEET", tweet_text)
    return prompt

def single_tweet_persona(cfg: DictConfig):
    setup_logging = hydra.utils.get_method(cfg.experiment.setup_logging)
    logger = setup_logging(cfg)
    logger.info(f"Synthesizing persona data for {cfg.task.name}")

    dataset_entry_point = hydra.utils.get_method(cfg.task.dataset.entry_point)
    dataset = dataset_entry_point(cfg, logger)

    if cfg.experiment.debug_print:
        logger.info("Loading dataset")
        logger.info(f"Dataset head: {dataset.head()}")
        logger.info("-"*100)
    with ProgressBar():
        user_list = dataset["user"].unique().compute()
    logger.info(f"Number of users: {len(user_list)}")

    with ProgressBar():
        tweet_countes = dataset["user"].value_counts().compute()
        user_list = tweet_countes[tweet_countes > cfg.task.min_tweet_number]
    persona_counter = 0
    all_prompts = []
    if cfg.task.starting_index == -1:
        logger.info("Sampling users from the dataset")
        sample_user_list = user_list.sample(cfg.task.max_persona_number)
    else:
        logger.info(f"Starting from givenindex {cfg.task.starting_index} and sampling {cfg.task.max_persona_number} users")
        shuffle_user_list = user_list.sample(len(user_list))
        sample_user_list = shuffle_user_list[cfg.task.starting_index:cfg.task.starting_index + cfg.task.max_persona_number]
    for user, index in sample_user_list.items():
            
        logger.info(f"Processing user {index+1} of {len(user_list)}")

        with ProgressBar():
            user_tweets = dataset[dataset["user"] == user].compute()
        tweet_number = user_tweets.shape[0]
        logger.info(f"Number of tweets for user {user}: {tweet_number}")
        if tweet_number < cfg.task.min_tweet_number:
            logger.info(f"User {user} has less than {cfg.task.min_tweet_number} tweets, skipping")
            continue
        
        tweet_counter = 0
        for index, row in user_tweets.iterrows():
            tweet_text = row["tweet"]
            prompt = check_tweet_useful_prompt(cfg, user, tweet_text)
            all_prompts.append({"User": user, "Tweet": tweet_text, "Prompt": prompt})
            tweet_counter += 1
            if tweet_counter >= cfg.task.max_tweet_number:
                logger.info(f"Reached the maximum number of tweets to use for user {user}: {cfg.task.max_tweet_number}")
                break
            if cfg.experiment.debug_print:
                if index == 0:
                    logger.info(f"Prompt for user {user} and tweet {tweet_text}")
                    logger.info(f"User: {user}, Tweet: {tweet_text}")
                    logger.info(f"Prompt: {prompt}")
                    logger.info("-"*100)
        persona_counter += 1
        if persona_counter >= cfg.task.max_persona_number:
            logger.info(f"Reached the maximum number of personas to construct: {cfg.task.max_persona_number}")
            break
        
        del user_tweets
        gc.collect()
    
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
    logger.info("Inference Results Dumped to " + cfg.experiment.output_dir + "/inference_results.csv")
    pkl.dump(final_result_list, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
    pd.DataFrame(final_result_list).to_csv(cfg.experiment.output_dir + "/inference_results.csv", index=False)

    return final_result_list


