import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
from datasets import load_dataset
from datasets import Dataset
import numpy as np


def load_persona_data(cfg: DictConfig):
    persona_path = cfg.task.dataset.persona_path
    persona_data = pd.read_csv(persona_path)

    # Support both legacy (UserName/UserProfile) and dedupe (user_id/user_profile) schemas.
    if "UserName" not in persona_data.columns and "user_id" in persona_data.columns:
        persona_data = persona_data.rename(columns={"user_id": "UserName"})
    if "UserProfile" not in persona_data.columns and "user_profile" in persona_data.columns:
        persona_data = persona_data.rename(columns={"user_profile": "UserProfile"})

    return persona_data

def load_labelled_userful_information_or_not_user_list(cfg: DictConfig):
    labelled_userful_information_or_not_path = cfg.task.dataset.twitter_path 
    file_list = [file for file in os.listdir(labelled_userful_information_or_not_path) if file.endswith(".csv")]
    user_name_list = [file.replace(".csv", "").replace("inference_results_user_", "") for file in file_list]
    return user_name_list



def load_labelled_userful_information_or_not(cfg: DictConfig, user_name: str):
    labelled_userful_information_or_not_path = cfg.task.dataset.twitter_path + f"/inference_results_user_{user_name}.csv"
    labelled_userful_information_or_not_df = pd.read_csv(labelled_userful_information_or_not_path)
    def process_response(response):
        try:
            tmp = response.split("assistantfinal")[-1]
            if  "True" in tmp:
                tmp = tmp.replace("True", "true")
            if "False" in tmp:
                tmp = tmp.replace("False", "false")
            tmp = json.loads(tmp)
        except Exception as e:
            tmp = None
        return tmp

    labelled_userful_information_or_not_df["Response"] = labelled_userful_information_or_not_df["Response"].apply(process_response)
    labelled_userful_information_or_not_df["UsefulInformation"] = labelled_userful_information_or_not_df["Response"].apply(lambda x: x["UsefulInformation"] if x and "UsefulInformation" in x else None)
    labelled_userful_information_or_not_df["Justificantion"] = labelled_userful_information_or_not_df["Response"].apply(lambda x: x["Justificantion"] if x and "Justificantion" in x else None)
    labelled_userful_information_or_not_df["UserInformation"] = labelled_userful_information_or_not_df["Response"].apply(lambda x: x["UserInformation"] if x and "UserInformation" in x else None)

    return labelled_userful_information_or_not_df
    

def generate_prompt(cfg: DictConfig, user_name: str, user_profile: str, tweets: list):
    tweet_string = "\n".join([f"Tweet {i+1}: {tweet}" for i, tweet in enumerate(tweets)])
    prompt = cfg.task.prompt.base.replace("TEMPLATE_USER_NAME", user_name).replace("TEMPLATE_USER_PROFILE", user_profile).replace("TEMPLATE_TWEETS", tweet_string)
    return prompt




def twitter_persona_predict(cfg: DictConfig):
    print(f"Evaluating twitter persona prediction for {cfg.task.name}")

    dataset_loader = hydra.utils.get_method(cfg.task.dataset.loader)
    twitter_dataset = dataset_loader(cfg)
    all_users = twitter_dataset.unique("user")
    persona_data = load_persona_data(cfg)

    all_prompts = []
    final_results = []

    total_tweet_number = cfg.task.max_tweet_per_persona
    persona_count = 0
    for index, row in persona_data.iterrows():
        user_name = row["UserName"]
        user_profile = row["UserProfile"]

        
        tweet_from_user_themself = twitter_dataset.filter(lambda x: x["user"] == user_name, num_proc=os.cpu_count())
        rest_tweet_number = total_tweet_number
        if len(tweet_from_user_themself) > cfg.task.exclude_tweet_number:
            sampled_number = min(int(total_tweet_number / 2), len(tweet_from_user_themself) - cfg.task.exclude_tweet_number)
            tweet_from_user_themself_excluded = tweet_from_user_themself.select(range(cfg.task.exclude_tweet_number, len(tweet_from_user_themself)))
            sampled_index = np.random.permutation(len(tweet_from_user_themself_excluded))[:sampled_number]
            tweet_from_user_themself_sampled = tweet_from_user_themself_excluded.select(sampled_index)

            for tweet_batch in range(0, len(tweet_from_user_themself_sampled), cfg.task.tweet_per_query):
                tweets = tweet_from_user_themself_sampled.select(range(tweet_batch, tweet_batch + cfg.task.tweet_per_query))
                tweet_list = [i["tweet"] for i in tweets]
                prompt = generate_prompt(cfg, user_name, user_profile, tweet_list)
                ground_truth = True
                all_prompts.append(prompt)
                final_results.append({
                    "User Name": user_name,
                    "User Profile": user_profile,
                    "Tweet": tweet_list,
                    "ActualPoster": user_name,
                    "Ground Truth": ground_truth
                })
                if cfg.experiment.debug_print:
                    print(prompt)
            rest_tweet_number -= len(tweet_from_user_themself_sampled)
        

        all_other_users_list = [u for u in all_users if u != user_name]
        rest_sample_user_number = int(rest_tweet_number / cfg.task.tweet_per_query)
        np.random.shuffle(all_other_users_list)
        sampled_users = all_other_users_list[:rest_sample_user_number]
        print(f"sampled_users: {sampled_users}")
        for user in sampled_users:
            tweet_from_other_users_sampled = twitter_dataset.filter(lambda x: x["user"] == user, num_proc=os.cpu_count())
            len_tweet_from_other_users_sampled = len(tweet_from_other_users_sampled)
            sampled_indexes_for_other_user = np.random.permutation(len_tweet_from_other_users_sampled)[:cfg.task.tweet_per_query]
            other_user_tweet_sampled = tweet_from_other_users_sampled.select(sampled_indexes_for_other_user)
            tweet_list = [i["tweet"] for i in other_user_tweet_sampled]
            prompt = generate_prompt(cfg, user_name, user_profile, tweet_list)
            ground_truth = False
            all_prompts.append(prompt)
            final_results.append({
                "User Name": user_name,
                "User Profile": user_profile,
                "Tweet": tweet_list,
                "ActualPoster": user,
                "Ground Truth": ground_truth
            })

            if cfg.experiment.debug_print:
                print(prompt)


        persona_count += 1
        print(f"Processed {persona_count} personas")
        if persona_count >= cfg.task.max_persona_number:
            break

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

    return final_results




def twitter_persona_predict_single_tweet(cfg: DictConfig):
    print(f"Evaluating twitter persona prediction for {cfg.task.name}")

    dataset_loader = hydra.utils.get_method(cfg.task.dataset.loader)
    twitter_dataset = dataset_loader(cfg)
    persona_data = load_persona_data(cfg)

    all_prompts = []
    final_results = []

    total_tweet_number = cfg.task.max_tweet_per_persona
    persona_count = 0
    for index, row in persona_data.iterrows():
        user_name = row["UserName"]
        user_profile = row["UserProfile"]

        
        tweet_from_user_themself = twitter_dataset.filter(lambda x: x["user"] == user_name, num_proc=os.cpu_count())
        rest_tweet_number = total_tweet_number
        if len(tweet_from_user_themself) > cfg.task.exclude_tweet_number:
            sampled_number = min(int(total_tweet_number / 2), len(tweet_from_user_themself) - cfg.task.exclude_tweet_number)
            tweet_from_user_themself_excluded = tweet_from_user_themself.select(range(cfg.task.exclude_tweet_number, len(tweet_from_user_themself)))
            sampled_index = np.random.permutation(len(tweet_from_user_themself_excluded))[:sampled_number]
            tweet_from_user_themself_sampled = tweet_from_user_themself_excluded.select(sampled_index)

            for tweet_batch in range(0, len(tweet_from_user_themself_sampled), cfg.task.tweet_per_query):
                tweets = tweet_from_user_themself_sampled.select(range(tweet_batch, tweet_batch + cfg.task.tweet_per_query))
                tweet_list = [i["tweet"] for i in tweets]
                prompt = generate_prompt(cfg, user_name, user_profile, tweet_list)
                ground_truth = True
                all_prompts.append(prompt)
                final_results.append({
                    "User Name": user_name,
                    "User Profile": user_profile,
                    "Tweet": tweet_list,
                    "ActualPoster": user_name,
                    "Ground Truth": ground_truth
                })
                if cfg.experiment.debug_print:
                    print(prompt)
            rest_tweet_number -= len(tweet_from_user_themself_sampled)
        

        all_other_tweets = twitter_dataset.filter(lambda x: x["user"] != user_name, num_proc=os.cpu_count())
        sampled_indexes_for_other_tweets = np.random.permutation(len(all_other_tweets))[:rest_tweet_number]
        other_user_tweet_sampled = all_other_tweets.select(sampled_indexes_for_other_tweets)
        for tweet in other_user_tweet_sampled:
            tweet_list = [tweet["tweet"]]
            prompt = generate_prompt(cfg, user_name, user_profile, tweet_list)
            ground_truth = False
            all_prompts.append(prompt)
            final_results.append({
                "User Name": user_name, 
                "User Profile": user_profile,
                "Tweet": tweet_list,
                "ActualPoster": tweet["user"],
                "Ground Truth": ground_truth
            })
            if cfg.experiment.debug_print:
                print(prompt)


        persona_count += 1
        print(f"Processed {persona_count} personas")
        if persona_count >= cfg.task.max_persona_number:
            break

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

    return final_results



def twitter_persona_predict_single_tweet_with_labelled_userful_information_or_not(cfg: DictConfig):
    print(f"Evaluating twitter persona prediction for {cfg.task.name}")

    persona_data = load_persona_data(cfg)
    labelled_userful_information_or_not_user_list = load_labelled_userful_information_or_not_user_list(cfg)
    all_prompts = []
    final_results = []

    total_tweet_number = cfg.task.max_tweet_per_persona
    persona_count = 0
    for index, row in persona_data.iterrows():
        user_name = row["UserName"]
        user_profile = row["UserProfile"]


        if user_name in labelled_userful_information_or_not_user_list:
            tweet_from_user_themself = load_labelled_userful_information_or_not(cfg, user_name)
            rest_tweet_number = total_tweet_number
            print(f"tweet_from_user_themself: {len(tweet_from_user_themself)}")
            if len(tweet_from_user_themself) > cfg.task.exclude_tweet_number:
                tweet_from_user_themself_excluded = tweet_from_user_themself[cfg.task.exclude_tweet_number:len(tweet_from_user_themself)]
                tweet_from_user_themself_excluded = tweet_from_user_themself_excluded[tweet_from_user_themself_excluded["UsefulInformation"] == True]

                sampled_number = min(int(total_tweet_number * cfg.task.positive_example_ratio), len(tweet_from_user_themself_excluded))
                
                tweet_from_user_themself_sampled = tweet_from_user_themself_excluded.sample(sampled_number)
                print(f"tweet_from_user_themself_sampled: {len(tweet_from_user_themself_sampled)}")
                for tweet_batch in range(0, len(tweet_from_user_themself_sampled), cfg.task.tweet_per_query):
                    tweets = tweet_from_user_themself_sampled[tweet_batch:tweet_batch + cfg.task.tweet_per_query]
                    tweet_list = [i[1]["Tweet"] for i in tweets.iterrows()]
                    try:
                        prompt = generate_prompt(cfg, user_name, user_profile, tweet_list)
                    except Exception as e:
                        print(e)
                        print(tweet_list)
                        print(user_name)
                        print(user_profile)
                        print("--------------------------------")
                        continue
                    ground_truth = True
                    all_prompts.append(prompt)
                    final_results.append({
                        "User Name": user_name,
                        "User Profile": user_profile,
                        "Tweet": tweet_list,
                        "ActualPoster": user_name,
                        "Ground Truth": ground_truth,
                        "UserInformation": [i[1]["UserInformation"] for i in tweets.iterrows()]
                    })
                    if cfg.experiment.debug_print:
                        print(prompt)
                rest_tweet_number -= len(tweet_from_user_themself_sampled)
            

            all_other_users_list = [u for u in labelled_userful_information_or_not_user_list if u != user_name]
            rest_sample_user_number = int(rest_tweet_number / cfg.task.tweet_per_query)
            np.random.shuffle(all_other_users_list)
            sampled_users = all_other_users_list[:rest_sample_user_number]
            for user in sampled_users:
                tweet_from_other_users_sampled = load_labelled_userful_information_or_not(cfg, user)
                tweet_from_other_users_sampled = tweet_from_other_users_sampled[tweet_from_other_users_sampled["UsefulInformation"] == True]
                other_user_tweet_sampled = tweet_from_other_users_sampled.sample(rest_tweet_number)
                print(f"other_user_tweet_sampled: {len(other_user_tweet_sampled)}")
                for index, row in other_user_tweet_sampled.iterrows():
                    tweet_list = [row["Tweet"]]
                    try:
                        prompt = generate_prompt(cfg, user_name, user_profile, tweet_list)
                    except Exception as e:
                        print(e)
                        print(tweet_list)
                        print(user_name)
                        print(user_profile)
                        print("--------------------------------")
                        continue
                    ground_truth = False
                    all_prompts.append(prompt)
                    final_results.append({
                        "User Name": user_name, 
                        "User Profile": user_profile,
                        "Tweet": tweet_list,
                        "ActualPoster": row["User"],
                        "Ground Truth": ground_truth,
                        "UserInformation": row["UserInformation"]
                    })
                    if cfg.experiment.debug_print:
                        print(prompt)


            persona_count += 1
            print(f"Processed {persona_count} personas")
            if persona_count >= cfg.task.max_persona_number:
                break
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

    return final_results