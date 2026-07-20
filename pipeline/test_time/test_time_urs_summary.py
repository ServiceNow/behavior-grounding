import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np


def _normalize_summary_input_answer(response: str, max_answer_length: int | None):
    if not isinstance(response, str):
        return ""
    if "assistantanalysis" in response.lower():
        response = response.split("assistantanalysis")[-1]
    response = response.strip()
    if max_answer_length is not None and max_answer_length > 0 and len(response) > max_answer_length:
        response = response[:max_answer_length] + "..."
    return response


def test_time_urs_summary(cfg: DictConfig):
    print(f"Testing URS Summary for {cfg.task.name}")
    df = pd.read_csv(cfg.task.dataset.path)
    prompt_list = []
    final_result_list = []
    question_list = df["question"].unique().tolist()
    number_of_answers_per_query_list = []
    if isinstance(cfg.task.number_of_answers_per_query, str):
        number_of_answers_per_query_list = [int(x) for x in cfg.task.number_of_answers_per_query.split("-")]
    else:
        number_of_answers_per_query_list.append(cfg.task.number_of_answers_per_query)
    no_cap_k_leq = int(cfg.task.get("no_cap_k_leq", 10))
        
    for question in question_list:
        answers = df[df["question"] == question]["Response"].dropna().tolist()
        intent = df[df["question"] == question]["intent"].unique().tolist()[0]
        reference_answer = df[df["question"] == question]["reference_answer"].unique().tolist()[0]
        for number_of_answers_per_query in number_of_answers_per_query_list:
            sampled_answers = answers[:number_of_answers_per_query]
            max_answer_length = None if number_of_answers_per_query <= no_cap_k_leq else cfg.task.max_answer_length
            normalized_sampled_answers = [
                _normalize_summary_input_answer(response, max_answer_length)
                for response in sampled_answers
            ]

            # Fast path: when only one answer is requested, skip summarization model inference
            # and directly pass through that cleaned answer for downstream scoring.
            if number_of_answers_per_query == 1 and len(normalized_sampled_answers) > 0:
                final_result_list.append({
                    "question": question,
                    "answers": answers,
                    "prompt": "[DIRECT_PASS_THROUGH_SINGLE_ANSWER]",
                    "intent": intent,
                    "reference_answer": reference_answer,
                    "number_of_answers_per_query": number_of_answers_per_query,
                    "Response": normalized_sampled_answers[0],
                })
                continue

            prompt = cfg.task.prompt.base.replace("TEMPLATE_QUESTION", question).replace("TEMPLATE_ANSWERS", "\n".join(normalized_sampled_answers))
            print("Length of prompt:", len(prompt))
            prompt_list.append(prompt)
            final_result_list.append({
                "question": question,
                "answers": answers,
                "prompt": prompt,
                "intent": intent,
                "reference_answer": reference_answer,
                "number_of_answers_per_query": number_of_answers_per_query,
            })
    print(f"Total number of prompts: {len(prompt_list)}")
    print(f"First 5 prompts:")
    for prompt in prompt_list[:1]:
        print(prompt)
        print("-"*100)

    if len(prompt_list) > 0:
        initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
        model, tokenizer = initialise_method(cfg)
        inference_method = hydra.utils.get_method(cfg.inference.inference_method)
        results = inference_method(model, tokenizer, cfg, prompt_list, final_result_list)

        pending_indices = [i for i, r in enumerate(final_result_list) if "Response" not in r]
        for i, result in enumerate(results):
            target_idx = pending_indices[i]
            final_result_list[target_idx]["Response"] = result
            try:
                response_parser = hydra.utils.get_method(cfg.inference.response_parser)
                final_result_list[target_idx]["Response"] = response_parser(result)
            except:
                pass
    if os.path.exists(cfg.experiment.output_dir) is False:
        os.makedirs(cfg.experiment.output_dir)
    print("Inference Results Dumped to", cfg.experiment.output_dir + "/inference_results.csv")
    pkl.dump(final_result_list, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
    pd.DataFrame(final_result_list).to_csv(cfg.experiment.output_dir + "/inference_results.csv", index=False)
