import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np


# Dataset attribute mapping - defines which column to use for each dataset
DATASET_ATTRIBUTES = {
    'mind': 'title',
    'pens': ['title', 'cat'],  # Multi-attribute: title + category
    'goodreads': 'title',
    'books': 'title',
    'movielens': 'title',
    'microlens': 'title',
    'netflix': ['title', 'year'],  # Multi-attribute: title + year
    'cds': 'title',
    'lastfm': ['track_name', 'artist_name'],  # Multi-attribute: track + artist
    'hm': 'detail_desc',
    'pog': 'title_en',  # Could also use title_cn
    'electronics': 'title',
    'steam': 'title',
    'hotelrec': ['hotel_name', 'hotel_location'],  # Multi-attribute
    'yelp': ['name', 'city'],  # Multi-attribute
}

# Topic names for prompts
DATASET_TOPICS = {
    'mind': 'news articles',
    'pens': 'news articles',
    'goodreads': 'books',
    'books': 'books',
    'movielens': 'movies',
    'microlens': 'movies',
    'netflix': 'movies',
    'cds': 'music albums',
    'lastfm': 'music tracks',
    'hm': 'fashion items',
    'pog': 'fashion products',
    'electronics': 'electronics products',
    'steam': 'video games',
    'hotelrec': 'hotels',
    'yelp': 'restaurants',
}


# Readable attribute names for display
ATTRIBUTE_LABELS = {
    'title': 'Title',
    'cat': 'Category',
    'track_name': 'Track',
    'artist_name': 'Artist',
    'hotel_name': 'Hotel',
    'hotel_location': 'Location',
    'name': 'Name',
    'city': 'City',
    'year': 'Year',
}


def get_readable_label(attr):
    """Get readable label for an attribute"""
    return ATTRIBUTE_LABELS.get(attr, attr.replace('_', ' ').title())


def format_attribute_value(attr, value):
    """Format attribute value for display (e.g., convert year to int)"""
    if attr == 'year' and pd.notna(value):
        try:
            return str(int(float(value)))
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def load_recbench_pool(cfg: DictConfig):
    """
    Load item pool from RecBench dataset
    Returns a Series of item descriptions
    """
    dataset_name = cfg.task.dataset.name
    base_path = cfg.task.dataset.base_path
    items_path = os.path.join(base_path, dataset_name, "items.parquet")
    
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Items file not found: {items_path}")
    
    items = pd.read_parquet(items_path)
    print(f"Loaded {len(items)} items from {dataset_name}")
    
    attr = DATASET_ATTRIBUTES.get(dataset_name, 'title')
    
    # Handle multi-attribute datasets
    if isinstance(attr, list):
        # Combine multiple attributes with readable labels and proper formatting
        def format_row(row):
            parts = []
            for a in attr:
                if pd.notna(row[a]):
                    label = get_readable_label(a)
                    value = format_attribute_value(a, row[a])
                    parts.append(f"{label}: {value}")
            return ', '.join(parts)
        
        item_pool = items.apply(format_row, axis=1)
    else:
        # Single attribute
        item_pool = items[attr]
    
    item_pool = item_pool.dropna()
    
    print(f"Item pool size: {len(item_pool)}")
    if len(item_pool) > 0:
        print(f"Sample item: {item_pool.iloc[0]}")
    
    return item_pool


def get_dataset_topic(cfg: DictConfig):
    """
    Get the topic name for the dataset (for prompt generation)
    """
    # Use custom topic if specified, otherwise use default
    if 'topic' in cfg.task.prompt and cfg.task.prompt.topic:
        return cfg.task.prompt.topic
    
    dataset_name = cfg.task.dataset.name
    return DATASET_TOPICS.get(dataset_name, 'items')


def generate_prompt(cfg: DictConfig, interaction_history: list, candidate_item: str):
    """
    Generate prompt for recommendation task
    """
    topic = get_dataset_topic(cfg)
    prompt = cfg.task.prompt.base.replace("TEMPLATE_TOPIC", topic)
    
    # Format interaction history as (1) (2) (3) ...
    interaction_history_formatted = [f"({i+1}) {item}" for i, item in enumerate(interaction_history)]
    interaction_history_str = "\n".join(interaction_history_formatted)
    
    prompt = prompt.replace("TEMPLATE_USER_INTERACTIONS", interaction_history_str)
    prompt = prompt.replace("TEMPLATE_CANDIDATE_NEWS_ARTICLE", candidate_item)
    prompt = prompt.replace("TEMPLATE_CANDIDATE_ITEM", candidate_item)
    
    return prompt


def recommendation_recbench_baseline(cfg: DictConfig):
    """
    Generate recommendation baseline for any RecBench dataset
    """
    print(f"Generating recommendation baseline for {cfg.task.dataset.name}")
    
    dataset_load_method = hydra.utils.get_method(cfg.task.dataset.load_method)
    item_pool = dataset_load_method(cfg)
    
    if len(item_pool) == 0:
        raise ValueError("Item pool is empty!")
    
    final_return_list = []
    prompt_list = []
    
    # Generate synthetic interactions
    for index in range(cfg.task.maximum_number):
        # Random history size
        if cfg.task.interaction_size_lower_bound == cfg.task.interaction_size_upper_bound:
            interaction_history_size = cfg.task.interaction_size_lower_bound
        else:
            interaction_history_size = np.random.randint(
                cfg.task.interaction_size_lower_bound, 
                cfg.task.interaction_size_upper_bound
            )
        
        # Ensure we have enough items to sample
        total_needed = interaction_history_size + 1
        if total_needed > len(item_pool):
            interaction_history_size = len(item_pool) - 1
            total_needed = len(item_pool)
        
        # Sample items (history + candidate)
        sampled_items = item_pool.sample(total_needed, replace=False)
        interaction_history = sampled_items.iloc[:interaction_history_size].tolist()
        candidate_item = sampled_items.iloc[-1]
        
        # Verify candidate is not in history
        if candidate_item in interaction_history:
            print(f"Candidate item {candidate_item} is in interaction history, skipping...")
            continue

        
        prompt = generate_prompt(cfg, interaction_history, candidate_item)
        prompt_list.append(prompt)
        
        final_return_list.append({
            "prompt": prompt,
            "interaction_history": interaction_history,
            "candidate_item": candidate_item,
            "dataset": cfg.task.dataset.name,
            "history_size": interaction_history_size
        })
        
    if cfg.experiment.debug_print:
        for i in range(5):
            print("=" * 100)
            print(f"Sample {i + 1}:")
            print("Prompt:", prompt_list[i])
            print("Interaction History:", final_return_list[i]["interaction_history"])
            print("Candidate Item:", final_return_list[i]["candidate_item"])
            print("=" * 100)
    
    print(f"Generated {len(final_return_list)} synthetic interactions")
    
    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, prompt_list)
    
    for i, result in enumerate(results):
        final_return_list[i]["response"] = result
        try:
            response_parser = hydra.utils.get_method(cfg.inference.response_parser)
            final_return_list[i]["parsed_response"] = response_parser(result)
        except Exception as e:
            print(f"Warning: Failed to parse response {i}: {e}")
            final_return_list[i]["parsed_response"] = None
    
    if not os.path.exists(cfg.experiment.output_dir):
        os.makedirs(cfg.experiment.output_dir)
    
    pkl_path = os.path.join(cfg.experiment.output_dir, "inference_results.pkl")
    csv_path = os.path.join(cfg.experiment.output_dir, "inference_results.csv")
    
    pkl.dump(final_return_list, open(pkl_path, "wb"))
    pd.DataFrame(final_return_list).to_csv(csv_path, index=False)
    
    print(f"Inference Results Dumped to {csv_path}")
    
    return final_return_list
