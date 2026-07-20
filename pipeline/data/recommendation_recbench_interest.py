import json
import os
from typing import Any
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl


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

# Item type labels for prompts (singular form for cleaner prompts)
DATASET_ITEM_TYPES = {
    'mind': 'News Article',
    'pens': 'News Article',
    'goodreads': 'Book',
    'books': 'Book',
    'movielens': 'Movie',
    'microlens': 'Movie',
    'netflix': 'Movie',
    'cds': 'Music Album',
    'lastfm': 'Music Track',
    'hm': 'Fashion Item',
    'pog': 'Fashion Product',
    'electronics': 'Electronics Product',
    'steam': 'Video Game',
    'hotelrec': 'Hotel',
    'yelp': 'Restaurant',
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


def get_dataset_item_type(cfg: DictConfig):
    """
    Get the item type label for the dataset (e.g., "News Article", "Book")
    """
    dataset_name = cfg.task.dataset.name
    return DATASET_ITEM_TYPES.get(dataset_name, 'Item')


def generate_prompts(cfg: DictConfig, item_description: str, user_name: str, user_profile: str):
    """Generate prompt for interest prediction"""
    prompt_template = cfg.task.prompt.base
    topic = get_dataset_topic(cfg)
    item_type = get_dataset_item_type(cfg)
    
    prompt = prompt_template.replace("TEMPLATE_ITEM_DESCRIPTION", item_description)
    prompt = prompt.replace("TEMPLATE_USER_NAME", user_name)
    prompt = prompt.replace("TEMPLATE_USER_PROFILE", user_profile)
    prompt = prompt.replace("TEMPLATE_ITEM_TYPE", item_type)
    prompt = prompt.replace("TEMPLATE_TOPIC", topic)
    
    return prompt


def load_persona_data(cfg: DictConfig):
    """Load persona data from CSV"""
    persona_path = cfg.task.dataset.persona_path
    persona_data = pd.read_csv(persona_path)
    return persona_data


def recommendation_recbench_interest(cfg: DictConfig):
    """
    Generate interest prediction data for any RecBench dataset with personas
    """
    print(f"Task: Recommendation RecBench Interest for {cfg.task.dataset.name}")
    
    dataset_load_method = hydra.utils.get_method(cfg.task.dataset.load_method)
    item_pool = dataset_load_method(cfg)
    
    if len(item_pool) == 0:
        raise ValueError("Item pool is empty!")
    
    persona_data = load_persona_data(cfg)
    
    # Sample personas
    sampled_persona = persona_data[
        cfg.task.persona_start_index: 
        min(cfg.task.persona_start_index + cfg.task.maximum_persona_number, len(persona_data))
    ]
    
    print(f"Processing {len(sampled_persona)} personas")
    
    prompt_list = []
    final_result_list = []
    
    for index, row in sampled_persona.iterrows():
        user_name = row["user_id"]
        user_profile = row["user_profile"]
        
        # Sample items for this persona
        num_items_to_sample = min(cfg.task.maximum_product_number_per_person, len(item_pool))
        sampled_items = item_pool.sample(num_items_to_sample)
        
        for item_description in sampled_items:
            prompt = generate_prompts(cfg, item_description, user_name, user_profile)
            prompt_list.append(prompt)
            final_result_list.append({
                "user_name": user_name, 
                "user_profile": user_profile, 
                "item_description": item_description, 
                "dataset": cfg.task.dataset.name,
                "prompt": prompt,
            })
    
    if cfg.experiment.debug_print:
        print("First 5 prompts:")
        for i in range(min(5, len(prompt_list))):
            print("=" * 100)
            print(f"Sample {i + 1}:")
            print(prompt_list[i])
            print("=" * 100)
    
    print(f"Number of prompts: {len(prompt_list)}")
    
    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, prompt_list, final_result_list)
    
    for i, result in enumerate(results):
        final_result_list[i]["response"] = result
        try:
            if 'response_parser' in cfg.inference:
                response_parser = hydra.utils.get_method(cfg.inference.response_parser)
                final_result_list[i]["parsed_response"] = response_parser(result)
        except Exception as e:
            print(f"Warning: Failed to parse response {i}: {e}")
            final_result_list[i]["parsed_response"] = None
    
    if not os.path.exists(cfg.experiment.output_dir):
        os.makedirs(cfg.experiment.output_dir)
    
    pkl_path = os.path.join(cfg.experiment.output_dir, "inference_results.pkl")
    csv_path = os.path.join(cfg.experiment.output_dir, "inference_results.csv")
    
    print(f"Inference Results Dumped to {csv_path}")
    pkl.dump(final_result_list, open(pkl_path, "wb"))
    pd.DataFrame(final_result_list).to_csv(csv_path, index=False)
    
    return final_result_list

