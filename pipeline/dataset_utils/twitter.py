from datasets import load_dataset
from datasets import Dataset
from omegaconf import DictConfig
import os



def load_bluesky_data(cfg: DictConfig) -> Dataset:
    # Load Bluesky JSONL dumps and normalize schema expected by evaluation tasks.
    path = cfg.task.dataset.path
    dataset = load_dataset("json", data_files=os.path.join(path, "*.jsonl"), num_proc=os.cpu_count())
    dataset = dataset["train"]
    if "author" in dataset.column_names and "user" not in dataset.column_names:
        dataset = dataset.rename_column("author", "user")
    if "text" in dataset.column_names and "tweet" not in dataset.column_names:
        dataset = dataset.rename_column("text", "tweet")
    return dataset

