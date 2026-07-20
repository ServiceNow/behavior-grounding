import json
import random
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


SYSTEM_PROMPT = (
    "You are a recommender. "
    'I will provide user behavior sequence, and a candidate item. '
    'Please response "YES" or "NO" to represent whether this user is interested in this item. '
    "You are not allowed to response any other words for any explanation or note. "
    "Now, your role formally begins. Any other information should not disturb you."
)


DEFAULT_HISTORY_BOUNDS = {
    "books": (1, 52),
    "lastfm": (100, 100),
    "microlens": (3, 15),
    "netflix": (1, 100),
    "pens": (1, 274),
}


REQUIRED_INTEREST_COLUMNS = ["user_name", "item_description", "response"]


def read_interest_inference_frame(inference_csv_path: Path) -> pd.DataFrame:
    available_columns = pd.read_csv(inference_csv_path, nrows=0).columns.tolist()
    missing_columns = [
        column for column in REQUIRED_INTEREST_COLUMNS if column not in available_columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required inference columns {missing_columns} in {inference_csv_path}"
        )
    return pd.read_csv(inference_csv_path, usecols=REQUIRED_INTEREST_COLUMNS)


def normalize_label_value(value: object) -> str:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return str(value).strip().upper()


def process_response(response: str) -> dict | None:
    if not isinstance(response, str) or not response.strip():
        return None

    candidates: list[str] = []
    if "assistantfinal" in response:
        candidates.append(response.split("assistantfinal")[-1].strip())

    last_open = response.rfind("{")
    last_close = response.rfind("}")
    if last_open != -1 and last_close != -1 and last_open < last_close:
        candidates.append(response[last_open : last_close + 1].strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def build_user_prompt(user_prompt_template: str, history_items: list[str], candidate_item: str) -> str:
    numbered_history = [f"({index + 1}) {item}" for index, item in enumerate(history_items)]
    history_str = "\n".join(numbered_history)
    return user_prompt_template.format(history=history_str, candidate=candidate_item)


def create_clean_interest_frame(cfg: DictConfig, frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["processed_response"] = cleaned["response"].apply(process_response)
    cleaned = cleaned[cleaned["processed_response"].notna()].copy()
    cleaned["justification"] = cleaned["processed_response"].apply(
        lambda value: value.get("Justification") if isinstance(value, dict) else None
    )
    cleaned["is_interested"] = cleaned["processed_response"].apply(
        lambda value: value.get("IsInterested") if isinstance(value, dict) else None
    )
    cleaned["is_interested"] = cleaned["is_interested"].apply(normalize_label_value)
    accepted_labels = {
        normalize_label_value(label) for label in cfg.task.labels.accepted
    }
    cleaned = cleaned[cleaned["is_interested"].isin(accepted_labels)].copy()
    cleaned = cleaned[cleaned["justification"].notna()].copy()
    return cleaned


def resolve_history_bounds(cfg: DictConfig) -> tuple[int, int]:
    dataset_name = cfg.task.dataset.name
    if dataset_name in cfg.task.history_bounds:
        bounds = cfg.task.history_bounds[dataset_name]
        return int(bounds[0]), int(bounds[1])
    if dataset_name in DEFAULT_HISTORY_BOUNDS:
        return DEFAULT_HISTORY_BOUNDS[dataset_name]
    raise ValueError(f"No history bounds configured for dataset={dataset_name}")


def build_sft_examples(cfg: DictConfig, cleaned_interest: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(cfg.experiment.seed)
    min_history_length, max_history_length = resolve_history_bounds(cfg)
    positive_label = normalize_label_value(cfg.task.labels.positive)
    negative_label = normalize_label_value(cfg.task.labels.negative)

    sft_rows: list[dict[str, object]] = []
    grouped = cleaned_interest.groupby("user_name", sort=False)

    for user_name, user_frame in grouped:
        yes_frame = user_frame[user_frame["is_interested"] == positive_label].reset_index(drop=True)
        no_frame = user_frame[user_frame["is_interested"] == negative_label].reset_index(drop=True)

        available_history_pool = len(yes_frame) - 1
        if available_history_pool < min_history_length or no_frame.empty:
            continue

        history_length = rng.randint(
            min_history_length, min(max_history_length, available_history_pool)
        )

        positive_index = rng.randrange(len(yes_frame))
        positive_candidate = yes_frame.iloc[positive_index]
        remaining_yes_frame = yes_frame.drop(index=positive_index).reset_index(drop=True)

        history_indices = rng.sample(range(len(remaining_yes_frame)), history_length)
        history_items = [
            remaining_yes_frame.iloc[index]["item_description"] for index in history_indices
        ]

        negative_index = rng.randrange(len(no_frame))
        negative_candidate = no_frame.iloc[negative_index]

        for candidate_item, answer in (
            (positive_candidate["item_description"], positive_label),
            (negative_candidate["item_description"], negative_label),
        ):
            sft_rows.append(
                {
                    "user_id": user_name,
                    "dataset": cfg.task.dataset.name,
                    "source_variant": cfg.task.variant,
                    "history_length": history_length,
                    "candidate_item": candidate_item,
                    "question": build_user_prompt(
                        cfg.task.user_prompt_template, history_items, candidate_item
                    ),
                    "answer": answer,
                    "system_prompt": cfg.task.system_prompt,
                }
            )

    if not sft_rows:
        raise ValueError("No SFT examples were created from the cleaned interest data.")

    return pd.DataFrame(sft_rows)


def write_splits(cfg: DictConfig, sft_frame: pd.DataFrame) -> None:
    output_paths = {
        "cleaned_interest_path": Path(cfg.task.output.cleaned_interest_path),
        "sft_full_path": Path(cfg.task.output.sft_full_path),
        "sft_train_path": Path(cfg.task.output.sft_train_path),
        "sft_test_path": Path(cfg.task.output.sft_test_path),
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    sft_frame.to_csv(output_paths["sft_full_path"], index=False)

    train_frame = sft_frame.sample(frac=cfg.task.train_ratio, random_state=cfg.experiment.seed)
    test_frame = sft_frame.drop(train_frame.index)

    train_frame.to_csv(output_paths["sft_train_path"], index=False)
    test_frame.to_csv(output_paths["sft_test_path"], index=False)

    print(f"Wrote full SFT data to {output_paths['sft_full_path']}")
    print(f"Wrote train split ({len(train_frame)}) to {output_paths['sft_train_path']}")
    print(f"Wrote test split ({len(test_frame)}) to {output_paths['sft_test_path']}")


def recbench_persona_sft_from_inference(cfg: DictConfig):
    inference_csv_path = Path(cfg.task.dataset.inference_csv_path)
    if not inference_csv_path.exists():
        raise FileNotFoundError(f"Inference CSV not found: {inference_csv_path}")

    print(f"Loading interest inference results from {inference_csv_path}")
    interest_frame = read_interest_inference_frame(inference_csv_path)
    cleaned_interest = create_clean_interest_frame(cfg, interest_frame)

    cleaned_interest_path = Path(cfg.task.output.cleaned_interest_path)
    cleaned_interest_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_interest.to_csv(cleaned_interest_path, index=False)
    print(f"Wrote cleaned interest data ({len(cleaned_interest)}) to {cleaned_interest_path}")

    sft_frame = build_sft_examples(cfg, cleaned_interest)
    write_splits(cfg, sft_frame)
    return sft_frame
