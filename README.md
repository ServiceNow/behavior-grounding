# Behaviorally Grounded User Profiles from the Wild

Code for the paper **"Behaviorally Grounded User Profiles from the Wild for Personalized Alignment and
Multi-Perspective Reasoning."**

We introduce **profile behavioral grounding**: a framework that extracts open-ended, high-fidelity user profiles
directly from authentic, anonymized social-media activity, and uses them to personalize language models. Unlike
rigid synthetic personas built from a few categorical attributes, these behavior-derived profiles capture the
nuanced, idiosyncratic signals that drive real preferences. We study them under two paradigms:

- **Train-time personalization** — profile-conditioned synthetic data for supervised fine-tuning (SFT) on
  recommendation and open-ended QA.
- **Test-time multi-perspective reasoning** — a non-parametric approach that samples semantically relevant profiles
  as candidate "experts" and aggregates their viewpoints into a single multifaceted answer.





## Repository layout

```
conf/                     Hydra configuration
  config.yaml             top-level defaults
  model/                  model configs (Qwen3 8B/14B/32B, Gemma3 4B, Olmo3 7B, GPT-OSS 20B/120B, Qwen3 embedding)
  inference/              inference backends (vLLM, Transformers, OpenRouter)
  task/
    persona/              profile extraction (single/multi-tweet, dedupe/summarize)
    data/                 profile-conditioned SFT data synthesis + synthetic-baseline generation
    finetuning/           supervised fine-tuning
    evaluate/             URS scoring, finetuned-persona evaluation, persona prediction, profile QA
    test_time/            multi-perspective sampling, aggregation, scoring
pipeline/                 implementation for each task category (entry points referenced by the configs)
main.py                   Hydra entry point
env.yaml                  conda environment
```

## Setup

```bash
conda env create -f env.yaml
conda activate behavior_grounding
```

Requires Python 3.11 and CUDA-capable GPUs for training/inference.

### Environment variables

Configs reference these paths via environment variables (no absolute paths are hardcoded):

| Variable     | Meaning                                                          |
| ------------ | ---------------------------------------------------------------- |
| `PWD`        | Project root (set automatically to the current directory)        |
| `MODEL_ROOT` | Directory containing local model weights                         |
| `DATA_ROOT`  | Directory containing external datasets (see below)               |

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/datasets
```

### External datasets

These are not redistributed here; obtain them from their original sources and place them under `$DATA_ROOT`:

- **2 Million Bluesky Posts** (Dale, 2024, Apache 2.0) — raw posts for profile extraction.
- **RecBench** — pairwise recommendation evaluation.
- **URS Bench** — multi-intent user-query benchmark.

The extracted user profiles used throughout are released as a dataset (see [Data](#data)); place the CSVs under
`persistent_data/user_profile/` or point the relevant configs at them.

## Usage

The project uses [Hydra](https://hydra.cc/). Run any task with:

```bash
python main.py task=<CATEGORY>/<TASK> model=<MODEL> [overrides...]
```

Examples:

```bash
# 1. Extract open-ended profiles from social-media posts
python main.py task=persona/social_media_twitter_persona model=gptoss120b
python main.py task=persona/social_media_twitter_dedupe   model=gptoss120b

# 2. Synthesize profile-conditioned SFT data
python main.py task=data/recommendation_recbench_interest model=gptoss120b
python main.py task=data/urs_personalised_data            model=gptoss120b

# 3. Fine-tune a model on the synthetic data
python main.py task=finetuning/sft_v2 model=qwen3_8b

# 4. Evaluate
python main.py task=evaluate/urs_persona            model=qwen3_8b
python main.py task=evaluate/twitter_persona_predict model=qwen3_8b

# 5. Test-time multi-perspective reasoning
python main.py task=test_time/test_time_urs_related_persona model=qwen3_8b
python main.py task=test_time/test_time_urs_summary         model=gptoss120b
```

Override any config value on the command line, e.g.
`python main.py task=finetuning/sft_v2 model=qwen3_8b training.learning_rate=2e-5`.

Outputs default to `outputs/<package>/<task>/<model>/`. Experiment tracking via Weights & Biases is configured in
`conf/config.yaml`.

## Data

The anonymized user profiles are released as a companion dataset:

- `open_ended_profiles.csv` — 824 behaviorally grounded profiles extracted from real Bluesky histories.
- `synthetic_baseline_profiles.csv` — 842 purely synthetic baseline profiles.

Both are pseudonymized and scrubbed of handles, contact details, and URLs. See the dataset card for schema,
provenance, and ethical considerations.

## License

Released under the **Apache License 2.0** (see `LICENSE`), consistent with the source Bluesky corpus.

## Citation

```bibtex
@inproceedings{behaviorally_grounded_profiles,
  title     = {Behaviorally Grounded User Profiles from the Wild for Personalized Alignment and Multi-Perspective Reasoning},
  author    = {PLACEHOLDER},
  booktitle = {PLACEHOLDER},
  year      = {2026}
}
```
