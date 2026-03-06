import os
import json
import ast
import copy
import math
import argparse
import re
from datetime import datetime
from collections import Counter
from pathlib import Path

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
import trl

from playpen import BasePlayPen
from playpen.training_utils import prepare_model_for_trainer

try:
    import wandb
except Exception:
    wandb = None


def use_bf16() -> bool:
    for env_name in ("PLAYPEN_BF16", "PLAYPEN_BF_16", "PLAYPEN_BF"):
        raw = os.getenv(env_name)
        if raw is None:
            continue
        raw = raw.strip().lower()
        if "=" in raw:
            raw = raw.split("=")[-1]
        return raw not in {"0", "false", "no", "off"}
    return True


def use_wandb() -> bool:
    raw = os.getenv("PLAYPEN_WANDB", "1")
    return raw.lower() not in {"0", "false", "no", "off"}


def configure_wandb_env() -> None:
    # Optional local key injection; harmless when not present (e.g., cluster secrets).
    key_path = Path(__file__).resolve().parents[2] / "key.json"
    if not key_path.exists():
        return
    try:
        key = json.loads(key_path.read_text(encoding="utf-8"))
        api_key = ((key or {}).get("wandb") or {}).get("api_key")
        if api_key and not os.getenv("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = str(api_key)
    except Exception:
        pass
    os.environ.setdefault("WANDB_PROJECT", "llama3-sft-adapters")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", str(value).strip().lower()).strip("-")
    return slug or "model"


def _resolve_adapter_reference(output_dir: str) -> str:
    out_dir = Path(output_dir)
    checkpoint_dirs = []
    for p in out_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            idx = int(p.name.split("checkpoint-")[-1])
        except Exception:
            idx = -1
        checkpoint_dirs.append((idx, p))
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda item: item[0])
        return str(checkpoint_dirs[-1][1])
    return str(out_dir)


def export_adapter_registry_snapshot(base_model_spec: dict, trained_runs: list) -> Path:
    if not trained_runs:
        raise ValueError("No trained runs provided for registry export.")

    source_registry = Path(
        os.getenv("PLAYPEN_SOURCE_MODEL_REGISTRY", "model_registry.json")
    ).expanduser()
    source_entries = []
    if source_registry.exists() and source_registry.is_file():
        payload = json.loads(source_registry.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            source_entries = [payload]
        elif isinstance(payload, list):
            source_entries = payload
        else:
            raise ValueError("Source model registry must be a JSON object or list.")

    base_spec = copy.deepcopy(base_model_spec or {})
    if not isinstance(base_spec, dict):
        raise ValueError("Invalid base model spec for registry export.")
    base_name = str(base_spec.get("model_name") or "model")
    base_config = dict(base_spec.get("model_config") or {})
    base_config.pop("peft_model", None)
    base_config.pop("peft_models", None)
    base_config.pop("merge", None)

    new_entries = []
    for run in trained_runs:
        run_suffix = str(run["run_suffix"])
        model_name = f"{base_name}-sft" if run_suffix == "all" else f"{base_name}-sft-{run_suffix}"

        entry = copy.deepcopy(base_spec)
        entry.pop("lookup_source", None)
        entry["model_name"] = model_name

        model_config = dict(base_config)
        model_config["peft_model"] = str(run["adapter_ref"])
        entry["model_config"] = model_config
        new_entries.append(entry)

    entries = [item for item in source_entries if isinstance(item, dict)]
    by_name = {}
    for idx, item in enumerate(entries):
        name = item.get("model_name")
        if isinstance(name, str) and name:
            by_name[name] = idx
    for entry in new_entries:
        name = entry["model_name"]
        if name in by_name:
            entries[by_name[name]] = entry
        else:
            by_name[name] = len(entries)
            entries.append(entry)

    out_dir = Path(os.getenv("PLAYPEN_MODEL_REGISTRY_DIR", "model_registries"))
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    file_name = f"model_registry.{_safe_slug(base_name)}.sft.{timestamp}.json"
    out_path = out_dir / file_name
    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return out_path


def build_and_summarize_datasets():
    # === Load base playpen dataset ===
    playpen_dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")
    playpen_dataset = playpen_dataset.filter(
        lambda episode: (episode["meta"] or {}).get("outcome", "").lower() == "success"
    )

    # === SFT-Final-Dataset ===
    sft_final_dataset = load_dataset("clembench-playpen/SFT-Final-Dataset", split="train")

    def parse_and_clean_sft_messages(example):
        chat_data = []
        try:
            chat_data = json.loads(example["chat"])
        except (json.JSONDecodeError, TypeError):
            try:
                chat_data = ast.literal_eval(example["chat"])
            except (ValueError, SyntaxError) as e:
                print(
                    f"Warning: Could not parse chat field with json or ast for example: "
                    f"{example.get('game_id')}, Error: {e}"
                )
                return {"messages": []}
        except Exception as e:
            print(f"Unhandled error during chat parsing: {e} for example: {example.get('game_id')}")
            return {"messages": []}

        cleaned_messages = []
        for msg in chat_data:
            current_message = {}
            if "role" in msg and msg["role"] is not None:
                current_message["role"] = str(msg["role"])
            if "content" in msg and msg["content"] is not None:
                current_message["content"] = str(msg["content"])
            if "role" in current_message and "content" in current_message:
                cleaned_messages.append(current_message)
        return {"messages": cleaned_messages}

    sft_final_dataset = sft_final_dataset.map(
        parse_and_clean_sft_messages,
        load_from_cache_file=False,
        remove_columns=["chat"] + [col for col in sft_final_dataset.column_names if col == "messages"],
    )

    def is_success_episode(example):
        success_flag = example.get("Success")
        if success_flag is not None:
            return int(success_flag) == 1
        meta = example.get("meta") or {}
        outcome = meta.get("outcome") or example.get("outcome")
        if outcome is None:
            return True
        return str(outcome).lower() == "success"

    filtered_sft_final = sft_final_dataset.filter(is_success_episode, load_from_cache_file=False)
    if len(filtered_sft_final) == 0 and len(sft_final_dataset) > 0:
        print("Warning: No explicit success labels in SFT dataset, keeping all samples.")
        filtered_sft_final = sft_final_dataset
    sft_final_dataset = filtered_sft_final

    def normalize_game_name(example):
        meta = example.get("meta") or {}
        for key in ("game", "game_name", "game_type"):
            value = meta.get(key)
            if value:
                return str(value).lower()
        for key in ("game", "game_name"):
            value = example.get(key)
            if value:
                return str(value).lower()
        return "unknown"

    def annotate_dataset(dataset, label):
        return dataset.map(
            lambda example: {
                "game_name_normalized": normalize_game_name(example),
            },
            desc=f"Annotating {label}",
            load_from_cache_file=False,
        )

    playpen_dataset = annotate_dataset(playpen_dataset, "playpen")
    sft_final_dataset = annotate_dataset(sft_final_dataset, "sft final")
    combined_dataset = concatenate_datasets([playpen_dataset, sft_final_dataset])
    combined_split = combined_dataset.train_test_split(0.2, shuffle=True, seed=42)
    combined_train = combined_split["train"]
    combined_val = combined_split["test"]

    def summarize_games(label, dataset):
        game_counts = Counter(dataset["game_name_normalized"])
        total = len(dataset)
        print(f"[{label}] total={total}")
        for game_name, count in sorted(game_counts.items(), key=lambda item: (-item[1], item[0])):
            share = (count / total * 100) if total else 0
            print(f"  - {game_name}: {count} ({share:.2f}%)")

    print(f"Size of the combined dataset: {len(combined_dataset)}")
    summarize_games("playpen_full", playpen_dataset)
    summarize_games("sft_final_full", sft_final_dataset)
    print("=== Combined training set distribution ===")
    summarize_games("combined_train", combined_train)
    print("=== Combined validation set distribution ===")
    summarize_games("combined_val", combined_val)

    return combined_train, combined_val


class PeftSftTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def learn(self, game_registry: GameRegistry):
        configure_wandb_env()
        combined_train, combined_val = build_and_summarize_datasets()

        def sanitize_game_name(game_name: str) -> str:
            slug = re.sub(r"[^a-z0-9]+", "_", str(game_name).strip().lower()).strip("_")
            return slug or "unknown"

        # Prepare per-game datasets (one adapter per game).
        game_train_datasets = {}
        game_eval_datasets = {}
        for game_name in sorted(set(combined_train["game_name_normalized"])):
            filtered = combined_train.filter(
                lambda example, gname=game_name: example["game_name_normalized"] == gname,
                load_from_cache_file=False,
            )
            if len(filtered):
                game_train_datasets[game_name] = filtered
            eval_subset = combined_val.filter(
                lambda example, gname=game_name: example["game_name_normalized"] == gname,
                load_from_cache_file=False,
            )
            if len(eval_subset):
                game_eval_datasets[game_name] = eval_subset

        # === Snapshot a clean base model so each adapter starts from identical weights ===
        base_model = copy.deepcopy(self.learner.model)
        base_model.to("cpu")
        base_model.eval()

        # Helper to train adapters for a given dataset selection
        def train_adapter(train_dataset, eval_dataset, run_suffix):
            total_samples = len(train_dataset)
            print(f"Starting training for target '{run_suffix}' with {total_samples} samples.")
            output_dir = f"models/sft+lora/{self.learner.name}-adapters/{run_suffix}"

            # Decide schedule based on dataset size (more epochs for smaller per-game datasets)
            is_game = run_suffix.startswith("game_")

            # Effective batch size: 2 * 8 = 16 episodes/step
            eff_batch_size = 2 * 8
            steps_per_epoch = max(1, math.ceil(total_samples / eff_batch_size))

            learning_rate = 5e-6  # same LR for all adapters

            if is_game:
                if total_samples < 3000:
                    num_train_epochs = 4.0
                elif total_samples < 6000:
                    num_train_epochs = 3.0
                else:
                    num_train_epochs = 3.0
            else:
                num_train_epochs = 2.0

            approx_steps = int(num_train_epochs * steps_per_epoch)
            print(
                f"[{run_suffix}] epochs={num_train_epochs}, "
                f"steps/epoch={steps_per_epoch}, ~total_steps={approx_steps}, lr={learning_rate}"
            )

            # fresh copy of the base model for every adapter
            model_for_adapter = copy.deepcopy(base_model)
            model_for_adapter = prepare_model_for_trainer(model_for_adapter)

            run_wandb = wandb is not None and use_wandb()
            # Explicit W&B run per adapter
            wandb_run_name = f"{self.learner.name}-sft-{run_suffix}"
            if run_wandb:
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "llama3-sft-adapters"),
                    name=wandb_run_name,
                    group=f"{self.learner.name}-sft-adapters",
                    reinit=True,
                )

            config = trl.SFTConfig(  # inherits TrainingArguments
                max_length=2048,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=8,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                lr_scheduler_type="linear",
                warmup_ratio=0.03,
                bf16=use_bf16(),
                gradient_checkpointing=True,
                seed=7331,
                output_dir=output_dir,
                eval_strategy="epoch",
                logging_steps=10,
                save_strategy="steps",
                save_steps=100,
                save_total_limit=3,
                report_to="wandb" if run_wandb else "none",
                run_name=wandb_run_name,
                logging_dir=f"./logs/{run_suffix}",
            )

            # Train LoRA adapters on all linear layers.
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_tokens"],
                task_type="CAUSAL_LM",
            )

            trainer = trl.SFTTrainer(
                model=model_for_adapter,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset, 
                args=config,
                peft_config=lora_cfg,
            )

            trainer.train()
            # Save adapter weights without merging into base model
            trainer.model.save_pretrained(output_dir)

            # Close this W&B run so the next adapter gets a fresh one
            if run_wandb:
                try:
                    wandb.finish()
                except Exception as e:
                    print(f"W&B finish failed (maybe not initialized): {e}")

            return _resolve_adapter_reference(output_dir)

        # Job catalog:
        # - all: one adapter on the full combined dataset
        # - game_<name>: one adapter per individual game
        available_jobs = []
        full_eval_dataset = combined_val if len(combined_val) else None
        available_jobs.append(
            {
                "suffix": "all",
                "label": "combined",
                "train_dataset": combined_train,
                "eval_dataset": full_eval_dataset,
            }
        )
        for game_name in sorted(game_train_datasets):
            eval_ds = game_eval_datasets.get(game_name)
            available_jobs.append(
                {
                    "suffix": f"game_{sanitize_game_name(game_name)}",
                    "label": game_name,
                    "train_dataset": game_train_datasets[game_name],
                    "eval_dataset": eval_ds,
                }
            )

        game_jobs = [job for job in available_jobs if job["suffix"].startswith("game_")]
        if not game_jobs:
            raise ValueError("No per-game training jobs are available.")
        print(f"Prepared {len(game_jobs)} per-game adapter jobs.")

        # Default training: all per-game adapters (e.g. 14 games -> 14 adapters).
        training_jobs = game_jobs

        # Selector via env:
        # - all-games: all game_* jobs
        # - all: combined + all game_* jobs
        # - combined/general: only the combined job
        # - or explicit suffixes/game names separated by commas
        requested_targets = os.getenv("PLAYPEN_TRAIN_TARGETS") or os.getenv("PLAYPEN_TRAIN_TARGET")
        if requested_targets:
            requested = {target.strip().lower() for target in requested_targets.split(",") if target.strip()}
            if "all-games" in requested:
                training_jobs = game_jobs
            elif "all" in requested:
                training_jobs = available_jobs
            elif "combined" in requested or "general" in requested:
                training_jobs = [available_jobs[0]]
            else:
                training_jobs = []
                for job in available_jobs:
                    suffix = job["suffix"].lower()
                    label = str(job["label"]).lower()
                    plain_suffix = suffix.replace("game_", "", 1) if suffix.startswith("game_") else suffix
                    if suffix in requested or plain_suffix in requested or label in requested:
                        training_jobs.append(job)
                if not training_jobs:
                    raise ValueError(f"No matching training jobs for PLAYPEN_TRAIN_TARGETS={requested_targets}")

        # Run training job(s).
        trained_runs = []
        for job in training_jobs:
            run_suffix = job["suffix"]
            train_ds = job["train_dataset"]
            eval_ds = job["eval_dataset"]
            print(f"=== Adapter dataset breakdown: {run_suffix} ({job['label']}) ===")
            adapter_ref = train_adapter(train_ds, eval_ds, run_suffix)
            trained_runs.append({
                "run_suffix": run_suffix,
                "label": job["label"],
                "adapter_ref": adapter_ref,
            })

        try:
            registry_path = export_adapter_registry_snapshot(self.learner.model_spec.to_dict(), trained_runs)
            print(f"Wrote adapter registry snapshot: {registry_path}")
        except Exception as e:
            print(f"Warning: Could not export adapter registry snapshot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-only", action="store_true",
                        help="Print dataset statistics without loading a model or starting training.")
    args = parser.parse_args()
    if args.stats_only:
        build_and_summarize_datasets()
