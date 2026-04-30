import os
import json
from pathlib import Path

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from datasets import load_dataset

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


class SimpleSftTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def learn(self, game_registry: GameRegistry):
        configure_wandb_env()

        # Load a conversational dataset for SFT, that is, a list of "messages" -- basically tuples of role and content.
        # The role can be "user" or "assistant" and typically alternates within the list.
        # During training, everything up to the last assistant message becomes the prefix for prediction.
        # The loss is calculated based on the differences to the last assistant message.
        # Here we load the canonical training split as available in the huggingface playpen-data repository.
        # By default, the dataset is stored in ~/.cache/huggingface/datasets/ on your machine. This might take a while.
        dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")

        # Only use successful episodes.
        dataset = dataset.filter(
            lambda episode: (episode.get("meta") or {}).get("outcome", "").lower() == "success"
        )

        # We shuffle and split the remaining filtered samples to receive a dev split
        # For evaluation on the actual games performance use the validation split
        # load_dataset("json", data_files="examples/trl/results.jsonl", split="validation")
        dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)

        run_wandb = wandb is not None and use_wandb()
        wandb_run_name = f"{self.learner.name}-sft-simple"
        if run_wandb:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "llama3-sft-adapters"),
                name=wandb_run_name,
                group=f"{self.learner.name}-sft-adapters",
                reinit=True,
            )

        # Initialize training configuration
        config = trl.SFTConfig(  # inherits TrainingArguments
            max_length=300,
            output_dir=f"models/sft/{self.learner.name}",
            eval_strategy="epoch",
            bf16=use_bf16(),
            report_to="wandb" if run_wandb else "none",
            run_name=wandb_run_name,
            logging_dir="./logs/sft-simple",
        )

        # Initialize trainer context
        trainer = trl.SFTTrainer(
            model=prepare_model_for_trainer(self.learner.model),
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],  # Note: we use a subset of train as dev
            args=config
        )

        # Train on the dataset
        trainer.train()

        if run_wandb:
            try:
                wandb.finish()
            except Exception as e:
                print(f"W&B finish failed (maybe not initialized): {e}")
