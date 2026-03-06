import os

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from peft import LoraConfig

from playpen import BasePlayPen
from playpen.training_utils import prepare_model_for_trainer


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


class PeftGRPOTrainer(BasePlayPen):
    # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Initialize training configuration
        self.config = trl.GRPOConfig(
            num_train_epochs=10,
            disable_dropout=True,
            max_prompt_length=None,
            max_completion_length=300,
            output_dir=f"models/sft+lora/{self.learner.name}",
            eval_strategy="epoch",
            bf16=use_bf16()
        )
        self.peft_config = LoraConfig(  # see https://huggingface.co/docs/trl/sft_trainer#training-adapters
            r=16, lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_token"],
            task_type="CAUSAL_LM",
        )

    def learn(self, game_registry: GameRegistry):
        # Initialize trainer context
        trainer = trl.GRPOTrainer(
            model=prepare_model_for_trainer(self.learner.model),
            processing_class=self.learner.tokenizer,
            train_dataset=...,
            eval_dataset=...,
            args=self.config,
            peft_config=self.peft_config
        )

        # Train on the dataset; this will save only the adapters to the checkpoints directory
        trainer.train()
