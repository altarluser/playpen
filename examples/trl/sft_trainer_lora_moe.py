import ast
import copy
import csv
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import trl
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry
from playpen import BasePlayPen
from playpen.training_utils import prepare_model_for_trainer

try:
    import wandb
except Exception:
    wandb = None

try:
    import yaml
except Exception:
    yaml = None


DEFAULT_CONFIG = {
    "training": {
        "per_game_max_samples": 620,
        "max_length": 2048,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-6,
        "learning_rates": {
            "lora": 1e-5,
            "experts": 2e-5,
            "routers": 5e-5,
        },
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "gradient_checkpointing": True,
        "seed": 7331,
        "eval_strategy": "epoch",
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    },
    "moe": {
        "num_moe_layers": 6,
        "num_experts": 8,
        "top_k": 2,
        "expert_hidden_size": 2048,
        "activation": "silu",
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
    },
    "routing": {
        "flush_every_steps": 1,
    },
}


def parse_per_game_cap(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        iv = int(value)
        return None if iv <= 0 else iv
    text = str(value).strip().lower()
    if text in {"all", "max", "none", "no_cap", "nocap", "unlimited", "-1", "0"}:
        return None
    iv = int(text)
    return None if iv <= 0 else iv


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


def wandb_init_timeout() -> int:
    raw = os.getenv("WANDB_INIT_TIMEOUT", "180")
    try:
        value = int(str(raw).strip())
        return max(30, value)
    except Exception:
        return 180


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config() -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(
        os.getenv("PLAYPEN_SFT_MOE_CONFIG", str(Path(__file__).with_suffix(".yaml")))
    ).expanduser()
    if cfg_path.exists():
        if yaml is None:
            raise RuntimeError(f"PyYAML is required to read config at {cfg_path}")
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config must be a top-level object: {cfg_path}")
        _deep_update(cfg, payload)
        print(f"Loaded MoE SFT config from {cfg_path}")
    else:
        print(f"Config file not found at {cfg_path}; using built-in defaults.")

    cfg["training"]["per_game_max_samples"] = parse_per_game_cap(
        cfg["training"].get("per_game_max_samples")
    )
    cap_raw = os.getenv("PLAYPEN_PER_GAME_MAX_SAMPLES")
    if cap_raw is not None and cap_raw.strip():
        cfg["training"]["per_game_max_samples"] = parse_per_game_cap(cap_raw.strip())
    return cfg


def configure_wandb_env() -> None:
    key_path = Path(__file__).resolve().parents[2] / "key.json"
    if key_path.exists():
        try:
            key = json.loads(key_path.read_text(encoding="utf-8"))
            api_key = ((key or {}).get("wandb") or {}).get("api_key")
            if api_key and not os.getenv("WANDB_API_KEY"):
                os.environ["WANDB_API_KEY"] = str(api_key)
        except Exception:
            pass
    os.environ.setdefault("WANDB_PROJECT", "llama3-sft-moe-adapters")


def resolve_trainer_device():
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    local_rank = os.getenv("LOCAL_RANK")
    if local_rank is not None:
        try:
            return torch.device(f"cuda:{int(str(local_rank).strip())}")
        except Exception:
            pass
    return torch.device("cuda:0")


def sanitize_hf_device_maps(model) -> None:
    for module in model.modules():
        if hasattr(module, "hf_device_map"):
            try:
                value = getattr(module, "hf_device_map")
                if not isinstance(value, dict):
                    setattr(module, "hf_device_map", {})
            except Exception:
                pass
        if hasattr(module, "device_map"):
            try:
                value = getattr(module, "device_map")
                if value is not None and not isinstance(value, dict):
                    setattr(module, "device_map", None)
            except Exception:
                pass


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


def extract_moe_state_dict(model) -> dict:
    state = {}
    for key, value in model.state_dict().items():
        if ".mlp.router." not in key and ".mlp.experts." not in key:
            continue
        normalized = str(key)
        if normalized.startswith("module."):
            normalized = normalized[len("module."):]
        while normalized.startswith("base_model.model."):
            normalized = normalized[len("base_model.model."):]
        state[normalized] = value.detach().cpu()
    return state


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

    source_template = None
    for item in source_entries:
        if isinstance(item, dict) and str(item.get("model_name")) == base_name:
            source_template = copy.deepcopy(item)
            break

    new_entries = []
    for run in trained_runs:
        run_suffix = str(run["run_suffix"])
        model_name = f"{base_name}-sft" if run_suffix == "all" else f"{base_name}-sft-{run_suffix}"
        entry = copy.deepcopy(source_template if source_template is not None else base_spec)
        entry.pop("lookup_source", None)
        entry["model_name"] = model_name
        model_config = dict(entry.get("model_config") or {})
        model_config.pop("peft_models", None)
        model_config.pop("merge", None)
        model_config["moe_enabled"] = True
        model_config["moe_num_moe_layers"] = int(run["moe_num_moe_layers"])
        model_config["moe_num_experts"] = int(run["moe_num_experts"])
        model_config["moe_top_k"] = int(run["moe_top_k"])
        model_config["moe_expert_hidden_size"] = int(run["moe_expert_hidden_size"])
        model_config["moe_activation"] = str(run["moe_activation"])

        full_model_ref = run.get("full_model_ref")
        if full_model_ref:
            # Export eval-ready self-contained model path (LoRA merged + trained MoE weights).
            entry["huggingface_id"] = str(full_model_ref)
            model_config.pop("peft_model", None)
            model_config["requires_api_key"] = False
        else:
            model_config["peft_model"] = str(run["adapter_ref"])

        if run.get("adapter_ref"):
            model_config["moe_lora_adapter_path"] = str(run["adapter_ref"])
        if run.get("moe_state_path"):
            model_config["moe_state_path"] = str(run["moe_state_path"])
        if run.get("routing_path"):
            model_config["moe_routing_path"] = str(run["routing_path"])

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
    playpen_dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")
    playpen_dataset = playpen_dataset.filter(
        lambda episode: (episode["meta"] or {}).get("outcome", "").lower() == "success"
    )

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
                    f"Warning: Could not parse chat field for example {example.get('game_id')}: {e}"
                )
                return {"messages": []}
        except Exception as e:
            print(f"Unhandled chat parsing error for {example.get('game_id')}: {e}")
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
            lambda example: {"game_name_normalized": normalize_game_name(example)},
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


class ExpertFFN(torch.nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int, activation: str):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.fc2 = torch.nn.Linear(expert_hidden_size, hidden_size, bias=False)
        self.activation = activation

    def forward(self, x):
        if self.activation == "gelu":
            x = torch.nn.functional.gelu(self.fc1(x))
        else:
            x = torch.nn.functional.silu(self.fc1(x))
        return self.fc2(x)


class RoutingStatsManager:
    def __init__(self, layer_indices, num_experts: int, output_dir: Path):
        self.layer_indices = list(layer_indices)
        self.num_experts = int(num_experts)
        self.output_dir = Path(output_dir)
        self.routing_dir = self.output_dir / "routing"
        self.routing_dir.mkdir(parents=True, exist_ok=True)
        self.step_log_path = self.routing_dir / "routing_steps.jsonl"
        self.summary_path = self.routing_dir / "routing_summary.json"
        self.specialization_csv_path = self.routing_dir / "expert_specialization.csv"
        self._reset_step()
        self.cumulative = {}
        for layer_idx in self.layer_indices:
            self.cumulative[layer_idx] = {
                "tokens": 0.0,
                "top1": [0.0] * self.num_experts,
                "topk": [0.0] * self.num_experts,
                "gate_mass": [0.0] * self.num_experts,
                "entropy_sum": 0.0,
            }

    def _reset_step(self):
        self.step_stats = {}
        for layer_idx in self.layer_indices:
            self.step_stats[layer_idx] = {
                "tokens": 0.0,
                "top1": [0.0] * self.num_experts,
                "topk": [0.0] * self.num_experts,
                "gate_mass": [0.0] * self.num_experts,
                "entropy_sum": 0.0,
            }

    @torch.no_grad()
    def record(self, layer_idx: int, gate_probs: torch.Tensor, topk_idx: torch.Tensor):
        if layer_idx not in self.step_stats:
            return
        top1_idx = topk_idx[:, 0]
        topk_flat = topk_idx.reshape(-1)
        top1_counts = torch.bincount(top1_idx, minlength=self.num_experts).to(dtype=torch.float64)
        topk_counts = torch.bincount(topk_flat, minlength=self.num_experts).to(dtype=torch.float64)
        gate_mass = gate_probs.sum(dim=0).to(dtype=torch.float64)
        entropy = (-gate_probs * torch.log(gate_probs.clamp_min(1e-9))).sum().item()
        tokens = float(topk_idx.shape[0])

        self._update(self.step_stats[layer_idx], tokens, top1_counts, topk_counts, gate_mass, entropy)
        self._update(self.cumulative[layer_idx], tokens, top1_counts, topk_counts, gate_mass, entropy)

    @staticmethod
    def _update(target, tokens, top1_counts, topk_counts, gate_mass, entropy):
        target["tokens"] += tokens
        target["entropy_sum"] += float(entropy)
        for i in range(len(target["top1"])):
            target["top1"][i] += float(top1_counts[i].item())
            target["topk"][i] += float(topk_counts[i].item())
            target["gate_mass"][i] += float(gate_mass[i].item())

    def flush_step(self, step: int):
        if not any(self.step_stats[layer]["tokens"] > 0 for layer in self.layer_indices):
            return
        payload = {"step": int(step), "layers": {}}
        for layer_idx in self.layer_indices:
            stats = self.step_stats[layer_idx]
            if stats["tokens"] <= 0:
                continue
            total_top1 = max(1.0, sum(stats["top1"]))
            payload["layers"][str(layer_idx)] = {
                "tokens": stats["tokens"],
                "top1_counts": stats["top1"],
                "top1_share": [x / total_top1 for x in stats["top1"]],
                "topk_counts": stats["topk"],
                "gate_mass": stats["gate_mass"],
                "avg_entropy": stats["entropy_sum"] / max(1.0, stats["tokens"]),
            }
        with self.step_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        self._reset_step()

    def export_summary(self):
        summary = {"layers": {}, "global": {}}
        global_top1 = [0.0] * self.num_experts
        total_tokens = 0.0

        for layer_idx in self.layer_indices:
            stats = self.cumulative[layer_idx]
            tokens = max(1.0, stats["tokens"])
            top1_total = max(1.0, sum(stats["top1"]))
            top1_share = [x / top1_total for x in stats["top1"]]
            entropy = stats["entropy_sum"] / tokens
            specialization = 1.0 - (entropy / math.log(max(2, self.num_experts)))
            summary["layers"][str(layer_idx)] = {
                "tokens": stats["tokens"],
                "top1_counts": stats["top1"],
                "top1_share": top1_share,
                "topk_counts": stats["topk"],
                "gate_mass": stats["gate_mass"],
                "avg_entropy": entropy,
                "specialization_score": specialization,
            }
            for i in range(self.num_experts):
                global_top1[i] += stats["top1"][i]
            total_tokens += stats["tokens"]

        global_total = max(1.0, sum(global_top1))
        summary["global"] = {
            "total_tokens": total_tokens,
            "expert_top1_counts": global_top1,
            "expert_top1_share": [x / global_total for x in global_top1],
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        with self.specialization_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_idx", "expert_id", "top1_count", "top1_share"])
            for layer_idx in self.layer_indices:
                top1 = summary["layers"][str(layer_idx)]["top1_counts"]
                share = summary["layers"][str(layer_idx)]["top1_share"]
                for expert_id in range(self.num_experts):
                    writer.writerow([layer_idx, expert_id, top1[expert_id], share[expert_id]])


class RoutingExportCallback(TrainerCallback):
    def __init__(self, manager: RoutingStatsManager, flush_every_steps: int):
        self.manager = manager
        self.flush_every_steps = max(1, int(flush_every_steps))

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step or 0)
        if step > 0 and step % self.flush_every_steps == 0:
            self.manager.flush_step(step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        step = int(state.global_step or 0)
        self.manager.flush_step(step)
        self.manager.export_summary()
        return control


class SparseMoEFFN(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        num_experts: int,
        top_k: int,
        activation: str,
        layer_idx: int,
        routing_manager: RoutingStatsManager,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.layer_idx = int(layer_idx)
        self.routing_manager = routing_manager
        self.router = torch.nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            [ExpertFFN(self.hidden_size, int(expert_hidden_size), activation) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        gate_logits = self.router(flat)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        topk_weight, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            matches = (topk_idx == expert_id)
            if not torch.any(matches):
                continue
            token_idx, route_idx = torch.where(matches)
            expert_input = flat.index_select(0, token_idx)
            expert_output = expert(expert_input)
            weights = topk_weight[token_idx, route_idx].unsqueeze(-1).to(dtype=expert_output.dtype)
            routed = (expert_output * weights).to(dtype=output.dtype)
            output.index_add_(0, token_idx, routed)

        self.routing_manager.record(self.layer_idx, gate_probs.detach(), topk_idx.detach())
        return output.view(original_shape)


def replace_last_mlp_with_moe(model, moe_cfg: dict, routing_manager: RoutingStatsManager):
    layers = model.model.layers
    num_layers_total = len(layers)
    num_moe_layers = int(moe_cfg["num_moe_layers"])
    layer_indices = list(range(max(0, num_layers_total - num_moe_layers), num_layers_total))
    for idx in layer_indices:
        old_mlp = layers[idx].mlp
        hidden_size = int(old_mlp.down_proj.out_features)
        device = old_mlp.down_proj.weight.device
        dtype = old_mlp.down_proj.weight.dtype
        new_mlp = SparseMoEFFN(
            hidden_size=hidden_size,
            expert_hidden_size=int(moe_cfg["expert_hidden_size"]),
            num_experts=int(moe_cfg["num_experts"]),
            top_k=int(moe_cfg["top_k"]),
            activation=str(moe_cfg["activation"]).lower(),
            layer_idx=idx,
            routing_manager=routing_manager,
        ).to(device=device, dtype=dtype)
        layers[idx].mlp = new_mlp
    return layer_indices


def set_trainable_for_moe_lora(model) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, SparseMoEFFN):
            for p in module.parameters():
                p.requires_grad = True

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def print_trainable_params(model):
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    pct = 100.0 * trainable / max(1, total)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")


def build_moe_optimizer(model, train_cfg: dict):
    lr_cfg = dict(train_cfg.get("learning_rates") or {})
    lora_lr = float(lr_cfg.get("lora", 1e-5))
    experts_lr = float(lr_cfg.get("experts", 2e-5))
    routers_lr = float(lr_cfg.get("routers", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    groups = {
        "lora": [],
        "experts": [],
        "routers": [],
    }
    fallback = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            groups["lora"].append(p)
        elif ".mlp.experts." in name:
            groups["experts"].append(p)
        elif ".mlp.router." in name:
            groups["routers"].append(p)
        else:
            fallback.append((name, p))

    if fallback:
        # Keep strict behavior to avoid silently training unexpected params.
        fallback_names = ", ".join(name for name, _ in fallback[:8])
        raise ValueError(
            "Found trainable params not matched to lora/experts/routers groups: "
            f"{fallback_names}"
        )

    param_groups = []
    if groups["lora"]:
        param_groups.append({"params": groups["lora"], "lr": lora_lr, "weight_decay": weight_decay})
    if groups["experts"]:
        param_groups.append({"params": groups["experts"], "lr": experts_lr, "weight_decay": weight_decay})
    if groups["routers"]:
        param_groups.append({"params": groups["routers"], "lr": routers_lr, "weight_decay": weight_decay})

    if not param_groups:
        raise ValueError("No trainable parameter groups found for optimizer.")

    print(
        "Optimizer LR groups: "
        f"lora={lora_lr} ({len(groups['lora'])} tensors), "
        f"experts={experts_lr} ({len(groups['experts'])} tensors), "
        f"routers={routers_lr} ({len(groups['routers'])} tensors)"
    )

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


class PeftMoESftTrainer(BasePlayPen):
    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)

    def learn(self, game_registry: GameRegistry):
        configure_wandb_env()
        cfg = load_config()
        train_cfg = cfg["training"]
        moe_cfg = cfg["moe"]
        lora_cfg_raw = cfg["lora"]
        routing_cfg = cfg["routing"]
        combined_train, combined_val = build_and_summarize_datasets()

        per_game_max_samples = train_cfg["per_game_max_samples"]
        game_train_datasets = {}
        for game_name in sorted(set(combined_train["game_name_normalized"])):
            filtered = combined_train.filter(
                lambda example, gname=game_name: example["game_name_normalized"] == gname,
                load_from_cache_file=False,
            )
            if not len(filtered):
                continue
            if per_game_max_samples is not None and len(filtered) > per_game_max_samples:
                capped = filtered.shuffle(seed=42).select(range(per_game_max_samples))
                print(f"[game={game_name}] train samples capped to {len(capped)} (from {len(filtered)})")
                game_train_datasets[game_name] = capped
            else:
                print(f"[game={game_name}] train samples using all available rows: {len(filtered)}")
                game_train_datasets[game_name] = filtered

        if not game_train_datasets:
            raise ValueError("No per-game training datasets after filtering/capping.")

        capped_train_parts = [game_train_datasets[g] for g in sorted(game_train_datasets)]
        combined_capped_train = concatenate_datasets(capped_train_parts).shuffle(seed=42)
        if per_game_max_samples is None:
            print(f"Combined generalist train has {len(combined_capped_train)} samples (no per-game cap).")
        else:
            print(
                f"Combined capped generalist train has {len(combined_capped_train)} samples "
                f"(max {per_game_max_samples} per game)."
            )

        base_model = copy.deepcopy(self.learner.model)
        base_model.to("cpu")
        base_model.eval()

        run_suffix = "moe"
        output_dir = f"models/sft+lora/{self.learner.name}-adapters/{run_suffix}"
        train_device = resolve_trainer_device()
        print(f"[{run_suffix}] trainer device={train_device}")

        model_for_adapter = copy.deepcopy(base_model)
        model_for_adapter = prepare_model_for_trainer(model_for_adapter)

        num_layers_total = len(model_for_adapter.model.layers)
        layer_indices = list(
            range(max(0, num_layers_total - int(moe_cfg["num_moe_layers"])), num_layers_total)
        )
        routing_manager = RoutingStatsManager(
            layer_indices=layer_indices,
            num_experts=int(moe_cfg["num_experts"]),
            output_dir=Path(output_dir),
        )
        replaced = replace_last_mlp_with_moe(model_for_adapter, moe_cfg, routing_manager)
        print(f"Replaced MLP with MoE in layers: {replaced}")

        lora_cfg = LoraConfig(
            r=int(lora_cfg_raw["r"]),
            lora_alpha=int(lora_cfg_raw["alpha"]),
            lora_dropout=float(lora_cfg_raw["dropout"]),
            target_modules=list(lora_cfg_raw["target_modules"]),
            task_type=str(lora_cfg_raw["task_type"]),
            layers_to_transform=layer_indices,
            layers_pattern="layers",
        )
        model_for_adapter = get_peft_model(model_for_adapter, lora_cfg)
        set_trainable_for_moe_lora(model_for_adapter)
        print_trainable_params(model_for_adapter)
        if bool(train_cfg["gradient_checkpointing"]) and hasattr(model_for_adapter, "enable_input_require_grads"):
            try:
                model_for_adapter.enable_input_require_grads()
            except Exception as e:
                print(f"Warning: could not enable input requires_grad for checkpointing: {e}")

        model_for_adapter = model_for_adapter.to(train_device)
        sanitize_hf_device_maps(model_for_adapter)

        run_wandb = wandb is not None and use_wandb()
        wandb_initialized = False
        wandb_run_name = f"{self.learner.name}-sft-{run_suffix}"
        if run_wandb:
            try:
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "llama3-sft-moe-adapters"),
                    name=wandb_run_name,
                    group=f"{self.learner.name}-sft-moe",
                    settings=wandb.Settings(init_timeout=wandb_init_timeout()),
                )
                wandb_initialized = True
            except Exception as e:
                run_wandb = False
                print(
                    f"W&B init failed for {wandb_run_name} ({type(e).__name__}: {e}). "
                    "Continuing with report_to='none'."
                )

        eval_strategy = str(train_cfg["eval_strategy"])
        save_strategy = str(train_cfg["save_strategy"])
        load_best_model_at_end = bool(train_cfg.get("load_best_model_at_end", False))
        if load_best_model_at_end and save_strategy != eval_strategy:
            print(
                f"[{run_suffix}] load_best_model_at_end requires save_strategy==eval_strategy. "
                f"Overriding save_strategy '{save_strategy}' -> '{eval_strategy}'."
            )
            save_strategy = eval_strategy

        config = trl.SFTConfig(
            max_length=int(train_cfg["max_length"]),
            per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
            learning_rate=float(
                max(
                    float((train_cfg.get("learning_rates") or {}).get("lora", train_cfg.get("learning_rate", 5e-6))),
                    float((train_cfg.get("learning_rates") or {}).get("experts", train_cfg.get("learning_rate", 5e-6))),
                    float((train_cfg.get("learning_rates") or {}).get("routers", train_cfg.get("learning_rate", 5e-6))),
                )
            ),
            num_train_epochs=float(train_cfg["num_train_epochs"]),
            lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
            warmup_ratio=float(train_cfg["warmup_ratio"]),
            bf16=use_bf16(),
            gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
            seed=int(train_cfg["seed"]),
            output_dir=output_dir,
            eval_strategy=eval_strategy,
            logging_steps=int(train_cfg["logging_steps"]),
            save_strategy=save_strategy,
            save_steps=int(train_cfg["save_steps"]),
            save_total_limit=int(train_cfg["save_total_limit"]),
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=str(train_cfg.get("metric_for_best_model", "eval_loss")),
            greater_is_better=bool(train_cfg.get("greater_is_better", False)),
            report_to="wandb" if run_wandb else "none",
            run_name=wandb_run_name,
            logging_dir=f"./logs/{run_suffix}",
        )

        trainer = trl.SFTTrainer(
            model=model_for_adapter,
            train_dataset=combined_capped_train,
            eval_dataset=combined_val if len(combined_val) else None,
            args=config,
        )
        trainer.optimizer = build_moe_optimizer(model_for_adapter, train_cfg)
        trainer.add_callback(
            RoutingExportCallback(
                manager=routing_manager,
                flush_every_steps=int(routing_cfg.get("flush_every_steps", 1)),
            )
        )

        trainer.train()

        moe_state_dir = Path(output_dir) / "moe"
        moe_state_dir.mkdir(parents=True, exist_ok=True)
        moe_state_path = moe_state_dir / "moe_state.pt"
        moe_state = extract_moe_state_dict(trainer.model)
        if not moe_state:
            raise ValueError("No MoE parameters found to export; expected router/expert weights.")
        torch.save(moe_state, str(moe_state_path))

        adapter_dir = Path(output_dir) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))

        # Persist a self-contained model for evaluation:
        # base weights + trained MoE blocks + merged LoRA.
        full_model_dir = Path(output_dir) / "full_model"
        full_model_dir.mkdir(parents=True, exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(full_model_dir))
        self.learner.tokenizer.save_pretrained(str(full_model_dir))

        routing_manager.export_summary()
        if wandb_initialized:
            try:
                wandb.finish()
            except Exception as e:
                print(f"W&B finish failed: {e}")

        trained_runs = [
            {
                "run_suffix": run_suffix,
                "label": "moe_generalist",
                "adapter_ref": _resolve_adapter_reference(str(adapter_dir)),
                "full_model_ref": str(full_model_dir),
                "moe_state_path": str(moe_state_path),
                "routing_path": str(Path(output_dir) / "routing"),
                "moe_num_moe_layers": int(moe_cfg["num_moe_layers"]),
                "moe_num_experts": int(moe_cfg["num_experts"]),
                "moe_top_k": int(moe_cfg["top_k"]),
                "moe_expert_hidden_size": int(moe_cfg["expert_hidden_size"]),
                "moe_activation": str(moe_cfg["activation"]).lower(),
            }
        ]
        registry_path = export_adapter_registry_snapshot(self.learner.model_spec.to_dict(), trained_runs)
        print(f"Wrote adapter registry snapshot: {registry_path}")
        print(f"Adapter saved at: {adapter_dir}")
        print(f"Full merged model saved at: {full_model_dir}")
        print(f"MoE state saved at: {moe_state_path}")
        print(f"Routing analysis written to: {Path(output_dir) / 'routing'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print dataset statistics without loading a model or starting training.",
    )
    args = parser.parse_args()
    if args.stats_only:
        build_and_summarize_datasets()
