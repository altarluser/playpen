from __future__ import annotations

import atexit
import json
import math
import os
import csv
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch


class EvalRoutingLogger:
    def __init__(self, log_dir: str, num_experts: int, flush_every: int = 8):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.step_log_path = self.log_dir / "routing_steps.jsonl"
        self.summary_path = self.log_dir / "routing_summary.json"
        self.latent_usage_csv_path = self.log_dir / "latent_eval_usage.csv"
        self.num_experts = int(num_experts)
        self.flush_every = max(1, int(flush_every))
        self.forward_calls = 0
        self.buffer: Dict[int, Dict[str, object]] = {}
        self.cumulative: Dict[int, Dict[str, object]] = {}
        self.context_cumulative: Dict[Tuple[str, str, str, int], Dict[str, object]] = {}
        atexit.register(self.close)

    def _empty_stats(self) -> Dict[str, object]:
        return {
            "tokens": 0.0,
            "top1": [0.0] * self.num_experts,
            "topk": [0.0] * self.num_experts,
            "gate_mass": [0.0] * self.num_experts,
            "entropy_sum": 0.0,
        }

    def _ensure_layer(self, layer_idx: int) -> None:
        if layer_idx not in self.buffer:
            self.buffer[layer_idx] = self._empty_stats()
        if layer_idx not in self.cumulative:
            self.cumulative[layer_idx] = self._empty_stats()

    @staticmethod
    def _context_from_env() -> Tuple[str, str, str]:
        game = str(os.getenv("PLAYPEN_EVAL_GAME", "unknown")).strip() or "unknown"
        split = str(os.getenv("PLAYPEN_EVAL_SPLIT", "unknown")).strip() or "unknown"
        regime = str(os.getenv("PLAYPEN_EVAL_REGIME", "unknown")).strip() or "unknown"
        return game, split, regime

    @staticmethod
    def _update(target: Dict[str, object], tokens, top1_counts, topk_counts, gate_mass, entropy) -> None:
        target["tokens"] = float(target["tokens"]) + float(tokens)
        target["entropy_sum"] = float(target["entropy_sum"]) + float(entropy)
        top1 = target["top1"]
        topk = target["topk"]
        mass = target["gate_mass"]
        for i in range(len(top1)):
            top1[i] += float(top1_counts[i].item())
            topk[i] += float(topk_counts[i].item())
            mass[i] += float(gate_mass[i].item())

    @torch.no_grad()
    def record(self, layer_idx: int, gate_probs: torch.Tensor, topk_idx: torch.Tensor) -> None:
        self._ensure_layer(int(layer_idx))
        game, split, regime = self._context_from_env()
        top1_idx = topk_idx[:, 0]
        topk_flat = topk_idx.reshape(-1)
        top1_counts = torch.bincount(top1_idx, minlength=self.num_experts).to(dtype=torch.float64)
        topk_counts = torch.bincount(topk_flat, minlength=self.num_experts).to(dtype=torch.float64)
        gate_mass = gate_probs.sum(dim=0).to(dtype=torch.float64)
        entropy = (-gate_probs * torch.log(gate_probs.clamp_min(1e-9))).sum().item()
        tokens = float(topk_idx.shape[0])

        self._update(self.buffer[int(layer_idx)], tokens, top1_counts, topk_counts, gate_mass, entropy)
        self._update(self.cumulative[int(layer_idx)], tokens, top1_counts, topk_counts, gate_mass, entropy)
        context_key = (game, split, regime, int(layer_idx))
        if context_key not in self.context_cumulative:
            self.context_cumulative[context_key] = self._empty_stats()
        self._update(self.context_cumulative[context_key], tokens, top1_counts, topk_counts, gate_mass, entropy)
        self.forward_calls += 1
        if self.forward_calls % self.flush_every == 0:
            self.flush_buffer()

    def flush_buffer(self) -> None:
        if not self.buffer:
            return
        payload = {"forward_call": int(self.forward_calls), "layers": {}}
        has_any = False
        for layer_idx in sorted(self.buffer):
            stats = self.buffer[layer_idx]
            tokens = float(stats["tokens"])
            if tokens <= 0:
                continue
            top1 = list(stats["top1"])
            total_top1 = max(1.0, sum(top1))
            payload["layers"][str(layer_idx)] = {
                "tokens": tokens,
                "top1_counts": top1,
                "top1_share": [x / total_top1 for x in top1],
                "topk_counts": list(stats["topk"]),
                "gate_mass": list(stats["gate_mass"]),
                "avg_entropy": float(stats["entropy_sum"]) / max(1.0, tokens),
            }
            self.buffer[layer_idx] = self._empty_stats()
            has_any = True
        if has_any:
            with self.step_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

    def export_summary(self) -> None:
        summary = {"layers": {}, "global": {}}
        global_top1 = [0.0] * self.num_experts
        total_tokens = 0.0

        for layer_idx in sorted(self.cumulative):
            stats = self.cumulative[layer_idx]
            tokens = max(1.0, float(stats["tokens"]))
            top1 = list(stats["top1"])
            top1_total = max(1.0, sum(top1))
            top1_share = [x / top1_total for x in top1]
            avg_entropy = float(stats["entropy_sum"]) / tokens
            specialization = 1.0 - (avg_entropy / math.log(max(2, self.num_experts)))
            summary["layers"][str(layer_idx)] = {
                "tokens": float(stats["tokens"]),
                "top1_counts": top1,
                "top1_share": top1_share,
                "topk_counts": list(stats["topk"]),
                "gate_mass": list(stats["gate_mass"]),
                "avg_entropy": avg_entropy,
                "specialization_score": specialization,
            }
            for i in range(self.num_experts):
                global_top1[i] += top1[i]
            total_tokens += float(stats["tokens"])

        global_total = max(1.0, sum(global_top1))
        summary["global"] = {
            "total_tokens": total_tokens,
            "expert_top1_counts": global_top1,
            "expert_top1_share": [x / global_total for x in global_top1],
            "forward_calls": int(self.forward_calls),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._export_context_usage_csv()

    def _export_context_usage_csv(self) -> None:
        with self.latent_usage_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "game",
                    "split",
                    "regime",
                    "layer",
                    "expert_id",
                    "top1_count",
                    "top1_proportion",
                    "tokens",
                    "avg_entropy",
                ]
            )
            for (game, split, regime, layer_idx), stats in sorted(self.context_cumulative.items()):
                top1 = list(stats["top1"])
                total_top1 = max(1.0, sum(top1))
                tokens = float(stats["tokens"])
                avg_entropy = float(stats["entropy_sum"]) / max(1.0, tokens)
                for expert_id in range(self.num_experts):
                    writer.writerow(
                        [
                            game,
                            split,
                            regime,
                            int(layer_idx),
                            int(expert_id),
                            float(top1[expert_id]),
                            float(top1[expert_id]) / total_top1,
                            tokens,
                            avg_entropy,
                        ]
                    )

    def close(self) -> None:
        try:
            self.flush_buffer()
            self.export_summary()
        except Exception:
            pass


class ExpertFFN(torch.nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int, activation: str):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.fc2 = torch.nn.Linear(expert_hidden_size, hidden_size, bias=False)
        self.activation = str(activation).lower()

    def forward(self, x):
        if self.activation == "gelu":
            x = torch.nn.functional.gelu(self.fc1(x))
        else:
            x = torch.nn.functional.silu(self.fc1(x))
        return self.fc2(x)


class SparseMoEFFN(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        num_experts: int,
        top_k: int,
        activation: str,
        layer_idx: int,
        routing_logger: Optional[EvalRoutingLogger] = None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.layer_idx = int(layer_idx)
        self.routing_logger = routing_logger
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
            matches = topk_idx == expert_id
            if not torch.any(matches):
                continue
            token_idx, route_idx = torch.where(matches)
            expert_input = flat.index_select(0, token_idx)
            expert_output = expert(expert_input)
            weights = topk_weight[token_idx, route_idx].unsqueeze(-1).to(dtype=expert_output.dtype)
            routed = (expert_output * weights).to(dtype=output.dtype)
            output.index_add_(0, token_idx, routed)

        if self.routing_logger is not None:
            self.routing_logger.record(self.layer_idx, gate_probs.detach(), topk_idx.detach())
        return output.view(original_shape)


class LlamaStyleExpertFFN(torch.nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int, activation: str):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.expert_hidden_size = int(expert_hidden_size)
        self.activation = str(activation).lower()
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.expert_hidden_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.expert_hidden_size, bias=False)
        self.down_proj = torch.nn.Linear(self.expert_hidden_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation != "silu":
            raise ValueError(f"Unsupported residual MoE activation: {self.activation}")
        gated = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


def _init_expert_from_dense(expert: LlamaStyleExpertFFN, dense_mlp) -> None:
    with torch.no_grad():
        dense_gate = getattr(dense_mlp, "gate_proj", None)
        dense_up = getattr(dense_mlp, "up_proj", None)
        dense_down = getattr(dense_mlp, "down_proj", None)
        if dense_gate is None or dense_up is None or dense_down is None:
            return

        dense_gate = dense_gate.weight
        dense_up = dense_up.weight
        dense_down = dense_down.weight

        expert.gate_proj.weight.zero_()
        expert.up_proj.weight.zero_()
        expert.down_proj.weight.zero_()

        gate_rows = min(expert.gate_proj.weight.shape[0], dense_gate.shape[0])
        gate_cols = min(expert.gate_proj.weight.shape[1], dense_gate.shape[1])
        expert.gate_proj.weight[:gate_rows, :gate_cols].copy_(dense_gate[:gate_rows, :gate_cols])

        up_rows = min(expert.up_proj.weight.shape[0], dense_up.shape[0])
        up_cols = min(expert.up_proj.weight.shape[1], dense_up.shape[1])
        expert.up_proj.weight[:up_rows, :up_cols].copy_(dense_up[:up_rows, :up_cols])

        down_rows = min(expert.down_proj.weight.shape[0], dense_down.shape[0])
        down_cols = min(expert.down_proj.weight.shape[1], dense_down.shape[1])
        expert.down_proj.weight[:down_rows, :down_cols].copy_(dense_down[:down_rows, :down_cols])


class ResidualSparseSkillMoEFFN(torch.nn.Module):
    def __init__(
        self,
        dense_mlp,
        hidden_size: int,
        expert_hidden_size: int,
        num_experts: int,
        top_k: int,
        activation: str,
        alpha: float,
        capacity_factor_train: float,
        capacity_factor_eval: float,
        layer_idx: int,
        routing_logger: Optional[EvalRoutingLogger] = None,
    ):
        super().__init__()
        self.dense_mlp = dense_mlp
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.activation = str(activation).lower()
        self.alpha = float(alpha)
        self.capacity_factor_train = float(capacity_factor_train)
        self.capacity_factor_eval = float(capacity_factor_eval)
        self.layer_idx = int(layer_idx)
        self.routing_logger = routing_logger
        self.router = torch.nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            [
                LlamaStyleExpertFFN(
                    hidden_size=self.hidden_size,
                    expert_hidden_size=int(expert_hidden_size),
                    activation=self.activation,
                )
                for _ in range(self.num_experts)
            ]
        )
        for expert in self.experts:
            _init_expert_from_dense(expert, self.dense_mlp)

        if self.top_k != 1:
            # Runtime supports only top-1 dispatch in residual mode.
            self.top_k = 1

    def _capacity(self, num_tokens: int) -> int:
        factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        return max(1, int(math.ceil(float(factor) * float(num_tokens) / max(1, self.num_experts))))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dense_out = self.dense_mlp(hidden_states)

        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        gate_logits = self.router(flat)
        gate_probs = torch.softmax(gate_logits.float(), dim=-1)
        top1_prob, top1_idx = torch.max(gate_probs, dim=-1)
        capacity = self._capacity(top1_idx.shape[0])

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx = torch.where(top1_idx == expert_id)[0]
            if token_idx.numel() == 0:
                continue
            probs = top1_prob.index_select(0, token_idx)
            order = torch.argsort(probs, descending=True)
            keep_n = min(capacity, token_idx.numel())
            kept = token_idx.index_select(0, order[:keep_n])

            expert_input = flat.index_select(0, kept)
            expert_output = expert(expert_input)
            weights = top1_prob.index_select(0, kept).unsqueeze(-1).to(dtype=expert_output.dtype)
            routed = (expert_output * weights).to(dtype=output.dtype)
            output.index_add_(0, kept, routed)

        if self.routing_logger is not None:
            self.routing_logger.record(self.layer_idx, gate_probs.detach(), top1_idx.unsqueeze(-1).detach())
        return dense_out + (self.alpha * output.view(original_shape))


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _find_layers_container(model):
    for _, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0 and hasattr(layers[0], "mlp"):
            return module
    raise ValueError("Could not find a transformer layers container with `.layers[*].mlp`.")


def _normalize_moe_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        while new_key.startswith("base_model.model."):
            new_key = new_key[len("base_model.model."):]
        normalized[new_key] = value
    return normalized


def _resolve_moe_state_path(model_spec) -> Optional[Path]:
    model_config = getattr(model_spec, "model_config", {}) or {}
    candidates = []
    explicit = model_config.get("moe_state_path")
    if explicit:
        candidates.append(Path(str(explicit)).expanduser())
    adapter_path = model_config.get("moe_lora_adapter_path")
    if adapter_path:
        ad = Path(str(adapter_path)).expanduser()
        candidates.append(ad.parent / "moe" / "moe_state.pt")
        candidates.append(ad.parent / "moe_state.pt")

    huggingface_id = getattr(model_spec, "huggingface_id", None)
    if huggingface_id:
        hf_path = Path(str(huggingface_id)).expanduser()
        if hf_path.exists():
            candidates.append(hf_path.parent / "moe" / "moe_state.pt")
            candidates.append(hf_path / "moe_state.pt")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _replace_last_mlp_with_moe(model, model_config, routing_logger: Optional[EvalRoutingLogger]):
    container = _find_layers_container(model)
    layers = container.layers
    num_layers_total = len(layers)
    num_moe_layers = _to_int(model_config.get("moe_num_moe_layers", model_config.get("moe_num_layers", 6)), 6)
    num_moe_layers = max(1, min(num_layers_total, num_moe_layers))

    num_experts = _to_int(model_config.get("moe_num_experts", 8), 8)
    top_k = _to_int(model_config.get("moe_top_k", 2), 2)
    expert_hidden = _to_int(
        model_config.get("moe_expert_hidden_size", model_config.get("moe_hidden_size", 2048)),
        2048,
    )
    activation = str(model_config.get("moe_activation", "silu")).lower()

    layer_indices = list(range(max(0, num_layers_total - num_moe_layers), num_layers_total))
    for idx in layer_indices:
        old_mlp = layers[idx].mlp
        hidden_size = None
        device = None
        dtype = None

        down_proj = getattr(old_mlp, "down_proj", None)
        if down_proj is not None and hasattr(down_proj, "out_features"):
            hidden_size = int(down_proj.out_features)
        if down_proj is not None and hasattr(down_proj, "weight"):
            device = down_proj.weight.device
            dtype = down_proj.weight.dtype

        if hidden_size is None:
            hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 4096))
        if device is None:
            try:
                p = next(model.parameters())
                device = p.device
                dtype = p.dtype
            except Exception:
                device = torch.device("cpu")
                dtype = torch.float32

        layers[idx].mlp = SparseMoEFFN(
            hidden_size=hidden_size,
            expert_hidden_size=expert_hidden,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            layer_idx=idx,
            routing_logger=routing_logger,
        ).to(device=device, dtype=dtype)

    return layer_indices


def _replace_last_mlp_with_residual_moe(model, model_config, routing_logger: Optional[EvalRoutingLogger]):
    container = _find_layers_container(model)
    layers = container.layers
    num_layers_total = len(layers)
    num_moe_layers = _to_int(model_config.get("moe_num_moe_layers", model_config.get("moe_num_layers", 6)), 6)
    num_moe_layers = max(1, min(num_layers_total, num_moe_layers))

    num_experts = _to_int(model_config.get("moe_num_experts", 4), 4)
    top_k = _to_int(model_config.get("moe_top_k", 1), 1)
    expert_hidden = _to_int(
        model_config.get("moe_expert_hidden_size", model_config.get("moe_hidden_size", 8192)),
        8192,
    )
    activation = str(model_config.get("moe_activation", "silu")).lower()
    alpha = float(model_config.get("moe_alpha", 0.1))
    capacity_train = float(model_config.get("moe_capacity_factor_train", 1.25))
    capacity_eval = float(model_config.get("moe_capacity_factor_eval", 2.0))

    layer_indices = list(range(max(0, num_layers_total - num_moe_layers), num_layers_total))
    for idx in layer_indices:
        old_mlp = layers[idx].mlp
        hidden_size = None
        device = None
        dtype = None

        down_proj = getattr(old_mlp, "down_proj", None)
        if down_proj is not None and hasattr(down_proj, "out_features"):
            hidden_size = int(down_proj.out_features)
        if down_proj is not None and hasattr(down_proj, "weight"):
            device = down_proj.weight.device
            dtype = down_proj.weight.dtype

        if hidden_size is None:
            hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 4096))
        if device is None:
            try:
                p = next(model.parameters())
                device = p.device
                dtype = p.dtype
            except Exception:
                device = torch.device("cpu")
                dtype = torch.float32

        layers[idx].mlp = ResidualSparseSkillMoEFFN(
            dense_mlp=old_mlp,
            hidden_size=hidden_size,
            expert_hidden_size=expert_hidden,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            alpha=alpha,
            capacity_factor_train=capacity_train,
            capacity_factor_eval=capacity_eval,
            layer_idx=idx,
            routing_logger=routing_logger,
        ).to(device=device, dtype=dtype)

    return layer_indices


def apply_moe_if_requested(model, model_spec, logger=None):
    model_config = getattr(model_spec, "model_config", {}) or {}
    if not bool(model_config.get("moe_enabled", False)):
        return model

    moe_mode = str(model_config.get("moe_mode", "replace")).strip().lower()
    num_experts = _to_int(model_config.get("moe_num_experts", 8), 8)
    eval_log_dir = os.getenv("PLAYPEN_MOE_EVAL_LOG_DIR") or model_config.get("moe_eval_log_dir")
    eval_flush = _to_int(os.getenv("PLAYPEN_MOE_EVAL_FLUSH_EVERY", "8"), 8)
    routing_logger = None
    if eval_log_dir:
        routing_logger = EvalRoutingLogger(
            log_dir=str(eval_log_dir),
            num_experts=num_experts,
            flush_every=eval_flush,
        )

    if moe_mode in {"residual", "residual_skill", "residual-skill"}:
        layer_indices = _replace_last_mlp_with_residual_moe(model, model_config, routing_logger)
    else:
        layer_indices = _replace_last_mlp_with_moe(model, model_config, routing_logger)
    state_path = _resolve_moe_state_path(model_spec)
    if state_path is None:
        if logger is not None:
            try:
                logger.warning(
                    "MoE enabled for %s but no moe_state_path found; using randomly initialized MoE blocks.",
                    getattr(model_spec, "model_name", "model"),
                )
            except Exception:
                pass
        return model

    try:
        try:
            raw_state = torch.load(str(state_path), map_location="cpu", weights_only=True)
        except TypeError:
            raw_state = torch.load(str(state_path), map_location="cpu")

        if isinstance(raw_state, Mapping) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], Mapping):
            raw_state = raw_state["state_dict"]
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"Unexpected MoE state format in {state_path}")

        moe_state = _normalize_moe_state_dict(raw_state)
        incompatible = model.load_state_dict(moe_state, strict=False)
        missing = len(getattr(incompatible, "missing_keys", []))
        unexpected = len(getattr(incompatible, "unexpected_keys", []))
        if logger is not None:
            try:
                logger.info(
                    "Applied MoE runtime for %s (layers=%s, state=%s, missing=%d, unexpected=%d).",
                    getattr(model_spec, "model_name", "model"),
                    ",".join(str(x) for x in layer_indices),
                    str(state_path),
                    missing,
                    unexpected,
                )
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"Failed to apply MoE runtime from {state_path}: {e}") from e

    return model
