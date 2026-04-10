from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from clemcore.backends import ModelRegistry, ModelSpec
import clemcore.backends.huggingface_local_api as core_hf

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _resolve_registered_model_spec(model_registry: ModelRegistry, model_name: str) -> ModelSpec:
    spec = ModelSpec.from_dict({"model_name": model_name})
    return model_registry.get_first_model_spec_that_unify_with(spec)


def _model_device(model) -> torch.device:
    try:
        for p in model.parameters():
            if p is not None and not getattr(p, "is_meta", False):
                return p.device
    except Exception:
        pass
    return torch.device("cpu")


def _strip_adapter_fields(spec: ModelSpec) -> ModelSpec:
    d = spec.to_dict()
    cfg = dict(d.get("model_config") or {})
    cfg.pop("peft_model", None)
    cfg.pop("peft_models", None)
    cfg.pop("merge", None)
    cfg.pop("merge_weights", None)
    d["model_config"] = cfg
    return ModelSpec.from_dict(d)


def _adapter_path_for_expert(spec: ModelSpec) -> str:
    cfg = dict(getattr(spec, "model_config", {}) or {})
    path = cfg.get("peft_model")
    if not path:
        raise ValueError(
            f"Expert '{spec.model_name}' does not define model_config.peft_model required for explicit LoRA routing."
        )
    return str(path)


@dataclass
class ExpertScore:
    expert_name: str
    mean_nll: float
    sum_nll: float
    num_target_tokens: int


class LossBasedExpertRouter:
    """
    Whole-game top-1 loss-based router for explicit LoRA experts.
    """

    def __init__(self, model_registry: ModelRegistry, base_model_name: str, experts: Sequence[str]):
        if PeftModel is None:
            raise RuntimeError("Loss-based LoRA routing requires the 'peft' package to be installed.")

        self.model_registry = model_registry
        self.experts = [str(x) for x in experts]
        if not self.experts:
            raise ValueError("No experts provided for loss-based router.")

        base_spec = _resolve_registered_model_spec(model_registry, str(base_model_name))
        base_spec = _strip_adapter_fields(base_spec)
        self.tokenizer, _, _ = core_hf.load_config_and_tokenizer(base_spec)
        self.model = core_hf.load_model(base_spec)
        self.model.eval()
        self.device = _model_device(self.model)
        self._adapter_model = None
        self._loaded_adapters = set()
        self._adapter_key_by_name: Dict[str, str] = {}
        self._prepare_adapters()

    def _prepare_adapters(self) -> None:
        for idx, expert_name in enumerate(self.experts):
            expert_spec = _resolve_registered_model_spec(self.model_registry, expert_name)
            adapter_path = _adapter_path_for_expert(expert_spec)
            adapter_key = f"expert_{idx}"
            self._adapter_key_by_name[expert_name] = adapter_key
            if self._adapter_model is None:
                self._adapter_model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path,
                    adapter_name=adapter_key,
                    is_trainable=False,
                )
                self._adapter_model.eval()
                self._adapter_model.to(self.device)
                self._loaded_adapters.add(adapter_key)
            else:
                self._adapter_model.load_adapter(adapter_path, adapter_name=adapter_key, is_trainable=False)
                self._loaded_adapters.add(adapter_key)

        if self._adapter_model is None:
            raise RuntimeError("No adapters were loaded for loss-based routing.")

    def _max_context_len(self) -> int:
        cfg_max = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        tok_max = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
        candidates = [x for x in (cfg_max, tok_max) if x and x < 10_000_000]
        return max(candidates) if candidates else 2048

    @torch.no_grad()
    def score_expert_nll(
        self,
        expert_name: str,
        prompt_text: str,
        target_text: str,
    ) -> Dict[str, object]:
        if expert_name not in self._adapter_key_by_name:
            raise ValueError(f"Unknown expert '{expert_name}' for loss router.")

        adapter_key = self._adapter_key_by_name[expert_name]
        self._adapter_model.set_adapter(adapter_key)

        prompt_ids = self.tokenizer(
            prompt_text or "",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )["input_ids"][0]
        target_ids = self.tokenizer(
            target_text or "",
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )["input_ids"][0]

        if target_ids.numel() == 0:
            return {
                "expert_name": expert_name,
                "mean_nll": float("inf"),
                "sum_nll": float("inf"),
                "num_target_tokens": 0,
            }

        max_len = self._max_context_len()
        if target_ids.numel() >= max_len:
            target_ids = target_ids[: max_len - 1]
            prompt_ids = prompt_ids.new_empty((0,), dtype=prompt_ids.dtype)
        else:
            keep_prompt = max_len - int(target_ids.numel())
            if prompt_ids.numel() > keep_prompt:
                prompt_ids = prompt_ids[-keep_prompt:]

        input_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[:, prompt_ids.numel() :] = input_ids[:, prompt_ids.numel() :]

        outputs = self._adapter_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mean_nll = float(outputs.loss.detach().float().item())
        num_target_tokens = int(target_ids.numel())
        sum_nll = float(mean_nll * num_target_tokens)
        return {
            "expert_name": expert_name,
            "mean_nll": mean_nll,
            "sum_nll": sum_nll,
            "num_target_tokens": num_target_tokens,
        }

    @torch.no_grad()
    def select_expert_by_loss(
        self,
        prompt_text: str,
        target_text: str,
    ) -> Dict[str, object]:
        scores = [self.score_expert_nll(expert, prompt_text, target_text) for expert in self.experts]
        scores.sort(key=lambda x: float(x["mean_nll"]))
        best = scores[0]
        second = scores[1] if len(scores) > 1 else None
        margin = float(second["mean_nll"] - best["mean_nll"]) if second is not None else float("nan")
        compact = [{"expert_name": s["expert_name"], "mean_nll": float(s["mean_nll"])} for s in scores]
        return {
            "selected_expert": str(best["expert_name"]),
            "expert_scores": compact,
            "margin_to_second_best": margin,
            "details": scores,
        }


def extract_prompt_and_first_target(example: Mapping[str, object]) -> Tuple[str, str]:
    messages = example.get("messages") or example.get("chat") or []
    if not isinstance(messages, list):
        messages = []

    normalized: List[Tuple[str, str]] = []
    for m in messages:
        if isinstance(m, Mapping):
            role = str(m.get("role", "") or "").strip().lower()
            content = str(m.get("content", "") or "")
            if role and content:
                normalized.append((role, content))

    if normalized:
        prompt_lines: List[str] = []
        target_text = ""
        for role, content in normalized:
            if role == "assistant":
                target_text = content
                break
            prompt_lines.append(f"{role}: {content}")
        if target_text:
            return "\n\n".join(prompt_lines).strip(), target_text.strip()

    prompt = str(example.get("prompt") or example.get("context") or "")
    target = str(
        example.get("target")
        or example.get("response")
        or example.get("first_assistant")
        or example.get("gold_response")
        or ""
    )
    return prompt.strip(), target.strip()


def append_router_log_row(path: Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def entropy_from_counts(counts: Iterable[int]) -> float:
    vals = [int(x) for x in counts if int(x) > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in vals:
        p = c / total
        h -= p * math.log(p)
    return float(h)


@torch.no_grad()
def score_expert_nll(
    model,
    tokenizer,
    expert_name: str,
    prompt_text: str,
    target_text: str,
) -> Dict[str, object]:
    """
    Generic teacher-forced scoring helper using mean token NLL over target tokens.
    If `model` is a PEFT model with adapters, this function activates `expert_name` first.
    """
    if hasattr(model, "set_adapter"):
        try:
            model.set_adapter(str(expert_name))
        except Exception:
            pass

    device = _model_device(model)
    prompt_ids = tokenizer(
        prompt_text or "",
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )["input_ids"][0]
    target_ids = tokenizer(
        target_text or "",
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
    )["input_ids"][0]
    if target_ids.numel() == 0:
        return {
            "expert_name": str(expert_name),
            "mean_nll": float("inf"),
            "sum_nll": float("inf"),
            "num_target_tokens": 0,
        }

    max_len = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if max_len <= 0:
        max_len = int(getattr(tokenizer, "model_max_length", 0) or 2048)
    if max_len >= 10_000_000:
        max_len = 2048

    if target_ids.numel() >= max_len:
        target_ids = target_ids[: max_len - 1]
        prompt_ids = prompt_ids.new_empty((0,), dtype=prompt_ids.dtype)
    else:
        keep_prompt = max_len - int(target_ids.numel())
        if prompt_ids.numel() > keep_prompt:
            prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[:, prompt_ids.numel() :] = input_ids[:, prompt_ids.numel() :]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    mean_nll = float(outputs.loss.detach().float().item())
    num_target_tokens = int(target_ids.numel())
    sum_nll = float(mean_nll * num_target_tokens)
    return {
        "expert_name": str(expert_name),
        "mean_nll": mean_nll,
        "sum_nll": sum_nll,
        "num_target_tokens": num_target_tokens,
    }


def select_expert_by_loss(
    experts: Sequence[str],
    prompt_text: str,
    target_text: str,
    *,
    router: LossBasedExpertRouter,
) -> Dict[str, object]:
    """
    Convenience function that scores experts and returns whole-game top-1 selection.
    """
    scores = [router.score_expert_nll(expert, prompt_text, target_text) for expert in experts]
    scores.sort(key=lambda x: float(x["mean_nll"]))
    best = scores[0]
    second = scores[1] if len(scores) > 1 else None
    margin = float(second["mean_nll"] - best["mean_nll"]) if second is not None else float("nan")
    return {
        "selected_expert": str(best["expert_name"]),
        "expert_scores": [{"expert_name": s["expert_name"], "mean_nll": float(s["mean_nll"])} for s in scores],
        "margin_to_second_best": margin,
        "details": scores,
    }
