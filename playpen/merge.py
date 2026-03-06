from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


def _normalize_merge_method(merge_method: Optional[str]) -> Optional[str]:
    if merge_method is None:
        return None
    merge_method = str(merge_method).strip().lower()
    return merge_method or None


def _get_peft_models(model_spec) -> List[str]:
    model_config = getattr(model_spec, "model_config", {}) or {}
    if "peft_models" in model_config:
        peft_models = model_config["peft_models"]
        if not isinstance(peft_models, list) or not peft_models:
            raise ValueError("'peft_models' must be a non-empty list.")
        return [str(x) for x in peft_models]
    if "peft_model" in model_config:
        return [str(model_config["peft_model"])]
    return []


def _move_state_dict(state, ref_state):
    return {k: v.to(device=ref_state[k].device, dtype=ref_state[k].dtype) for k, v in state.items()}


def _load_safetensors(path: Path):
    try:
        from safetensors.torch import load_file
    except Exception as e:
        raise RuntimeError("Found .safetensors adapter weights but 'safetensors' is not installed.") from e
    return load_file(str(path))


def _load_adapter_state(adapter_path: str):
    import torch

    ap = Path(adapter_path)
    st = ap / "adapter_model.safetensors"
    if st.exists():
        return _load_safetensors(st)

    bn = ap / "adapter_model.bin"
    if bn.exists():
        return torch.load(str(bn), map_location="cpu")

    st2 = ap / "pytorch_model.safetensors"
    if st2.exists():
        return _load_safetensors(st2)

    bn2 = ap / "pytorch_model.bin"
    if bn2.exists():
        return torch.load(str(bn2), map_location="cpu")

    raise FileNotFoundError(
        f"No adapter weights found in {adapter_path} "
        f"(expected adapter_model.safetensors/.bin or pytorch_model.safetensors/.bin)."
    )


def _load_adapter_scale(adapter_path: str) -> float:
    cfg_path = Path(adapter_path) / "adapter_config.json"
    if not cfg_path.exists():
        return 1.0
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        r = float(cfg.get("r", 1.0))
        lora_alpha = float(cfg.get("lora_alpha", 1.0))
        if r == 0:
            return 1.0
        return lora_alpha / r
    except Exception:
        return 1.0


def _strip_prefix(key: str) -> str:
    if key.startswith("base_model.model."):
        return key[len("base_model.model."):]
    return key


def _adapter_delta_dict(sd: Dict, scale: float, base_state_fp32: Dict):
    delta: Dict = {}
    for key_a in list(sd.keys()):
        if ".lora_A." not in key_a or not key_a.endswith(".weight"):
            continue
        key_b = key_a.replace(".lora_A.", ".lora_B.")
        if key_b not in sd:
            continue

        a = sd[key_a].detach().cpu().float()
        b = sd[key_b].detach().cpu().float()

        base_key = _strip_prefix(key_a)
        module_prefix = base_key.split(".lora_A.")[0]
        weight_key = f"{module_prefix}.weight"
        if weight_key not in base_state_fp32:
            continue

        dw = (b @ a) * float(scale)
        if weight_key in delta:
            delta[weight_key] = delta[weight_key] + dw
        else:
            delta[weight_key] = dw
    return delta


def _trim_topk_approx(x, k_frac: float, sample_size: int = 200_000):
    import torch

    flat = x.view(-1)
    n = flat.numel()
    k = int(round(k_frac * n))
    if k <= 0:
        return torch.zeros_like(x)
    if k >= n:
        return x

    abs_flat = flat.abs()
    if n <= sample_size:
        topk_vals = torch.topk(abs_flat, k, largest=True, sorted=False).values
        threshold = topk_vals.min()
        return x * (abs_flat >= threshold).view_as(x)

    idx = torch.randint(0, n, (sample_size,), device=abs_flat.device)
    sample = abs_flat[idx]
    q = 1.0 - float(k_frac)
    threshold = torch.quantile(sample, q)
    return x * (abs_flat >= threshold).view_as(x)


def _merge_adapters(base_model, adapter_paths: Sequence[str], merge_method: str, merge_weights, model_config: Mapping):
    import torch

    if merge_weights is None:
        merge_weights = [1.0] * len(adapter_paths)
    if len(merge_weights) != len(adapter_paths):
        raise ValueError("merge_weights length must match number of adapters.")

    base_state = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    base_state_fp32 = {k: v.float() for k, v in base_state.items()}

    if merge_method == "task_arithmetic":
        merged_state = {k: v.clone() for k, v in base_state_fp32.items()}
        with torch.inference_mode():
            for adapter_path, alpha in zip(adapter_paths, merge_weights):
                sd = _load_adapter_state(str(adapter_path))
                scale = _load_adapter_scale(str(adapter_path))
                delta = _adapter_delta_dict(sd, scale, base_state_fp32)
                a = float(alpha)
                for key, d in delta.items():
                    merged_state[key] = merged_state[key] + (a * d)

        base_model.load_state_dict(_move_state_dict(merged_state, base_model.state_dict()))
        return base_model

    if merge_method == "weight_averaging":
        weight_sum = float(sum(merge_weights))
        if weight_sum == 0.0:
            raise ValueError("weight_averaging requires sum(merge_weights) != 0.")

        merged_state = {k: v.clone() for k, v in base_state_fp32.items()}
        with torch.inference_mode():
            for adapter_path, weight in zip(adapter_paths, merge_weights):
                sd = _load_adapter_state(str(adapter_path))
                scale = _load_adapter_scale(str(adapter_path))
                delta = _adapter_delta_dict(sd, scale, base_state_fp32)
                coeff = float(weight) / weight_sum
                for key, d in delta.items():
                    merged_state[key] = merged_state[key] + (coeff * d)

        base_model.load_state_dict(_move_state_dict(merged_state, base_model.state_dict()))
        return base_model

    if merge_method == "ties":
        ties_k = float(model_config.get("ties_k", model_config.get("merge_ties_k", 0.20)))
        ties_lambda = float(model_config.get("ties_lambda", model_config.get("merge_ties_lambda", 1.0)))
        ties_sample_size = int(model_config.get("ties_sample_size", model_config.get("merge_ties_sample_size", 200_000)))

        if not (0.0 < ties_k <= 1.0):
            raise ValueError("ties_k must be in (0, 1].")
        if ties_lambda < 0.0:
            raise ValueError("ties_lambda must be >= 0.")

        skip_prefixes = ("lm_head.", "model.embed_tokens.", "embed_tokens.")
        per_adapter: List[Dict] = []
        all_keys = set()

        with torch.inference_mode():
            for adapter_path, weight in zip(adapter_paths, merge_weights):
                sd = _load_adapter_state(str(adapter_path))
                scale = _load_adapter_scale(str(adapter_path))
                delta = _adapter_delta_dict(sd, scale, base_state_fp32)

                ww = float(weight)
                if ww != 1.0:
                    for key in list(delta.keys()):
                        delta[key] = delta[key] * ww

                per_adapter.append(delta)
                for key in delta.keys():
                    if not key.startswith(skip_prefixes):
                        all_keys.add(key)

        merged_state = {k: v.clone() for k, v in base_state_fp32.items()}
        with torch.inference_mode():
            for key in list(all_keys):
                base_t = base_state_fp32[key]

                deltas = []
                for dct in per_adapter:
                    d = dct.get(key)
                    if d is None:
                        deltas.append(torch.zeros_like(base_t))
                    else:
                        deltas.append(d)

                trimmed = [_trim_topk_approx(d, ties_k, ties_sample_size) for d in deltas]
                sum_trim = torch.zeros_like(base_t)
                for t in trimmed:
                    sum_trim = sum_trim + t
                gamma = torch.sign(sum_trim)

                stack = torch.stack(trimmed, dim=0)
                signs = torch.sign(stack)
                aligned = (signs == gamma.unsqueeze(0)) & (stack != 0)
                sum_aligned = (stack * aligned).sum(dim=0)
                count = aligned.sum(dim=0).clamp_min(1)
                tau_m = sum_aligned / count

                merged_state[key] = base_t + float(ties_lambda) * tau_m

        base_model.load_state_dict(_move_state_dict(merged_state, base_model.state_dict()))
        return base_model

    raise ValueError(
        f"Unsupported merge method '{merge_method}'. "
        f"Supported methods: task_arithmetic, weight_averaging, ties."
    )


def apply_merge_if_requested(model, model_spec, logger=None):
    model_config = getattr(model_spec, "model_config", {}) or {}
    merge_method = _normalize_merge_method(model_config.get("merge"))
    if not merge_method:
        return model

    if model_config.get("load_in_8bit") or model_config.get("load_in_4bit"):
        raise ValueError("Adapter merging is not supported with 8-bit/4-bit base models.")

    adapter_models = _get_peft_models(model_spec)
    if not adapter_models:
        raise ValueError("Merge requested but no adapters specified in 'peft_model' or 'peft_models'.")

    merge_weights = model_config.get("merge_weights")
    if logger is not None:
        try:
            logger.info("Merging adapters via %s: %s", merge_method, adapter_models)
        except Exception:
            pass
    return _merge_adapters(model, adapter_models, merge_method, merge_weights, model_config)
