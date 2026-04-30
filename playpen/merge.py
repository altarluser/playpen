from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


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


def _dedup_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _candidate_adapter_dirs(adapter_path: str) -> List[Path]:
    raw = Path(str(adapter_path)).expanduser()
    candidates: List[Path] = [raw]
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)
        repo_root = Path(__file__).resolve().parent.parent
        candidates.append(repo_root / raw)

        env_repo = os.getenv("PLAYPEN_REPO_DIR")
        if env_repo:
            candidates.append(Path(env_repo).expanduser() / raw)

        env_adapter_root = os.getenv("PLAYPEN_ADAPTER_ROOT")
        if env_adapter_root:
            adapter_root = Path(env_adapter_root).expanduser()
            normalized = str(raw).replace("\\", "/")
            if normalized.startswith("models/sft+lora/"):
                rel = normalized[len("models/sft+lora/"):]
                rel_path = Path(rel)
                candidates.append(adapter_root / rel_path)
                if rel_path.parts and adapter_root.name == rel_path.parts[0]:
                    candidates.append(adapter_root / Path(*rel_path.parts[1:]))
            else:
                candidates.append(adapter_root / raw)
    return _dedup_paths(candidates)


def _resolve_adapter_dir(adapter_path: str) -> Tuple[Path, List[Path]]:
    candidates = _candidate_adapter_dirs(adapter_path)
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand, candidates
    return candidates[0], candidates


def _load_adapter_state(adapter_path: str):
    import torch

    ap, tried = _resolve_adapter_dir(adapter_path)
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

    tried_text = ", ".join(str(p) for p in tried)
    raise FileNotFoundError(
        f"No adapter weights found in {adapter_path} (resolved={ap}; tried=[{tried_text}]) "
        f"(expected adapter_model.safetensors/.bin or pytorch_model.safetensors/.bin)."
    )


def _load_adapter_scale(adapter_path: str) -> float:
    ap, _ = _resolve_adapter_dir(adapter_path)
    cfg_path = ap / "adapter_config.json"
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

    param_map = {name: p for name, p in base_model.named_parameters() if p is not None}
    if not param_map:
        raise RuntimeError("No trainable parameters found on base model for adapter merge.")
    valid_weight_keys = set(param_map.keys())

    if merge_method == "task_arithmetic":
        with torch.inference_mode():
            for adapter_path, alpha in zip(adapter_paths, merge_weights):
                sd = _load_adapter_state(str(adapter_path))
                scale = _load_adapter_scale(str(adapter_path))
                delta = _adapter_delta_dict(sd, scale, valid_weight_keys)
                a = float(alpha)
                for key, d in delta.items():
                    p = param_map.get(key)
                    if p is None:
                        continue
                    p.data.add_((a * d).to(device=p.device, dtype=p.dtype))
        return base_model

    if merge_method == "weight_averaging":
        weight_sum = float(sum(merge_weights))
        if weight_sum == 0.0:
            raise ValueError("weight_averaging requires sum(merge_weights) != 0.")

        with torch.inference_mode():
            for adapter_path, weight in zip(adapter_paths, merge_weights):
                sd = _load_adapter_state(str(adapter_path))
                scale = _load_adapter_scale(str(adapter_path))
                delta = _adapter_delta_dict(sd, scale, valid_weight_keys)
                coeff = float(weight) / weight_sum
                for key, d in delta.items():
                    p = param_map.get(key)
                    if p is None:
                        continue
                    p.data.add_((coeff * d).to(device=p.device, dtype=p.dtype))
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
                delta = _adapter_delta_dict(sd, scale, valid_weight_keys)

                ww = float(weight)
                if ww != 1.0:
                    for key in list(delta.keys()):
                        delta[key] = delta[key] * ww

                per_adapter.append(delta)
                for key in delta.keys():
                    if not key.startswith(skip_prefixes):
                        all_keys.add(key)

        with torch.inference_mode():
            for key in list(all_keys):
                p = param_map.get(key)
                if p is None:
                    continue
                base_t = p.detach().float().cpu()

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

                update = (float(ties_lambda) * tau_m).to(device=p.device, dtype=p.dtype)
                p.data.add_(update)
        return base_model

    raise ValueError(
        f"Unsupported merge method '{merge_method}'. "
        f"Supported methods: task_arithmetic, weight_averaging, ties."
    )


def _merge_adapters_via_peft(base_model, adapter_paths: Sequence[str], merge_method: str, merge_weights, model_config: Mapping):
    """
    Standard PEFT merge path.
    - task_arithmetic/weight_averaging -> PEFT `linear`
    - ties -> PEFT `ties`
    This avoids materializing full B@A deltas and is typically much more memory-efficient.
    """
    from peft import PeftModel

    resolved_dirs: List[str] = []
    for p in adapter_paths:
        ap, _ = _resolve_adapter_dir(str(p))
        resolved_dirs.append(str(ap))

    if merge_weights is None:
        merge_weights = [1.0] * len(resolved_dirs)
    if len(merge_weights) != len(resolved_dirs):
        raise ValueError("merge_weights length must match number of adapters.")

    if merge_method == "weight_averaging":
        total = float(sum(merge_weights))
        if total == 0.0:
            raise ValueError("weight_averaging requires sum(merge_weights) != 0.")
        if any(float(w) < 0.0 for w in merge_weights):
            raise ValueError("weight_averaging requires non-negative merge_weights.")
        weights = [float(w) / total for w in merge_weights]
        combination_type = "linear"
    elif merge_method == "task_arithmetic":
        # Standard task arithmetic: use raw coefficients directly.
        weights = [float(w) for w in merge_weights]
        combination_type = "linear"
    elif merge_method == "ties":
        weights = [float(w) for w in merge_weights]
        combination_type = "ties"
    else:
        raise ValueError(
            f"Unsupported merge method '{merge_method}'. "
            f"Supported methods: task_arithmetic, weight_averaging, ties."
        )

    peft_model = PeftModel.from_pretrained(base_model, resolved_dirs[0], adapter_name="merge_src_0", is_trainable=False)
    adapter_names = ["merge_src_0"]
    for i, ap in enumerate(resolved_dirs[1:], start=1):
        name = f"merge_src_{i}"
        peft_model.load_adapter(ap, adapter_name=name, is_trainable=False)
        adapter_names.append(name)

    merger = peft_model
    if not hasattr(merger, "add_weighted_adapter"):
        merger = getattr(peft_model, "base_model", peft_model)
    if not hasattr(merger, "add_weighted_adapter"):
        raise RuntimeError("Installed PEFT backend does not expose add_weighted_adapter on the loaded model.")

    merge_name = "__playpen_merged__"
    if combination_type == "ties":
        ties_k = float(model_config.get("ties_k", model_config.get("merge_ties_k", 0.20)))
        if not (0.0 < ties_k <= 1.0):
            raise ValueError("ties_k must be in (0, 1].")
        majority_sign_method = str(
            model_config.get("ties_majority_sign_method", model_config.get("merge_ties_majority_sign_method", "total"))
        )
        merger.add_weighted_adapter(
            adapters=adapter_names,
            weights=weights,
            adapter_name=merge_name,
            combination_type="ties",
            density=ties_k,
            majority_sign_method=majority_sign_method,
        )
    else:
        merger.add_weighted_adapter(
            adapters=adapter_names,
            weights=weights,
            adapter_name=merge_name,
            combination_type="linear",
        )

    if hasattr(peft_model, "set_adapter"):
        peft_model.set_adapter(merge_name)

    try:
        return merger.merge_and_unload(safe_merge=True, adapter_names=[merge_name])
    except TypeError:
        # Compatibility with older signatures without adapter_names.
        return merger.merge_and_unload(safe_merge=True)


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
    if merge_weights is None:
        # Standard defaults:
        # - task_arithmetic: equal coefficients (1.0 each)
        # - weight_averaging: equal weights then normalized to 1/N
        # - ties: equal weights
        merge_weights = [1.0] * len(adapter_models)
    if logger is not None:
        try:
            logger.info("Merging adapters via %s: %s", merge_method, adapter_models)
            logger.info("Merge weights (raw): %s", merge_weights)
            if merge_method == "ties":
                ties_k = float(model_config.get("ties_k", model_config.get("merge_ties_k", 0.20)))
                majority_sign_method = str(
                    model_config.get("ties_majority_sign_method", model_config.get("merge_ties_majority_sign_method", "total"))
                )
                logger.info(
                    "TIES config: density(ties_k)=%.4f, majority_sign_method=%s",
                    ties_k,
                    majority_sign_method,
                )
        except Exception:
            pass

    allow_fallback = str(os.getenv("PLAYPEN_MERGE_ALLOW_FALLBACK", "0")).strip().lower() in {"1", "true", "yes", "on"}
    allow_fallback = bool(model_config.get("merge_allow_fallback", allow_fallback))

    # Manual LoRA-delta merge path for WA/TA by default.
    # This intentionally ignores modules_to_save and only merges LoRA deltas
    # (lora_A/lora_B), which avoids PEFT add_weighted_adapter conflicts.
    prefer_manual_for = str(
        os.getenv("PLAYPEN_MERGE_PREFER_MANUAL", "weight_averaging,task_arithmetic")
    ).strip().lower()
    manual_set = {x.strip() for x in prefer_manual_for.split(",") if x.strip()}
    if merge_method in manual_set:
        if logger is not None:
            try:
                logger.info(
                    "Using manual LoRA-delta merge path for method=%s (modules_to_save ignored).",
                    merge_method,
                )
            except Exception:
                pass
        return _merge_adapters(model, adapter_models, merge_method, merge_weights, model_config)

    # Prefer standard PEFT merge implementations when available.
    try:
        return _merge_adapters_via_peft(model, adapter_models, merge_method, merge_weights, model_config)
    except Exception as e:
        if not allow_fallback:
            raise RuntimeError(
                "PEFT-native merge failed and fallback is disabled. "
                "Set model_config.merge_allow_fallback=true or PLAYPEN_MERGE_ALLOW_FALLBACK=1 to enable fallback."
            ) from e
        if logger is not None:
            try:
                logger.warning("PEFT-native merge failed, falling back to legacy merge path: %s", e)
            except Exception:
                pass
        return _merge_adapters(model, adapter_models, merge_method, merge_weights, model_config)
