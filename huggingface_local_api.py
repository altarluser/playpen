"""
Local backend override for `huggingface_local`.
This file is discovered by clemcore's BackendRegistry from cwd before packaged backends.
It adds native adapter merge support used by playpen eval/run.
"""

import gc
import os
import signal
import time
from pathlib import Path

import torch

import clemcore.backends as backends
import clemcore.backends.huggingface_local_api as core_hf

from playpen.merge import apply_merge_if_requested
from playpen.moe_runtime import apply_moe_if_requested


logger = core_hf.logger
stdout_logger = core_hf.stdout_logger


def _resolve_playpen_adapter_path(path_value: str) -> str:
    raw = str(path_value).strip()
    if not raw:
        return raw

    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)

    normalized = raw.replace("\\", "/")
    if not normalized.startswith("models/sft+lora/"):
        return raw

    adapter_root = os.getenv("PLAYPEN_ADAPTER_ROOT", "").strip()
    if not adapter_root:
        return raw
    root = Path(adapter_root).expanduser()
    rel = Path(normalized[len("models/sft+lora/"):])

    candidates = [root / rel]
    if rel.parts and root.name == rel.parts[0]:
        candidates.append(root / Path(*rel.parts[1:]))

    for cand in candidates:
        if cand.exists():
            return str(cand)
    return str(candidates[-1])


def _resolve_model_spec_adapter_paths(model_spec: backends.ModelSpec) -> backends.ModelSpec:
    spec_dict = model_spec.to_dict()
    model_config = dict(spec_dict.get("model_config") or {})
    changed = False

    if "peft_model" in model_config and model_config.get("peft_model") is not None:
        resolved = _resolve_playpen_adapter_path(str(model_config["peft_model"]))
        if resolved != str(model_config["peft_model"]):
            model_config["peft_model"] = resolved
            changed = True

    if "peft_models" in model_config and isinstance(model_config.get("peft_models"), list):
        resolved_list = []
        for item in model_config["peft_models"]:
            resolved_list.append(_resolve_playpen_adapter_path(str(item)))
        if resolved_list != model_config["peft_models"]:
            model_config["peft_models"] = resolved_list
            changed = True

    if not changed:
        return model_spec

    spec_dict["model_config"] = model_config
    return backends.ModelSpec.from_dict(spec_dict)


def _model_uses_cpu_or_disk_offload(model) -> bool:
    hf_device_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_device_map, dict):
        return False
    for dev in hf_device_map.values():
        if isinstance(dev, str) and dev in {"cpu", "disk"}:
            return True
    return False


def _infer_generation_device(model) -> str:
    # Prefer a real (non-meta) parameter device when available.
    try:
        for p in model.parameters():
            if p is None or getattr(p, "is_meta", False):
                continue
            return str(p.device)
    except Exception:
        pass

    # Fall back to HF device map if present.
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for dev in hf_device_map.values():
            if isinstance(dev, int):
                return f"cuda:{dev}"
            if isinstance(dev, str) and dev not in {"disk", "cpu"}:
                return dev
        if any(dev == "cpu" for dev in hf_device_map.values()):
            return "cpu"

    # Last-resort hardware-based fallback.
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_bool_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str, default: str) -> int:
    raw = str(os.getenv(name, default)).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _assert_cuda_only(model, model_spec: backends.ModelSpec):
    if not _parse_bool_env("PLAYPEN_REQUIRE_CUDA", "0"):
        return
    device = _infer_generation_device(model)
    has_cpu_or_disk_offload = _model_uses_cpu_or_disk_offload(model)
    if device.startswith("cuda") and not has_cpu_or_disk_offload:
        return
    hf_device_map = getattr(model, "hf_device_map", None)
    raise RuntimeError(
        "CUDA-only loading is enabled (PLAYPEN_REQUIRE_CUDA=1) but model was not fully placed on CUDA. "
        f"model={model_spec.model_name}, inferred_device={device}, hf_device_map={hf_device_map}"
    )


def load_model(model_spec: backends.ModelSpec):
    resolved_spec = _resolve_model_spec_adapter_paths(model_spec)
    retries = max(0, _parse_int_env("PLAYPEN_CUDA_LOAD_RETRIES", "1"))
    attempt = 0
    while True:
        model = core_hf.load_model(resolved_spec)
        model = apply_moe_if_requested(model, resolved_spec, logger=stdout_logger)
        model = apply_merge_if_requested(model, resolved_spec, logger=stdout_logger)
        try:
            _assert_cuda_only(model, resolved_spec)
            return model
        except Exception:
            if attempt >= retries:
                raise
            attempt += 1
            stdout_logger.warning(
                "Model load landed on CPU/disk offload. Retrying CUDA load "
                f"{attempt}/{retries} for {resolved_spec.model_name}."
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)


class HuggingfaceLocalModel(core_hf.HuggingfaceLocalModel):
    def __init__(self, model_spec: backends.ModelSpec):
        # Re-implement __init__ to use local load_model() above.
        backends.BatchGenerativeModel.__init__(self, model_spec)
        loaded = core_hf.load_config_and_tokenizer(model_spec)
        if not isinstance(loaded, tuple):
            raise TypeError(
                "Unexpected return type from clemcore.backends.huggingface_local_api.load_config_and_tokenizer: "
                f"{type(loaded).__name__}"
            )
        if len(loaded) == 3:
            self.tokenizer, self.config, self.context_size = loaded
        elif len(loaded) == 2:
            self.tokenizer, self.config = loaded
            self.context_size = getattr(self.config, "max_position_embeddings", None)
        else:
            raise ValueError(
                "Unexpected return arity from load_config_and_tokenizer: "
                f"{len(loaded)} (expected 2 or 3)."
            )
        # clemcore compatibility: some versions use `chat_template_kwargs`,
        # others use private `_chat_template_kwargs`.
        if not hasattr(self, "chat_template_kwargs"):
            self.chat_template_kwargs = {}
        if not hasattr(self, "_chat_template_kwargs"):
            self._chat_template_kwargs = dict(getattr(self, "chat_template_kwargs", {}) or {})
        else:
            # Keep both names in sync for whichever one downstream code reads.
            self.chat_template_kwargs = dict(getattr(self, "_chat_template_kwargs", {}) or {})
        self.model = load_model(model_spec)

        if not self.model.generation_config.pad_token_id:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.device = _infer_generation_device(self.model)
        stdout_logger.info(f"Generation device for {model_spec.model_name}: {self.device}")

    def generate_response(self, messages, *args, **kwargs):
        timeout_s = _parse_int_env("PLAYPEN_GENERATION_TIMEOUT_SECONDS", "0")
        if timeout_s <= 0:
            return super().generate_response(messages, *args, **kwargs)

        if not hasattr(signal, "SIGALRM"):
            # Non-POSIX fallback: no hard timer available.
            return super().generate_response(messages, *args, **kwargs)

        def _handle_timeout(_signum, _frame):
            raise TimeoutError(
                f"Generation timed out after {timeout_s}s for model {self.model_spec.model_name}"
            )

        prev_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(timeout_s)
        try:
            return super().generate_response(messages, *args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)


class HuggingfaceLocal(core_hf.HuggingfaceLocal):
    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        torch.set_num_threads(1)
        return HuggingfaceLocalModel(model_spec)
