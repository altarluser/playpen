"""
Local backend override for `huggingface_local`.
This file is discovered by clemcore's BackendRegistry from cwd before packaged backends.
It adds native adapter merge support used by playpen eval/run.
"""

import torch

import clemcore.backends as backends
import clemcore.backends.huggingface_local_api as core_hf

from playpen.merge import apply_merge_if_requested
from playpen.moe_runtime import apply_moe_if_requested


logger = core_hf.logger
stdout_logger = core_hf.stdout_logger


def load_model(model_spec: backends.ModelSpec):
    model = core_hf.load_model(model_spec)
    model = apply_moe_if_requested(model, model_spec, logger=stdout_logger)
    return apply_merge_if_requested(model, model_spec, logger=stdout_logger)


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


class HuggingfaceLocalModel(core_hf.HuggingfaceLocalModel):
    def __init__(self, model_spec: backends.ModelSpec):
        # Re-implement __init__ to use local load_model() above.
        backends.BatchGenerativeModel.__init__(self, model_spec)
        self.tokenizer, self.config, self.context_size = core_hf.load_config_and_tokenizer(model_spec)
        self.model = load_model(model_spec)

        if not self.model.generation_config.pad_token_id:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.device = _infer_generation_device(self.model)
        stdout_logger.info(f"Generation device for {model_spec.model_name}: {self.device}")


class HuggingfaceLocal(core_hf.HuggingfaceLocal):
    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        torch.set_num_threads(1)
        return HuggingfaceLocalModel(model_spec)
