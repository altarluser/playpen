from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import gc
import math
import multiprocessing as mp
import os
import queue
import time
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
    raw = Path(str(path)).expanduser()
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
                rel = Path(normalized[len("models/sft+lora/"):])
                candidates.append(adapter_root / rel)
                if rel.parts and adapter_root.name == rel.parts[0]:
                    candidates.append(adapter_root / Path(*rel.parts[1:]))
            else:
                candidates.append(adapter_root / raw)

    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        cfg_path = cand / "adapter_config.json"
        if cand.exists() and cand.is_dir() and cfg_path.exists():
            return str(cand)

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Expert '{spec.model_name}' adapter path not found for peft_model={path}. "
        f"Tried: [{tried}]"
    )


@dataclass
class ExpertScore:
    expert_name: str
    mean_nll: float
    sum_nll: float
    num_target_tokens: int


@dataclass
class ExpertRuntime:
    expert_name: str
    model: object
    device: torch.device
    adapter_key: str
    max_context_len: int


def _max_context_len_from(model, tokenizer) -> int:
    cfg_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    tok_max = int(getattr(tokenizer, "model_max_length", 0) or 0)
    candidates = [x for x in (cfg_max, tok_max) if x and x < 10_000_000]
    return max(candidates) if candidates else 2048


@torch.no_grad()
def _worker_score_nll(model, tokenizer, device: torch.device, expert_name: str, prompt_text: str, target_text: str):
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

    max_len = _max_context_len_from(model, tokenizer)
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
    return {
        "expert_name": str(expert_name),
        "mean_nll": mean_nll,
        "sum_nll": float(mean_nll * num_target_tokens),
        "num_target_tokens": num_target_tokens,
    }


@torch.no_grad()
def _worker_generate(model, tokenizer, device: torch.device, prompt_text: str, max_new_tokens: int, temperature: float) -> str:
    prompt_ids = tokenizer(
        prompt_text or "",
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )["input_ids"].to(device)
    max_len = _max_context_len_from(model, tokenizer)
    if prompt_ids.shape[1] >= max_len:
        prompt_ids = prompt_ids[:, -max_len:]
    do_sample = float(temperature) > 0.0
    gen_kwargs = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "do_sample": do_sample,
        "temperature": float(temperature) if do_sample else None,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    out_ids = model.generate(prompt_ids, **gen_kwargs)
    gen_ids = out_ids[:, prompt_ids.shape[1] :]
    if gen_ids.numel() == 0:
        return ""
    return str(tokenizer.decode(gen_ids[0], skip_special_tokens=True)).strip()


def _isolated_worker_main(
    expert_name: str,
    base_model_name: str,
    adapter_path: str,
    visible_gpu: str,
    in_q,
    out_q,
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        model_registry = ModelRegistry.from_packaged_and_cwd_files()
        base_spec = _resolve_registered_model_spec(model_registry, str(base_model_name))
        base_spec = _strip_adapter_fields(base_spec)
        loaded = core_hf.load_config_and_tokenizer(base_spec)
        tokenizer = loaded[0] if isinstance(loaded, tuple) and len(loaded) >= 1 else loaded
        model = core_hf.load_model(base_spec)
        model.eval()
        peft_model = PeftModel.from_pretrained(model, adapter_path, adapter_name="active", is_trainable=False)
        peft_model.eval()
        device = _model_device(peft_model)
        out_q.put({"kind": "ready", "expert_name": expert_name, "device": str(device)})

        while True:
            msg = in_q.get()
            if msg is None or msg.get("op") == "shutdown":
                break
            req_id = int(msg["req_id"])
            op = str(msg.get("op", "score"))
            prompt_text = str(msg.get("prompt_text", "") or "")
            target_text = str(msg.get("target_text", "") or "")
            if op == "score":
                row = _worker_score_nll(peft_model, tokenizer, device, expert_name, prompt_text, target_text)
                out_q.put({"kind": "result", "req_id": req_id, "expert_name": expert_name, "row": row})
            elif op == "gen_score":
                candidate = _worker_generate(
                    peft_model,
                    tokenizer,
                    device,
                    prompt_text,
                    int(msg.get("max_new_tokens", 64) or 64),
                    float(msg.get("temperature", 0.0) or 0.0),
                )
                row = _worker_score_nll(peft_model, tokenizer, device, expert_name, prompt_text, candidate)
                row["candidate_text"] = candidate
                out_q.put({"kind": "result", "req_id": req_id, "expert_name": expert_name, "row": row})
            else:
                out_q.put(
                    {
                        "kind": "error",
                        "req_id": req_id,
                        "expert_name": expert_name,
                        "error": f"Unsupported op: {op}",
                    }
                )
    except Exception as e:
        out_q.put({"kind": "fatal", "expert_name": expert_name, "error": str(e)})


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
        loaded = core_hf.load_config_and_tokenizer(base_spec)
        if not isinstance(loaded, tuple):
            raise TypeError(
                "Unexpected return type from clemcore.backends.huggingface_local_api.load_config_and_tokenizer: "
                f"{type(loaded).__name__}"
            )
        if len(loaded) >= 1:
            self.tokenizer = loaded[0]
        else:
            raise ValueError(
                "Unexpected return arity from load_config_and_tokenizer: "
                f"{len(loaded)} (expected >=1)."
            )
        self._isolated_workers_enabled = False
        self._isolated_in_queues: Dict[str, object] = {}
        self._isolated_out_queue = None
        self._isolated_procs: Dict[str, object] = {}
        self._req_counter = 0

        use_isolated = str(os.getenv("PLAYPEN_LOSS_ROUTER_ISOLATED_WORKERS", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_isolated:
            devices = self._parse_parallel_devices()
            if len(devices) >= len(self.experts):
                if self._try_start_isolated_workers(base_model_name=str(base_model_name), devices=devices[: len(self.experts)]):
                    self._isolated_workers_enabled = True
                    print(
                        f"[MoE-LossRouter] isolated workers enabled: experts={len(self.experts)} devices={devices[:len(self.experts)]}",
                        flush=True,
                    )
                    return

        retries = max(0, int(os.getenv("PLAYPEN_CUDA_LOAD_RETRIES", "2")))
        allow_noncuda = str(os.getenv("PLAYPEN_LOSS_ROUTER_ALLOW_NONCUDA", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        attempt = 0
        self.model = None
        while True:
            self.model = core_hf.load_model(base_spec)
            self.model.eval()
            hf_device_map = getattr(self.model, "hf_device_map", None)
            has_cpu_or_disk = isinstance(hf_device_map, dict) and any(
                isinstance(dev, str) and dev in {"cpu", "disk"} for dev in hf_device_map.values()
            )
            self.device = _model_device(self.model)
            on_cuda = str(self.device).startswith("cuda")
            if on_cuda and not has_cpu_or_disk:
                break
            if allow_noncuda and not has_cpu_or_disk:
                print(
                    f"[MoE-LossRouter] non-CUDA override enabled (device={self.device})",
                    flush=True,
                )
                break
            if attempt >= retries:
                raise RuntimeError(
                    "Loss-router base model could not be placed fully on CUDA after retries. "
                    f"attempts={attempt + 1}, device={self.device}, hf_device_map={hf_device_map}"
                )
            attempt += 1
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)
        self._adapter_model = None
        self._loaded_adapters = set()
        self._adapter_key_by_name: Dict[str, str] = {}
        self._expert_runtimes: Dict[str, ExpertRuntime] = {}
        self._parallel_enabled = False
        self._parallel_worker_count = 0
        self._prepare_parallel_or_single(base_spec)

    def _try_start_isolated_workers(self, base_model_name: str, devices: Sequence[str]) -> bool:
        try:
            ctx = mp.get_context("spawn")
            out_q = ctx.Queue()
            self._isolated_out_queue = out_q
            for idx, expert_name in enumerate(self.experts):
                expert_spec = _resolve_registered_model_spec(self.model_registry, expert_name)
                adapter_path = _adapter_path_for_expert(expert_spec)
                in_q = ctx.Queue()
                dev = str(devices[idx])
                vis = dev.split(":", 1)[1] if ":" in dev else dev
                p = ctx.Process(
                    target=_isolated_worker_main,
                    args=(expert_name, base_model_name, adapter_path, vis, in_q, out_q),
                    daemon=True,
                )
                p.start()
                self._isolated_in_queues[expert_name] = in_q
                self._isolated_procs[expert_name] = p

            ready_needed = set(self.experts)
            t0 = time.time()
            while ready_needed and (time.time() - t0) < 600:
                msg = out_q.get(timeout=30)
                kind = str(msg.get("kind"))
                ex = str(msg.get("expert_name"))
                if kind == "ready" and ex in ready_needed:
                    ready_needed.remove(ex)
                elif kind in {"fatal", "error"}:
                    raise RuntimeError(f"isolated worker {ex} failed during startup: {msg.get('error')}")
            if ready_needed:
                raise RuntimeError(f"isolated workers did not become ready: {sorted(ready_needed)}")
            return True
        except Exception as e:
            print(f"[MoE-LossRouter] isolated worker setup failed: {e}", flush=True)
            try:
                self._shutdown_isolated_workers()
            except Exception:
                pass
            return False

    def _shutdown_isolated_workers(self) -> None:
        for q in self._isolated_in_queues.values():
            try:
                q.put({"op": "shutdown"})
            except Exception:
                pass
        for p in self._isolated_procs.values():
            try:
                p.join(timeout=5)
            except Exception:
                pass
        self._isolated_in_queues.clear()
        self._isolated_procs.clear()

    def __del__(self):
        if self._isolated_workers_enabled:
            try:
                self._shutdown_isolated_workers()
            except Exception:
                pass

    def _isolated_request_all(self, op: str, prompt_text: str, target_text: str = "", *, max_new_tokens: int = 64, temperature: float = 0.0):
        self._req_counter += 1
        req_id = int(self._req_counter)
        for expert_name in self.experts:
            self._isolated_in_queues[expert_name].put(
                {
                    "req_id": req_id,
                    "op": str(op),
                    "prompt_text": str(prompt_text or ""),
                    "target_text": str(target_text or ""),
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": float(temperature),
                }
            )

        remaining = set(self.experts)
        rows: List[Dict[str, object]] = []
        t0 = time.time()
        while remaining:
            if (time.time() - t0) > 1800:
                raise TimeoutError(f"Timed out waiting for isolated worker responses: remaining={sorted(remaining)}")
            msg = self._isolated_out_queue.get(timeout=60)
            kind = str(msg.get("kind"))
            if kind in {"fatal", "error"}:
                raise RuntimeError(f"isolated worker error: {msg}")
            if kind != "result":
                continue
            if int(msg.get("req_id", -1)) != req_id:
                continue
            ex = str(msg.get("expert_name"))
            if ex in remaining:
                remaining.remove(ex)
                rows.append(dict(msg.get("row") or {}))
        return rows

    def _parse_parallel_devices(self) -> List[str]:
        raw = str(os.getenv("PLAYPEN_LOSS_ROUTER_DEVICES", "")).strip()
        if raw:
            out = []
            for x in raw.split(","):
                v = x.strip()
                if not v:
                    continue
                if v.startswith("cuda:"):
                    out.append(v)
                elif v.isdigit():
                    out.append(f"cuda:{v}")
            return out
        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)]

    def _prepare_parallel_or_single(self, base_spec: ModelSpec) -> None:
        parallel_flag = str(os.getenv("PLAYPEN_LOSS_ROUTER_PARALLEL", "0")).strip().lower() in {"1", "true", "yes", "on"}
        devices = self._parse_parallel_devices()
        if parallel_flag and len(devices) >= len(self.experts):
            try:
                self._prepare_parallel_runtimes(base_spec, devices[: len(self.experts)])
                self._parallel_enabled = True
                self._parallel_worker_count = len(self._expert_runtimes)
                print(
                    f"[MoE-LossRouter] parallel scoring enabled: experts={len(self.experts)} devices={devices[:len(self.experts)]}",
                    flush=True,
                )
                return
            except Exception as e:
                print(f"[MoE-LossRouter] parallel setup failed, falling back to single-model path: {e}", flush=True)
                self._expert_runtimes.clear()

        self._prepare_adapters()

    def _prepare_parallel_runtimes(self, base_spec: ModelSpec, devices: Sequence[str]) -> None:
        retries = max(0, int(os.getenv("PLAYPEN_CUDA_LOAD_RETRIES", "2")))
        for idx, expert_name in enumerate(self.experts):
            target_device = str(devices[idx])
            expert_spec = _resolve_registered_model_spec(self.model_registry, expert_name)
            adapter_path = _adapter_path_for_expert(expert_spec)
            adapter_key = "active"

            attempt = 0
            model = None
            while True:
                model = core_hf.load_model(base_spec)
                model.eval()
                hf_device_map = getattr(model, "hf_device_map", None)
                has_cpu_or_disk = isinstance(hf_device_map, dict) and any(
                    isinstance(dev, str) and dev in {"cpu", "disk"} for dev in hf_device_map.values()
                )
                cuda_devices_in_map = set()
                if isinstance(hf_device_map, dict):
                    for dev in hf_device_map.values():
                        if isinstance(dev, int):
                            cuda_devices_in_map.add(f"cuda:{dev}")
                        elif isinstance(dev, str) and dev.startswith("cuda"):
                            cuda_devices_in_map.add(dev)
                dev = _model_device(model)
                if str(dev).startswith("cuda") and not has_cpu_or_disk:
                    # Require non-sharded single-device placement per expert runtime.
                    if len(cuda_devices_in_map) > 1:
                        raise RuntimeError(
                            f"Expert {expert_name}: model is sharded across devices {sorted(cuda_devices_in_map)}; "
                            "parallel-per-expert mode requires one device per runtime."
                        )
                    break
                if attempt >= retries:
                    raise RuntimeError(
                        f"Expert {expert_name}: base model not fully on CUDA after retries. "
                        f"device={dev}, hf_device_map={hf_device_map}"
                    )
                attempt += 1
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(2)

            if str(_model_device(model)) != target_device:
                model = model.to(target_device)

            peft_model = PeftModel.from_pretrained(
                model,
                adapter_path,
                adapter_name=adapter_key,
                is_trainable=False,
            )
            peft_model.eval().to(target_device)
            max_len = self._max_context_len_for_model(peft_model)
            self._expert_runtimes[expert_name] = ExpertRuntime(
                expert_name=expert_name,
                model=peft_model,
                device=torch.device(target_device),
                adapter_key=adapter_key,
                max_context_len=max_len,
            )

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

    def _max_context_len_for_model(self, model) -> int:
        cfg_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)
        tok_max = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
        candidates = [x for x in (cfg_max, tok_max) if x and x < 10_000_000]
        return max(candidates) if candidates else 2048

    @torch.no_grad()
    def _score_with_runtime(self, rt: ExpertRuntime, prompt_text: str, target_text: str) -> Dict[str, object]:
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
                "expert_name": rt.expert_name,
                "mean_nll": float("inf"),
                "sum_nll": float("inf"),
                "num_target_tokens": 0,
            }

        max_len = int(rt.max_context_len)
        if target_ids.numel() >= max_len:
            target_ids = target_ids[: max_len - 1]
            prompt_ids = prompt_ids.new_empty((0,), dtype=prompt_ids.dtype)
        else:
            keep_prompt = max_len - int(target_ids.numel())
            if prompt_ids.numel() > keep_prompt:
                prompt_ids = prompt_ids[-keep_prompt:]

        input_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0).to(rt.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[:, prompt_ids.numel() :] = input_ids[:, prompt_ids.numel() :]

        outputs = rt.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mean_nll = float(outputs.loss.detach().float().item())
        num_target_tokens = int(target_ids.numel())
        sum_nll = float(mean_nll * num_target_tokens)
        return {
            "expert_name": rt.expert_name,
            "mean_nll": mean_nll,
            "sum_nll": sum_nll,
            "num_target_tokens": num_target_tokens,
        }

    @torch.no_grad()
    def _generate_with_runtime(
        self,
        rt: ExpertRuntime,
        prompt_text: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        prompt_ids = self.tokenizer(
            prompt_text or "",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )["input_ids"].to(rt.device)

        max_len = int(rt.max_context_len)
        if prompt_ids.shape[1] >= max_len:
            prompt_ids = prompt_ids[:, -max_len:]

        do_sample = float(temperature) > 0.0
        gen_kwargs = {
            "max_new_tokens": max(1, int(max_new_tokens)),
            "do_sample": do_sample,
            "temperature": float(temperature) if do_sample else None,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        out_ids = rt.model.generate(prompt_ids, **gen_kwargs)
        gen_ids = out_ids[:, prompt_ids.shape[1] :]
        if gen_ids.numel() == 0:
            return ""
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return str(text).strip()

    @torch.no_grad()
    def generate_initial_answer(
        self,
        expert_name: str,
        prompt_text: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        if self._isolated_workers_enabled:
            rows = self._isolated_request_all(
                "gen_score",
                prompt_text,
                "",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for r in rows:
                if str(r.get("expert_name")) == str(expert_name):
                    return str(r.get("candidate_text") or "")
            return ""
        if expert_name not in self._adapter_key_by_name:
            raise ValueError(f"Unknown expert '{expert_name}' for loss router.")

        adapter_key = self._adapter_key_by_name[expert_name]
        self._adapter_model.set_adapter(adapter_key)

        prompt_ids = self.tokenizer(
            prompt_text or "",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )["input_ids"].to(self.device)

        max_len = self._max_context_len()
        if prompt_ids.shape[1] >= max_len:
            prompt_ids = prompt_ids[:, -max_len:]

        do_sample = float(temperature) > 0.0
        gen_kwargs = {
            "max_new_tokens": max(1, int(max_new_tokens)),
            "do_sample": do_sample,
            "temperature": float(temperature) if do_sample else None,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        out_ids = self._adapter_model.generate(prompt_ids, **gen_kwargs)
        gen_ids = out_ids[:, prompt_ids.shape[1] :]
        if gen_ids.numel() == 0:
            return ""
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return str(text).strip()

    @torch.no_grad()
    def score_expert_nll(
        self,
        expert_name: str,
        prompt_text: str,
        target_text: str,
    ) -> Dict[str, object]:
        if self._isolated_workers_enabled:
            rows = self._isolated_request_all("score", prompt_text, target_text)
            for r in rows:
                if str(r.get("expert_name")) == str(expert_name):
                    return r
            raise RuntimeError(f"isolated worker did not return score for expert={expert_name}")
        if self._parallel_enabled and expert_name in self._expert_runtimes:
            return self._score_with_runtime(self._expert_runtimes[expert_name], prompt_text, target_text)
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
        if self._isolated_workers_enabled:
            scores = self._isolated_request_all("score", prompt_text, target_text)
        elif self._parallel_enabled:
            with ThreadPoolExecutor(max_workers=self._parallel_worker_count) as ex:
                futs = [
                    ex.submit(self.score_expert_nll, expert, prompt_text, target_text)
                    for expert in self.experts
                ]
                scores = [f.result() for f in futs]
        else:
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

    @torch.no_grad()
    def select_expert_from_prompt(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Dict[str, object]:
        scores = []
        if self._isolated_workers_enabled:
            scores = self._isolated_request_all(
                "gen_score",
                prompt_text,
                "",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        elif self._parallel_enabled:
            def _job(expert_name: str):
                rt = self._expert_runtimes[expert_name]
                candidate = self._generate_with_runtime(
                    rt,
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                scored = self._score_with_runtime(rt, prompt_text, candidate)
                scored["candidate_text"] = candidate
                return scored

            with ThreadPoolExecutor(max_workers=self._parallel_worker_count) as ex:
                futs = [ex.submit(_job, expert) for expert in self.experts]
                scores = [f.result() for f in futs]
        else:
            for expert in self.experts:
                candidate = self.generate_initial_answer(
                    expert,
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                scored = self.score_expert_nll(expert, prompt_text, candidate)
                scored["candidate_text"] = candidate
                scores.append(scored)
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

    prompt = str(
        example.get("prompt")
        or example.get("context")
        or example.get("instruction")
        or example.get("user_prompt")
        or example.get("input")
        or ""
    )
    target = str(
        example.get("target")
        or example.get("response")
        or example.get("first_assistant")
        or example.get("gold_response")
        or example.get("expected_response")
        or example.get("answer")
        or example.get("reference")
        or example.get("solution")
        or example.get("target_word")
        or example.get("target_label")
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
