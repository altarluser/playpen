from __future__ import annotations

import json
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset, load_from_disk

from clemcore.backends import ModelRegistry, ModelSpec
import clemcore.backends.huggingface_local_api as core_hf

from playpen.adapter_bar_sequence_router import SequenceAdapterRouter

try:
    import yaml
except Exception:
    yaml = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


DEFAULT_GAME_TO_CLUSTER = {
    # Mirrors examples/trl/sft_trainer_lora.py::CLUSTER_MAP
    "codenames": "wordguessing",
    "taboo": "wordguessing",
    "guesswhat": "wordguessing",
    "wordle": "wordguessing",
    "wordle_withclue": "wordguessing",
    "wordle_withcritic": "wordguessing",
    "adventuregame": "explorationnavigation",
    "textmapworld": "explorationnavigation",
    "textmapworld_graphreasoning": "explorationnavigation",
    "textmapworld_specificroom": "explorationnavigation",
    "imagegame": "cooperation",
    "matchit_ascii": "cooperation",
    "referencegame": "cooperation",
    "privateshared": "cooperation",
}


def normalize_game_name(name: Any) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    text = text.replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or None


def _json_load_maybe(raw: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, Mapping) else None
        except Exception:
            return None
    return None


def get_game_name(example: Mapping[str, Any], *, training: bool = False) -> Optional[str]:
    g = normalize_game_name(example.get("game"))
    if g:
        return g

    meta = example.get("meta")
    meta_obj = _json_load_maybe(meta)
    if meta_obj is not None:
        g = normalize_game_name(meta_obj.get("game"))
        if g:
            return g

    for key in ("game_name", "main_game", "source_game", "benchmark_game"):
        g = normalize_game_name(example.get(key))
        if g:
            return g

    if training:
        raise ValueError(f"Missing game name in training example; keys={sorted(example.keys())}")
    return None


def _load_cfg(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML adapter_bar config files.")
        out = yaml.safe_load(text)
    else:
        out = json.loads(text)
    if not isinstance(out, Mapping):
        raise ValueError("Config must be a mapping.")
    return dict(out)


def _adapter_bar_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if "adapter_bar_sequence" in cfg and isinstance(cfg["adapter_bar_sequence"], Mapping):
        return dict(cfg["adapter_bar_sequence"])
    return dict(cfg)


def load_adapter_bar_mode_config(config_path: str, router_type: str) -> Dict[str, Any]:
    root = _adapter_bar_cfg(_load_cfg(config_path))
    mode_cfg = dict(root.get(str(router_type), {}) or {})
    merged = dict(root)
    merged.update(mode_cfg)
    merged["router_type"] = str(router_type)
    merged.setdefault("game_to_cluster", dict(DEFAULT_GAME_TO_CLUSTER))
    return merged


def _load_eval_dataset(dataset_name: str):
    local_paths = {
        "instances": __import__("os").getenv("PLAYPEN_EVAL_INSTANCES_PATH"),
        "instances-static": __import__("os").getenv("PLAYPEN_EVAL_INSTANCES_STATIC_PATH"),
    }
    local_path = local_paths.get(dataset_name)
    if local_path:
        return load_from_disk(str(Path(local_path).expanduser()))
    return load_dataset("colab-potsdam/playpen-data", dataset_name, split="validation")


def _load_router_training_mixture(seed: int):
    # Keep router training data source aligned with the existing SFT trainer:
    # playpen-data interactions + SFT-Final-Dataset.
    playpen_dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")
    playpen_dataset = playpen_dataset.filter(
        lambda episode: ((episode.get("meta") or {}).get("outcome", "") or "").lower() == "success"
    )

    sft_final_dataset = load_dataset("clembench-playpen/SFT-Final-Dataset", split="train")

    def parse_and_clean_sft_messages(example):
        chat_data = []
        raw_chat = example.get("chat")
        try:
            chat_data = json.loads(raw_chat)
        except Exception:
            try:
                import ast
                chat_data = ast.literal_eval(raw_chat)
            except Exception:
                return {"messages": []}

        cleaned = []
        if isinstance(chat_data, list):
            for msg in chat_data:
                if not isinstance(msg, Mapping):
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if role is None or content is None:
                    continue
                cleaned.append({"role": str(role), "content": str(content)})
        return {"messages": cleaned}

    drop_cols = ["chat"] + [col for col in sft_final_dataset.column_names if col == "messages"]
    sft_final_dataset = sft_final_dataset.map(
        parse_and_clean_sft_messages,
        load_from_cache_file=False,
        remove_columns=drop_cols,
    )

    def is_success_episode(example):
        success_flag = example.get("Success")
        if success_flag is not None:
            try:
                return int(success_flag) == 1
            except Exception:
                pass
        meta = example.get("meta") or {}
        outcome = meta.get("outcome") or example.get("outcome")
        if outcome is None:
            return True
        return str(outcome).lower() == "success"

    filtered_sft_final = sft_final_dataset.filter(is_success_episode, load_from_cache_file=False)
    if len(filtered_sft_final) == 0 and len(sft_final_dataset) > 0:
        filtered_sft_final = sft_final_dataset
    sft_final_dataset = filtered_sft_final

    combined_dataset = concatenate_datasets([playpen_dataset, sft_final_dataset]).shuffle(seed=int(seed))
    return combined_dataset


def _input_text_from_example(example: Mapping[str, Any]) -> str:
    msgs = example.get("messages") or example.get("chat") or []
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            if not isinstance(m, Mapping):
                continue
            role = str(m.get("role") or "").strip().lower()
            content = str(m.get("content") or "").strip()
            if role and content:
                parts.append(f"{role}: {content}")
        if parts:
            return "\n\n".join(parts)
    for k in ("prompt", "context", "instruction", "user_prompt", "input"):
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _label_for(game: Optional[str], router_type: str, cfg: Mapping[str, Any]) -> Optional[str]:
    if game is None:
        return None
    if router_type == "cluster":
        mapping = dict(cfg.get("game_to_cluster") or {})
        return normalize_game_name(mapping.get(game))
    return game


def build_router_splits(config_path: str, router_type: str, debug_num_examples: Optional[int] = None) -> Dict[str, Any]:
    cfg = load_adapter_bar_mode_config(config_path, router_type)
    seed = int(cfg.get("split_seed", 42))
    n_train = int(cfg.get("examples_per_game_train", 100))
    n_val = int(cfg.get("examples_per_game_validation", 20))
    strict = bool(cfg.get("strict_label_matching", True))
    out_dir = Path(str(cfg.get("output_dir", f"outputs/adapter_bar_sequence/{router_type}"))).expanduser()
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    ds = _load_router_training_mixture(seed)
    rows = [dict(ds[i]) for i in range(len(ds))]
    if debug_num_examples is not None and debug_num_examples > 0:
        rows = rows[: int(debug_num_examples)]

    by_game: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    skipped: List[Dict[str, Any]] = []
    for idx, ex in enumerate(rows):
        try:
            game = get_game_name(ex, training=True)
        except Exception as e:
            skipped.append({"source_index": idx, "reason": str(e)})
            continue
        by_game[game].append((idx, ex))

    import random
    rng = random.Random(seed)
    expert_names = [normalize_game_name(x) for x in (cfg.get("expert_names") or [])]
    expert_set = {x for x in expert_names if x}
    label_to_id = {lbl: i for i, lbl in enumerate(expert_names) if lbl}

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    counts_by_game = {}
    insufficient_games: List[Dict[str, Any]] = []

    for game, items in sorted(by_game.items()):
        items = list(items)
        rng.shuffle(items)
        total = len(items)
        if total >= n_train:
            train_take = n_train
            val_take = min(n_val, max(0, total - train_take))
        else:
            if strict:
                raise ValueError(f"Game '{game}' has only {total} examples (<{n_train}) in strict mode.")
            val_take = min(n_val, total)
            train_take = max(0, total - val_take)
            insufficient_games.append(
                {"game": game, "available": total, "train_used": train_take, "validation_used": val_take}
            )

        train_items = items[:train_take]
        val_items = items[train_take : train_take + val_take]
        counts_by_game[game] = {"total": total, "train": len(train_items), "validation": len(val_items)}

        for split_name, selected, target in (("train", train_items, train_rows), ("validation", val_items, val_rows)):
            for src_idx, ex in selected:
                game_name = get_game_name(ex, training=True)
                label = _label_for(game_name, router_type, cfg)
                if not label:
                    skipped.append({"source_index": src_idx, "game": game_name, "reason": "label_missing"})
                    continue
                if expert_set and label not in expert_set:
                    msg = f"label '{label}' has no matching adapter"
                    if strict:
                        raise ValueError(msg)
                    skipped.append({"source_index": src_idx, "game": game_name, "reason": msg})
                    continue
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
                row = {
                    "example_id": ex.get("example_id") or f"{game_name}:{src_idx}",
                    "game": game_name,
                    "label_name": label,
                    "label_id": int(label_to_id[label]),
                    "router_type": router_type,
                    "split": split_name,
                    "input_text": _input_text_from_example(ex),
                    "source_index": int(src_idx),
                    "meta": ex.get("meta"),
                }
                target.append(row)

    train_path = split_dir / "router_train.jsonl"
    val_path = split_dir / "router_validation.jsonl"
    for path, arr in ((train_path, train_rows), (val_path, val_rows)):
        with path.open("w", encoding="utf-8") as f:
            for row in arr:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts_by_label = Counter([r["label_name"] for r in train_rows + val_rows])
    summary = {
        "seed": seed,
        "examples_per_game_train": n_train,
        "examples_per_game_validation": n_val,
        "counts_by_game": counts_by_game,
        "counts_by_label": dict(counts_by_label),
        "skipped_examples": skipped,
        "insufficient_games_non_strict": insufficient_games,
    }
    (split_dir / "router_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def _base_model_spec(model_registry: ModelRegistry, model_name: str) -> ModelSpec:
    return model_registry.get_first_model_spec_that_unify_with(ModelSpec.from_dict({"model_name": model_name}))


def _strip_adapters(spec: ModelSpec) -> ModelSpec:
    d = spec.to_dict()
    cfg = dict(d.get("model_config") or {})
    for k in ("peft_model", "peft_models", "merge", "merge_weights"):
        cfg.pop(k, None)
    d["model_config"] = cfg
    return ModelSpec.from_dict(d)


@dataclass
class RouterBundle:
    tokenizer: Any
    base_model: Any
    router: SequenceAdapterRouter
    labels: List[str]
    label_to_id: Dict[str, int]
    cfg: Dict[str, Any]


@torch.no_grad()
def _encode_pool(tokenizer, base_model, text: str, pooling: str, device: torch.device) -> torch.Tensor:
    toks = tokenizer(text or "", return_tensors="pt", truncation=True, max_length=2048)
    toks = {k: v.to(device) for k, v in toks.items()}
    out = base_model(**toks, output_hidden_states=True)
    h = out.hidden_states[-1]
    if str(pooling).lower() == "mean":
        mask = toks.get("attention_mask")
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-9)
    else:
        last_idx = toks["attention_mask"].sum(dim=1).clamp_min(1) - 1
        pooled = h[torch.arange(h.size(0), device=h.device), last_idx]
    return pooled


def train_router(config_path: str, router_type: str, debug: bool = False) -> Dict[str, Any]:
    cfg = load_adapter_bar_mode_config(config_path, router_type)
    out_dir = Path(str(cfg.get("output_dir", f"outputs/adapter_bar_sequence/{router_type}"))).expanduser()
    split_dir = out_dir / "splits"
    train_rows = _read_jsonl(split_dir / "router_train.jsonl")
    val_rows = _read_jsonl(split_dir / "router_validation.jsonl")
    if debug:
        train_rows = train_rows[: min(len(train_rows), 64)]
        val_rows = val_rows[: min(len(val_rows), 32)]

    labels = [normalize_game_name(x) for x in (cfg.get("expert_names") or []) if normalize_game_name(x)]
    label_to_id = {x: i for i, x in enumerate(labels)}

    model_registry = ModelRegistry.from_packaged_and_cwd_files()
    base_model_name = str(cfg.get("base_model_name") or cfg.get("model_name") or "")
    if not base_model_name:
        raise ValueError("adapter_bar_sequence config requires base_model_name for router training")
    base_spec = _strip_adapters(_base_model_spec(model_registry, base_model_name))
    tokenizer, _, _ = core_hf.load_config_and_tokenizer(base_spec)
    base_model = core_hf.load_model(base_spec)
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()
    device = next(base_model.parameters()).device

    hidden_size = int(getattr(base_model.config, "hidden_size"))
    router = SequenceAdapterRouter(
        hidden_size=hidden_size,
        num_experts=len(labels),
        router_hidden_size=int(cfg.get("router_hidden_size", 512)),
        router_dropout=float(cfg.get("router_dropout", 0.1)),
        activation=str(cfg.get("router_activation", "gelu")),
    ).to(device)

    opt = torch.optim.AdamW(router.parameters(), lr=float(cfg.get("router_lr", 1e-4)))
    batch_size = int(cfg.get("router_batch_size", 16))
    epochs = int(cfg.get("router_num_epochs", 3))
    pooling = str(cfg.get("router_pooling", "last_token"))

    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    best_val = -1.0
    best_epoch = 0
    best_state = None

    def iter_batches(rows):
        for i in range(0, len(rows), batch_size):
            yield rows[i : i + batch_size]

    for epoch in range(epochs):
        router.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for batch in iter_batches(train_rows):
            feats = []
            ys = []
            for r in batch:
                lid = int(r["label_id"])
                feats.append(_encode_pool(tokenizer, base_model, str(r.get("input_text") or ""), pooling, device))
                ys.append(lid)
            x = torch.cat(feats, dim=0)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            logits = router(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(batch)
            pred = logits.argmax(dim=-1)
            tr_correct += int((pred == y).sum().item())
            tr_total += len(batch)

        router.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0
        per_game = defaultdict(lambda: [0, 0])
        per_label = defaultdict(lambda: [0, 0])
        confusion = [[0 for _ in labels] for _ in labels]
        pred_dist = Counter()
        gold_dist = Counter()
        with torch.no_grad():
            for batch in iter_batches(val_rows):
                feats = []
                ys = []
                games = []
                lbls = []
                for r in batch:
                    lid = int(r["label_id"])
                    feats.append(_encode_pool(tokenizer, base_model, str(r.get("input_text") or ""), pooling, device))
                    ys.append(lid)
                    games.append(str(r.get("game")))
                    lbls.append(str(r.get("label_name")))
                x = torch.cat(feats, dim=0)
                y = torch.tensor(ys, dtype=torch.long, device=device)
                logits = router(x)
                loss = F.cross_entropy(logits, y)
                va_loss += float(loss.item()) * len(batch)
                pred = logits.argmax(dim=-1)
                va_correct += int((pred == y).sum().item())
                va_total += len(batch)
                for i in range(len(batch)):
                    yi = int(y[i].item())
                    pi = int(pred[i].item())
                    confusion[yi][pi] += 1
                    pred_dist[labels[pi]] += 1
                    gold_dist[labels[yi]] += 1
                    g = games[i]
                    per_game[g][1] += 1
                    per_game[g][0] += 1 if yi == pi else 0
                    l = lbls[i]
                    per_label[l][1] += 1
                    per_label[l][0] += 1 if yi == pi else 0

        tr_loss /= max(1, tr_total)
        va_loss /= max(1, va_total)
        tr_acc = tr_correct / max(1, tr_total)
        va_acc = va_correct / max(1, va_total)

        history["train_loss"].append(tr_loss)
        history["validation_loss"].append(va_loss)
        history["train_accuracy"].append(tr_acc)
        history["validation_accuracy"].append(va_acc)

        if va_acc >= best_val:
            best_val = va_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in router.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in router.state_dict().items()}

    ckpt = {
        "router_state_dict": best_state,
        "labels": labels,
        "label_to_id": label_to_id,
        "config": cfg,
        "router_type": router_type,
        "hidden_size": hidden_size,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_dir / "router.pt")

    metrics = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val,
        "final_validation_accuracy": history["validation_accuracy"][-1] if history["validation_accuracy"] else 0.0,
        "train_loss_per_epoch": history["train_loss"],
        "validation_loss_per_epoch": history["validation_loss"],
        "train_accuracy_per_epoch": history["train_accuracy"],
        "validation_accuracy_per_epoch": history["validation_accuracy"],
        "per_game_validation_accuracy": {k: (v[0] / max(1, v[1])) for k, v in per_game.items()},
        "per_label_validation_accuracy": {k: (v[0] / max(1, v[1])) for k, v in per_label.items()},
        "confusion_matrix": confusion,
        "prediction_distribution": dict(pred_dist),
        "gold_label_distribution": dict(gold_dist),
        "config": cfg,
    }
    (out_dir / "router_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_router_bundle(config_path: str, router_path: str, router_type: str) -> RouterBundle:
    cfg = load_adapter_bar_mode_config(config_path, router_type)
    ckpt = torch.load(str(Path(router_path).expanduser()), map_location="cpu")
    labels = list(ckpt["labels"])
    label_to_id = dict(ckpt.get("label_to_id") or {x: i for i, x in enumerate(labels)})
    model_registry = ModelRegistry.from_packaged_and_cwd_files()
    base_model_name = str(cfg.get("base_model_name") or cfg.get("model_name") or "")
    if not base_model_name:
        raise ValueError("adapter_bar eval requires base_model_name in config")
    base_spec = _strip_adapters(_base_model_spec(model_registry, base_model_name))
    tokenizer, _, _ = core_hf.load_config_and_tokenizer(base_spec)
    base_model = core_hf.load_model(base_spec)
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    hidden_size = int(getattr(base_model.config, "hidden_size"))
    router = SequenceAdapterRouter(
        hidden_size=hidden_size,
        num_experts=len(labels),
        router_hidden_size=int(cfg.get("router_hidden_size", 512)),
        router_dropout=float(cfg.get("router_dropout", 0.1)),
        activation=str(cfg.get("router_activation", "gelu")),
    )
    router.load_state_dict(ckpt["router_state_dict"], strict=True)
    router.eval().to(next(base_model.parameters()).device)
    return RouterBundle(tokenizer=tokenizer, base_model=base_model, router=router, labels=labels, label_to_id=label_to_id, cfg=cfg)


@torch.no_grad()
def predict_adapter(bundle: RouterBundle, input_text: str) -> Dict[str, Any]:
    device = next(bundle.base_model.parameters()).device
    pooled = _encode_pool(bundle.tokenizer, bundle.base_model, input_text, bundle.cfg.get("router_pooling", "last_token"), device)
    logits = bundle.router(pooled)
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    return {
        "predicted_adapter_id": pred_id,
        "predicted_adapter": bundle.labels[pred_id],
        "router_probs": [float(x) for x in probs.detach().cpu().tolist()],
        "router_confidence": float(probs[pred_id].item()),
    }


class AdapterCacheManager:
    def __init__(self, adapter_paths: Mapping[str, str], preload_adapters: bool, max_active_adapters: int = 2):
        self.adapter_paths = {str(k): str(v) for k, v in dict(adapter_paths).items()}
        self.preload_adapters = bool(preload_adapters)
        self.max_active_adapters = max(1, int(max_active_adapters))
        self._cache = OrderedDict()
        if self.preload_adapters:
            for name, path in self.adapter_paths.items():
                self._cache[name] = path

    def touch(self, name: str) -> Tuple[bool, float, List[str]]:
        t0 = time.time()
        hit = name in self._cache
        if hit:
            self._cache.move_to_end(name)
            return True, 0.0, list(self._cache.keys())
        path = self.adapter_paths.get(name)
        if path is None:
            raise KeyError(f"No adapter path configured for '{name}'")
        self._cache[name] = path
        if len(self._cache) > self.max_active_adapters:
            self._cache.popitem(last=False)
        return False, float(time.time() - t0), list(self._cache.keys())
