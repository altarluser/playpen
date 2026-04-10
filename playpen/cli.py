import argparse
import inspect
import importlib.util as importlib_util
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, List
from datetime import datetime

import clemcore.cli as clem
from clemcore.backends import ModelSpec, ModelRegistry, BackendRegistry
from clemcore.clemgame import GameRegistry, GameSpec
from playpen import BasePlayPen, to_sub_selector
from playpen.moe import load_moe_config, tasks_by_game_experiment
from playpen.moe_loss_router import (
    LossBasedExpertRouter,
    append_router_log_row,
    extract_prompt_and_first_target,
)


def train(file_path: str, learner: ModelSpec, teacher: ModelSpec, temperature: float, max_tokens: int):
    def is_playpen(obj):
        return (inspect.isclass(obj)
                and issubclass(obj, BasePlayPen)
                and obj is not BasePlayPen
                and obj.__module__ == module.__name__  # defined in this file
                )

    try:
        file_name = os.path.splitext(file_path)[0]
        spec = importlib_util.spec_from_file_location(file_name, file_path)
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        playpen_subclasses = inspect.getmembers(module, predicate=is_playpen)
        if len(playpen_subclasses) == 0:
            raise ValueError(f"Cannot load playpen trainer, because no BasePlayPen found in {file_path}.\n"
                             f"Make sure that you have implemented a subclass of BasePlayPen and try again.")
        _, playpen_cls = playpen_subclasses[0]
    except Exception as e:
        raise RuntimeError(f"Cannot load playpen trainer, because {e}")

    game_registry = GameRegistry.from_directories_and_cwd_files()
    model_registry = ModelRegistry.from_packaged_and_cwd_files()

    learner_spec = model_registry.get_first_model_spec_that_unify_with(learner)
    print(f"Found registered model spec that unifies with {learner.to_string()} -> {learner_spec}")

    model_specs = [learner_spec]
    if teacher is not None:
        teacher_spec = model_registry.get_first_model_spec_that_unify_with(teacher)
        print(f"Found registered model spec that unifies with {teacher.to_string()} -> {teacher_spec}")
        model_specs.append(teacher_spec)

    backend_registry = BackendRegistry.from_packaged_and_cwd_files()
    for model_spec in model_specs:
        backend_selector = model_spec.backend
        if not backend_registry.is_supported(backend_selector):
            raise ValueError(f"Specified model backend '{backend_selector}' not found in backend registry.")
        print(f"Found registry entry for backend {backend_selector} "
              f"-> {backend_registry.get_first_file_matching(backend_selector)}")

    models = []
    for model_spec in model_specs:  # only now since model loading might take long
        print(f"Dynamically import backend {model_spec.backend}")
        backend = backend_registry.get_backend_for(model_spec.backend)
        model = backend.get_model_for(model_spec)
        model.set_gen_args(max_tokens=max_tokens, temperature=temperature)
        print(f"Successfully loaded {model_spec.model_name} model")
        models.append(model)

    learner_model = models[0]
    if len(models) == 1:
        playpen_cls(learner_model).learn(game_registry)
    else:
        teacher_model = models[1]
        playpen_cls(learner_model, teacher_model).learn(game_registry)


def store_eval_score(file_path: Path, name: str, value):
    try:  # first, try to load file to not overwrite already written eval scores
        with open(file_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        print(f"Update {file_path}")
    except FileNotFoundError:
        print(f"Create {file_path}")
        scores = {}
    new_scores = {**scores, **{name: value}}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(new_scores, f)
    print(json.dumps(new_scores, indent=2))
    return new_scores



def get_default_results_dir():
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    results_dir = Path("playpen-eval") / timestamp
    return results_dir


def _clem_run(game_selector, model_specs, gen_args: Dict, results_dir: Path, selector_fn=None):
    run_sig = inspect.signature(clem.run)
    kwargs = dict(gen_args=gen_args)
    if "results_dir_path" in run_sig.parameters:
        kwargs["results_dir_path"] = results_dir
    elif "results_dir" in run_sig.parameters:
        kwargs["results_dir"] = str(results_dir)
    if selector_fn is not None:
        if "sub_selector" in run_sig.parameters:
            kwargs["sub_selector"] = selector_fn
        elif "task_selector" in run_sig.parameters:
            kwargs["task_selector"] = selector_fn
    return clem.run(game_selector, model_specs, **kwargs)


def _with_temp_model_registry_entry(entry: Dict) -> Optional[callable]:
    registry_path = Path("model_registry.json")
    if not registry_path.exists() or not registry_path.is_file():
        return None
    try:
        original_text = registry_path.read_text(encoding="utf-8")
        payload = json.loads(original_text)
    except Exception:
        return None
    if not isinstance(payload, list):
        return None

    updated = False
    for idx, item in enumerate(payload):
        if isinstance(item, dict) and item.get("model_name") == entry.get("model_name"):
            payload[idx] = entry
            updated = True
            break
    if not updated:
        payload.insert(0, entry)

    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def restore():
        registry_path.write_text(original_text, encoding="utf-8")

    return restore


def _with_model_registry_file(model_registry_file: Path) -> Optional[callable]:
    requested = Path(model_registry_file).expanduser()
    target = Path("model_registry.json")

    if not requested.exists() or not requested.is_file():
        raise FileNotFoundError(f"Model registry file not found: {requested}")

    try:
        if requested.resolve() == target.resolve():
            return None
    except Exception:
        if str(requested) == str(target):
            return None

    source_text = requested.read_text(encoding="utf-8")
    payload = json.loads(source_text)
    if not isinstance(payload, (list, dict)):
        raise ValueError(
            f"Model registry file must contain a JSON object or list, got {type(payload).__name__}: {requested}"
        )
    normalized_text = json.dumps(payload, indent=2)

    had_target = target.exists() and target.is_file()
    original_text = target.read_text(encoding="utf-8") if had_target else None
    target.write_text(normalized_text, encoding="utf-8")

    def restore():
        if had_target:
            target.write_text(original_text, encoding="utf-8")
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass

    return restore


@contextmanager
def _eval_context_env(*, game: Optional[str] = None, split: Optional[str] = None, regime: Optional[str] = None):
    keys = {
        "PLAYPEN_EVAL_GAME": game,
        "PLAYPEN_EVAL_SPLIT": split,
        "PLAYPEN_EVAL_REGIME": regime,
    }
    previous = {k: os.environ.get(k) for k in keys}
    try:
        for key, value in keys.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _find_clembench_roots() -> Tuple[Path, ...]:
    roots = []
    local = Path("clembench")
    if local.exists() and local.is_dir():
        roots.append(local)

    registry_file = Path("game_registry.json")
    if registry_file.exists() and registry_file.is_file():
        try:
            entries = json.loads(registry_file.read_text(encoding="utf-8"))
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    p = entry.get("benchmark_path")
                    if not p:
                        continue
                    candidate = Path(str(p)).expanduser()
                    if candidate.exists() and candidate.is_dir():
                        roots.append(candidate)
        except Exception:
            pass

    # de-duplicate while keeping order
    dedup = []
    seen = set()
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return tuple(dedup)


def _build_game_meta_index(roots: Tuple[Path, ...]) -> Dict[str, Dict]:
    meta: Dict[str, Dict] = {}
    for root in roots:
        try:
            for clemgame_path in root.glob("**/clemgame.json"):
                try:
                    payload = json.loads(clemgame_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, list):
                    continue
                for spec in payload:
                    if not isinstance(spec, dict):
                        continue
                    g = spec.get("game_name")
                    if not g:
                        continue
                    meta[str(g)] = spec
        except Exception:
            continue
    return meta


def _has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in ("*", "?", "[", "]"))


def _detect_moe_type(moe) -> str:
    if getattr(moe, "router", None) is not None:
        return "router"
    if any(getattr(r, "keywords", None) for r in moe.routes):
        return "keyword"
    if moe.route_by_experiment:
        return "game_experiment"
    if all(not _has_glob(getattr(r, "game", "")) for r in moe.routes):
        return "game_name"
    return "pattern"


def _update_players_model_jsons(results_root: Path, game_name: str, moe_info: Dict) -> int:
    results_root = Path(results_root)
    candidates = [p for p in results_root.rglob("players_model.json") if game_name in p.parts]
    if not candidates:
        candidates = list(results_root.rglob("players_model.json"))

    updated = 0
    for p in candidates:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        def apply_moe(obj):
            if isinstance(obj, dict):
                existing = obj.get("moe")
                if isinstance(existing, dict):
                    obj["moe"] = {**existing, **moe_info}
                else:
                    obj["moe"] = dict(moe_info)
            return obj

        if isinstance(data, list):
            data = [apply_moe(x) if isinstance(x, dict) else x for x in data]
        elif isinstance(data, dict):
            data = apply_moe(data)
        else:
            continue

        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        updated += 1

    return updated


def _collect_prompt_strings(obj, *, max_chars: int) -> List[str]:
    out: List[str] = []

    def walk(x):
        if len(" ".join(out)) >= max_chars:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                key = str(k).lower()
                if any(tok in key for tok in ("prompt", "instruction", "system", "rules", "role")) and isinstance(v, str):
                    s = v.strip()
                    if s:
                        out.append(s)
                        continue
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)
        elif isinstance(x, str):
            # Only take free strings when they're not tiny.
            s = x.strip()
            if len(s) >= 40:
                out.append(s)

    walk(obj)
    # Dedup while keeping order
    dedup = []
    seen = set()
    for s in out:
        key = s[:200]
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)
        if sum(len(x) for x in dedup) >= max_chars:
            break
    return dedup


def _try_load_instance_prompts(roots: Tuple[Path, ...], game_spec: Dict, *, max_chars: int = 4000) -> List[str]:
    main_game = str(game_spec.get("main_game") or game_spec.get("game_name") or "").strip()
    game_name = str(game_spec.get("game_name") or "").strip()
    instances = game_spec.get("instances") or "instances"
    instances_file = str(instances)
    if not instances_file.endswith(".json"):
        instances_file = instances_file + ".json"

    candidates = []
    for root in roots:
        for g in (main_game, game_name):
            if not g:
                continue
            candidates.append(root / g / "in" / instances_file)
            # Some games store everything in instances.json only; try that as a fallback.
            candidates.append(root / g / "in" / "instances.json")

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                payload = json.loads(p.read_text(encoding="utf-8"))
                return _collect_prompt_strings(payload, max_chars=max_chars)
        except Exception:
            continue
    return []


def _game_context_text(game: str, experiment: Optional[str], meta_index: Dict[str, Dict]) -> str:
    parts = [f"game={game}"]
    if experiment:
        parts.append(f"experiment={experiment}")
    spec = meta_index.get(game) or {}
    desc = spec.get("description")
    if desc:
        parts.append(f"description={desc}")
    roles = spec.get("roles")
    if isinstance(roles, list) and roles:
        parts.append("roles=" + ", ".join(str(r) for r in roles))
    players = spec.get("players")
    if players is not None:
        parts.append(f"players={players}")
    main_game = spec.get("main_game")
    if main_game:
        parts.append(f"main_game={main_game}")

    roots = _find_clembench_roots()
    prompts = _try_load_instance_prompts(roots, spec, max_chars=4000)
    if prompts:
        parts.append("prompts=" + "\n---\n".join(prompts))
    return "\n".join(parts)


def _row_get(row: Mapping[str, Any], *keys: str, default=None):
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return default


def _regime_from_row(row: Mapping[str, Any], id_regimes: Tuple[str, ...]) -> str:
    raw = _row_get(row, "regime", "domain", "ood", default=None)
    if raw is None:
        return "id"
    text = str(raw).strip().lower()
    if text in {"0", "false", "id", "in_domain", "in-domain", "indomain"}:
        return "id"
    if text in {"1", "true", "ood", "out_domain", "out-of-domain", "outdomain"}:
        return "ood"
    if text in {str(x).strip().lower() for x in id_regimes}:
        return "id"
    return text


def generate_with_selected_expert(
    *,
    suite_results_dir: Path,
    game_name: str,
    task_id: int,
    run_spec: ModelSpec,
    gen_args: Dict,
    split: str = "validation",
    regime: str = "unknown",
):
    def sub_selector_single(game: str, experiment: str, _g=game_name, _task_id=int(task_id)):
        if game != _g:
            return []
        return [_task_id]

    with _eval_context_env(game=game_name, split=split, regime=regime):
        _clem_run(game_name, [run_spec], gen_args, suite_results_dir, selector_fn=sub_selector_single)


def evaluate_suite(suite: str, model_spec: ModelSpec, gen_args: Dict, results_dir: Path, game_selector: str,
                   dataset_name: str):
    suite_results_dir = results_dir / suite
    if dataset_name is not None:
        from datasets import load_dataset
        dataset = load_dataset("colab-potsdam/playpen-data", dataset_name, split="validation")
        selector_fn = to_sub_selector(dataset)
        tasks_by_group = tasks_by_game_experiment(dataset)
        model_cfg = dict(getattr(model_spec, "model_config", {}) or {})
        moe_enabled = bool(model_cfg.get("moe_enabled", False))
        # If user passed an explicit game name (e.g. "-g wordle"), avoid dataset task sub-selection.
        # Some clemcore versions can otherwise pick up interactions outside that game during scoring.
        if isinstance(game_selector, str):
            selector = game_selector.strip()
            if selector and not selector.startswith("{") and not selector.startswith("["):
                selector_fn = None
        if moe_enabled:
            explicit_game = None
            if isinstance(game_selector, str):
                selector = game_selector.strip()
                if selector and not selector.startswith("{") and not selector.startswith("["):
                    explicit_game = selector

            if explicit_game is not None:
                targets = [(g, e) for (g, e) in sorted(tasks_by_group.keys()) if g == explicit_game]
            else:
                targets = sorted(tasks_by_group.keys())

            for game_name, experiment_name in targets:
                def sub_selector_exact(game: str, experiment: str, _g=game_name, _e=experiment_name):
                    if game == _g and experiment == _e:
                        return tasks_by_group.get((game, experiment), [])
                    return []

                with _eval_context_env(game=game_name, split="validation", regime="unknown"):
                    _clem_run(game_name, [model_spec], gen_args, suite_results_dir, selector_fn=sub_selector_exact)
        else:
            context_game = game_selector if isinstance(game_selector, str) else "benchmark"
            with _eval_context_env(game=context_game, split="validation", regime="unknown"):
                _clem_run(game_selector, [model_spec], gen_args, suite_results_dir, selector_fn=selector_fn)
    clem.score(game_selector, str(suite_results_dir))
    clem.transcripts(game_selector, str(suite_results_dir))
    df = clem.clemeval.perform_evaluation(str(suite_results_dir), return_dataframe=True)
    clem_score = df["-, clemscore"].iloc[0]
    return clem_score


def _extract_clemscore(df, model_name: str) -> float:
    # clemcore returns a DataFrame indexed by model name in typical usage, but keep this robust.
    try:
        return float(df.loc[model_name, "-, clemscore"])
    except Exception:
        try:
            return float(df["-, clemscore"].iloc[0])
        except Exception as e:
            raise RuntimeError(f"Could not extract clemscore for model '{model_name}' from evaluation output.") from e


def _resolve_registered_model_spec(model_registry: ModelRegistry, model_name: str) -> ModelSpec:
    spec = ModelSpec.from_dict({"model_name": model_name})
    return model_registry.get_first_model_spec_that_unify_with(spec)


def _with_model_config(spec: ModelSpec, extra_model_config: Dict) -> ModelSpec:
    spec_dict = spec.to_dict()
    model_config = dict(spec_dict.get("model_config", {}))
    model_config.update(extra_model_config or {})
    spec_dict["model_config"] = model_config
    return ModelSpec.from_dict(spec_dict)


def _with_model_name(spec: ModelSpec, model_name: str) -> ModelSpec:
    spec_dict = spec.to_dict()
    spec_dict["model_name"] = model_name
    return ModelSpec.from_dict(spec_dict)


def evaluate_suite_moe(
    suite: str,
    moe,
    gen_args: Dict,
    results_dir: Path,
    game_selector,
    dataset_name: str,
    merge: str,
):
    suite_results_dir = results_dir / suite

    routing_manifest = {
        "moe_name": moe.name,
        "default_model": moe.default_model,
        "route_by_experiment": moe.route_by_experiment,
        "routes": [r.__dict__ for r in moe.routes],
        "assignments": [],
    }

    if dataset_name is not None:
        from datasets import load_dataset

        dataset = load_dataset("colab-potsdam/playpen-data", dataset_name, split="validation")
        tasks_by_group = tasks_by_game_experiment(dataset)
        game_meta_index = _build_game_meta_index(_find_clembench_roots())
        moe_type = _detect_moe_type(moe)

        # If a user explicitly passes a single game name (e.g., "-g wordle_withcritic"),
        # keep routing restricted to that game. For benchmark selectors we rely on the dataset.
        explicit_game = None
        if isinstance(game_selector, str):
            selector = game_selector.strip()
            if selector and not selector.startswith("{") and not selector.startswith("["):
                explicit_game = selector

        if explicit_game is not None:
            game_names = [explicit_game]
        else:
            game_names = sorted({g for (g, _) in tasks_by_group.keys()})
        game_name_set = set(game_names)

        model_registry = ModelRegistry.from_packaged_and_cwd_files()
        moe_entry_exists = True
        try:
            _resolve_registered_model_spec(model_registry, moe.name)
        except Exception:
            moe_entry_exists = False

        def sub_selector_all(game: str, experiment: str):
            return tasks_by_group.get((game, experiment), [])

        if moe.loss_router is not None:
            configured_experts = list(moe.loss_router.experts or [])
            if not configured_experts:
                configured_experts = sorted({r.model for r in moe.routes if getattr(r, "model", None)})
            if not configured_experts:
                raise ValueError("loss_router is enabled but no experts were configured.")

            loss_router = LossBasedExpertRouter(
                model_registry=model_registry,
                base_model_name=moe.default_model,
                experts=configured_experts,
            )
            router_log_path = (
                Path(moe.loss_router.log_path).expanduser()
                if moe.loss_router.log_path
                else (results_dir / f"{moe.name}.{suite}.loss_router.jsonl")
            )
            oracle_field = str(moe.loss_router.oracle_field or "oracle_expert")
            id_regimes = tuple(moe.loss_router.id_regimes or ("id",))
            merge_label = merge if merge is not None else "none"

            for row in dataset:
                game_name = str(_row_get(row, "game", "game_name", default="") or "")
                experiment_name = _row_get(row, "experiment", "experiment_name", default=None)
                experiment_name = str(experiment_name) if experiment_name is not None else None
                task_id = _row_get(row, "task_id", "game_id", "instance_id", default=None)
                if not game_name or task_id is None:
                    continue

                if explicit_game is not None and game_name != explicit_game:
                    continue
                if game_name not in game_name_set:
                    continue

                prompt_text, target_text = extract_prompt_and_first_target(row)
                if not prompt_text or not target_text:
                    context_text = _game_context_text(game_name, experiment_name, game_meta_index)
                    expert = moe.select_model(game_name, experiment_name, context_text=context_text)
                    scored = {
                        "selected_expert": expert,
                        "expert_scores": [],
                        "margin_to_second_best": float("nan"),
                        "details": [],
                    }
                else:
                    scored = loss_router.select_expert_by_loss(prompt_text=prompt_text, target_text=target_text)
                    expert = str(scored["selected_expert"])

                expert_spec = _resolve_registered_model_spec(model_registry, expert)
                run_spec = _with_model_name(expert_spec, moe.name)
                extra_cfg = {"moe_expert": expert}
                if merge is not None:
                    extra_cfg["merge"] = merge
                run_spec = _with_model_config(run_spec, extra_cfg)

                restore_registry = None
                if not moe_entry_exists:
                    restore_registry = _with_temp_model_registry_entry(run_spec.to_dict())
                    if restore_registry is not None:
                        print(f"[MoE] injected temporary model_registry entry for {moe.name}")

                regime = _regime_from_row(row, id_regimes)
                try:
                    generate_with_selected_expert(
                        suite_results_dir=suite_results_dir,
                        game_name=game_name,
                        task_id=int(task_id),
                        run_spec=run_spec,
                        gen_args=gen_args,
                        split="validation",
                        regime=regime,
                    )
                finally:
                    if restore_registry is not None:
                        restore_registry()

                example_id = str(_row_get(row, "example_id", default=f"{game_name}:{experiment_name}:{task_id}"))
                oracle_expert = _row_get(row, oracle_field, "oracle_expert", default=None)
                details = list(scored.get("details") or [])
                score_map = {
                    str(item["expert_name"]): float(item["mean_nll"])
                    for item in details
                    if isinstance(item, Mapping) and item.get("expert_name") is not None
                }
                selected_mean_nll = score_map.get(expert)
                margin = float(scored.get("margin_to_second_best", float("nan")))

                log_row = {
                    "example_id": example_id,
                    "game": game_name,
                    "experiment": experiment_name,
                    "task_id": int(task_id),
                    "split": "validation",
                    "regime": regime,
                    "selected_expert": expert,
                    "oracle_expert": str(oracle_expert) if oracle_expert is not None else None,
                    "top1_minus_top2_margin": margin,
                    "selected_mean_nll": selected_mean_nll,
                    "final_evaluation_score": None,
                }
                for expert_name in configured_experts:
                    log_row[f"mean_nll_{expert_name}"] = score_map.get(expert_name)
                append_router_log_row(router_log_path, log_row)

                print(
                    f"[MoE-LossRouter] suite={suite} game={game_name} task_id={task_id} "
                    f"selected={expert} margin={margin:.6f} merge={merge_label}"
                )

                moe_info = {
                    "moe_name": moe.name,
                    "moe_type": "loss_router",
                    "default_model": moe.default_model,
                    "expert_model": expert,
                    "game": game_name,
                    "experiment": experiment_name,
                    "task_id": int(task_id),
                    "merge": merge_label,
                }
                updated = _update_players_model_jsons(suite_results_dir, game_name, moe_info)
                if updated == 0:
                    print(f"[MoE] warning: no players_model.json found under {suite_results_dir} for game={game_name}")

                routing_manifest["assignments"].append(
                    {
                        "game": game_name,
                        "experiment": experiment_name,
                        "task_id": int(task_id),
                        "expert_model": expert,
                        "router": "loss_router",
                    }
                )
            routing_manifest["loss_router_log"] = str(router_log_path)
        else:
            if moe.route_by_experiment:
                targets = [(g, e) for (g, e) in sorted(tasks_by_group.keys()) if g in game_name_set]
            else:
                targets = [(g, None) for g in game_names]

            for game_name, experiment_name in targets:
                context_text = _game_context_text(game_name, experiment_name, game_meta_index)
                expert = moe.select_model(game_name, experiment_name, context_text=context_text)
                expert_spec = _resolve_registered_model_spec(model_registry, expert)
                experiment_label = experiment_name if experiment_name is not None else "-"
                merge_label = merge if merge is not None else "none"
                print(f"[MoE] suite={suite} game={game_name} experiment={experiment_label} -> expert={expert} (merge={merge_label})")

                # Keep one virtual model name on disk/leaderboard while varying the underlying adapter per call.
                run_spec = _with_model_name(expert_spec, moe.name)

                extra_cfg = {"moe_expert": expert}
                if merge is not None:
                    extra_cfg["merge"] = merge
                run_spec = _with_model_config(run_spec, extra_cfg)

                restore_registry = None
                if not moe_entry_exists:
                    restore_registry = _with_temp_model_registry_entry(run_spec.to_dict())
                    if restore_registry is not None:
                        print(f"[MoE] injected temporary model_registry entry for {moe.name}")

                if moe.route_by_experiment and experiment_name is not None:
                    def sub_selector_only(game: str, experiment: str, _g=game_name, _e=experiment_name):
                        if game == _g and experiment == _e:
                            return tasks_by_group.get((game, experiment), [])
                        return []

                    try:
                        with _eval_context_env(game=game_name, split="validation", regime="unknown"):
                            _clem_run(game_name, [run_spec], gen_args, suite_results_dir, selector_fn=sub_selector_only)
                    finally:
                        if restore_registry is not None:
                            restore_registry()
                else:
                    try:
                        with _eval_context_env(game=game_name, split="validation", regime="unknown"):
                            _clem_run(game_name, [run_spec], gen_args, suite_results_dir, selector_fn=sub_selector_all)
                    finally:
                        if restore_registry is not None:
                            restore_registry()

                moe_info = {
                    "moe_name": moe.name,
                    "moe_type": moe_type,
                    "default_model": moe.default_model,
                    "expert_model": expert,
                    "game": game_name,
                    "experiment": experiment_name,
                    "merge": merge_label,
                }
                updated = _update_players_model_jsons(suite_results_dir, game_name, moe_info)
                if updated == 0:
                    print(f"[MoE] warning: no players_model.json found under {suite_results_dir} for game={game_name}")

                routing_manifest["assignments"].append(
                    {"game": game_name, "experiment": experiment_name, "expert_model": expert}
                )

        manifest_path = results_dir / f"{moe.name}.{suite}.moe.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(routing_manifest, f, indent=2)

    clem.score(game_selector, str(suite_results_dir))
    clem.transcripts(game_selector, str(suite_results_dir))
    df = clem.clemeval.perform_evaluation(str(suite_results_dir), return_dataframe=True)
    return _extract_clemscore(df, moe.name)


def evaluate(suite: str, model_spec: ModelSpec, gen_args: Dict, results_dir: Path, game_selector: str,
             skip_gameplay: bool):
    overall_results_file = results_dir / f"{model_spec.model_name}.val.json"
    if suite in ["all", "clem"]:
        dataset_name = None if skip_gameplay else "instances"
        _game_selector = GameSpec.from_dict({"benchmark": ["2.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        clem_score = evaluate_suite("clem", model_spec, gen_args, results_dir, _game_selector, dataset_name)
        store_eval_score(overall_results_file, "clemscore", clem_score)
    if suite in ["all", "static"]:
        dataset_name = None if skip_gameplay else "instances-static"
        _game_selector = GameSpec.from_dict({"benchmark": ["static_1.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        stat_score = evaluate_suite("static", model_spec, gen_args, results_dir, _game_selector, dataset_name)
        store_eval_score(overall_results_file, "statscore", stat_score)


def evaluate_moe(
    suite: str,
    moe,
    gen_args: Dict,
    results_dir: Path,
    game_selector: str,
    skip_gameplay: bool,
    merge: str,
):
    overall_results_file = results_dir / f"{moe.name}.val.json"
    if suite in ["all", "clem"]:
        dataset_name = None if skip_gameplay else "instances"
        _game_selector = GameSpec.from_dict({"benchmark": ["2.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        clem_score = evaluate_suite_moe("clem", moe, gen_args, results_dir, _game_selector, dataset_name, merge)
        store_eval_score(overall_results_file, "clemscore", clem_score)
    if suite in ["all", "static"]:
        dataset_name = None if skip_gameplay else "instances-static"
        _game_selector = GameSpec.from_dict({"benchmark": ["static_1.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        stat_score = evaluate_suite_moe("static", moe, gen_args, results_dir, _game_selector, dataset_name, merge)
        store_eval_score(overall_results_file, "statscore", stat_score)


def cli(args: argparse.Namespace):
    if args.command_name == "list":
        if args.mode == "games":
            clem.list_games(args.selector, args.verbose)
        elif args.mode == "models":
            clem.list_models(args.verbose)
        elif args.mode == "backends":
            clem.list_backends(args.verbose)
        else:
            print(f"Cannot list {args.mode}. Choose an option documented at 'list -h'.")
    if args.command_name == "run":
        if getattr(args, "bf16", None) is not None:
            os.environ["PLAYPEN_BF16"] = "1" if args.bf16.lower() == "true" else "0"
        learner_spec = ModelSpec.from_string(args.learner)
        teacher_spec = ModelSpec.from_string(args.teacher) if args.teacher is not None else None
        train(args.file_path, learner_spec, teacher_spec, args.temperature, args.max_tokens)

    if args.command_name == "eval":
        gen_args = dict(temperature=args.temperature, max_tokens=args.max_tokens)
        restore_registry = _with_model_registry_file(args.model_registry)
        try:
            if args.moe is not None:
                moe = load_moe_config(args.moe, default_name=args.model, default_model=args.model)
                evaluate_moe(args.suite, moe, gen_args, args.results_dir, args.game, args.skip_gameplay, args.merge)
            else:
                model_spec = ModelSpec.from_string(args.model)
                if args.merge is not None:
                    model_spec_dict = model_spec.to_dict()
                    model_config = dict(model_spec_dict.get("model_config", {}))
                    model_config["merge"] = args.merge
                    model_spec_dict["model_config"] = model_config
                    model_spec = ModelSpec.from_dict(model_spec_dict)
                evaluate(args.suite, model_spec, gen_args, args.results_dir, args.game, args.skip_gameplay)
        finally:
            if restore_registry is not None:
                restore_registry()


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    list_parser = sub_parsers.add_parser("list")
    list_parser.add_argument("mode", choices=["games", "models", "backends"],
                             default="games", nargs="?", type=str,
                             help="Choose to list available games, models or backends."
                                  " Default: games")
    list_parser.add_argument("-v", "--verbose", action="store_true")
    list_parser.add_argument("-s", "--selector", type=str, default="all")

    train_parser = sub_parsers.add_parser("run")
    train_parser.add_argument("file_path", type=str,
                              help="The path to the trainer file to use for learning.")
    train_parser.add_argument("-l", "--learner", type=str,
                              help="The model name of the learner model (as listed by 'playpen list models').")
    train_parser.add_argument("-t", "--teacher", type=str, default=None,
                              help="(Optional) Model name of the partner model (as listed by 'playpen list models')."
                                   " Note: Non-interactive methods (like SFT) may not require a teacher model."
                                   " Default: None.",
                              required=False)
    train_parser.add_argument("-T", "--temperature", type=float, required=False, default=0.0,
                              help="The temperature used for generation. Should be the same as during training. "
                                   "Default: 0.0.")
    train_parser.add_argument("-L", "--max_tokens", type=int, required=False, default=300,
                              help="The token limit for generated responses. Should be the same as during training. "
                                   "Default: 300.")
    train_parser.add_argument("--bf16", choices=["true", "false"], default=None,
                              help="(Optional) Override trainer bf16 setting for this run by exporting "
                                   "PLAYPEN_BF16 internally. Default: not set (trainer default).")

    # Note: For now, we directly bound the eval to the playpen-data validate split.
    eval_parser = sub_parsers.add_parser("eval",
                                         description="Run the playpen eval pipelines to compute clem- and statscore.")
    eval_parser.add_argument("model", type=str,
                             help="The model name of the model to be evaluated (as listed by 'playpen list models').")
    eval_parser.add_argument("--suite", choices=["clem", "static", "all"], default="all",
                             nargs="?", type=str,
                             help="(Optional) Suite selector for the eval run."
                                  " Default: all")
    eval_parser.add_argument("-g", "--game", type=str,
                             help="(Optional) Game selector, such as a game name or a GameSpec JSON string."
                                  " Default: {\"benchmark\": [\"2.0\"]} (clem suite)"
                                  " or {\"benchmark\": [\"static_1.0\"]} (static suite)")
    eval_parser.add_argument("-r", "--results_dir", type=Path, default=get_default_results_dir(),
                             help="(Optional) Relative or absolute path to a playpen-eval results directory."
                                  " This is expected to be one level above 'clem' or 'static' results."
                                  " Default: playpen-eval/<timestamp>.")
    eval_parser.add_argument("--model-registry", type=Path, default=Path("model_registry.json"),
                             help="(Optional) Path to the model registry JSON used for this eval run."
                                  " Default: ./model_registry.json")
    eval_parser.add_argument("--skip_gameplay", action="store_true",
                             help="(Optional) Flag only re-calculate the clemscore for a given 'results_dir'."
                                  " Using this option skips gameplay. Only relevant for the clem suite."
                                  " Default: False.")
    eval_parser.add_argument("--merge", choices=["task_arithmetic", "weight_averaging", "ties"],
                             help="(Optional) Merge LoRA adapters specified in model_config before evaluation."
                                  " Default: no merge (adapter injection).")
    eval_parser.add_argument("--moe", type=str, default=None, required=False,
                             help="(Optional) Route games/experiments to different expert models (MoE). "
                                  "Provide a JSON file path or an inline JSON/Python-literal dict. "
                                  "Results are stored under a single virtual model name given by 'model'. "
                                  "Default: disabled.")
    eval_parser.add_argument("-T", "--temperature", type=float, default=0.0,
                             help="The temperature used for generation. Should be the same as during training."
                                  " Default: 0.0.")
    eval_parser.add_argument("-L", "--max_tokens", type=int, default=300,
                             help="The token limit for generated responses. Should be the same as during training."
                                  " Default: 300.")

    # todo: add a 'playpen play' option to allow collection of new interaction data on the train split

    cli(parser.parse_args())


if __name__ == "__main__":
    main()
