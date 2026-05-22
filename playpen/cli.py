import argparse
import inspect
import importlib.util as importlib_util
import json
import os
import fcntl
import signal
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, List
from datetime import datetime

import clemcore.cli as clem
from clemcore.backends import ModelSpec, ModelRegistry, BackendRegistry
from clemcore.clemgame import GameRegistry, GameSpec
import clemcore.backends.huggingface_local_api as core_hf
from playpen import BasePlayPen
from playpen.moe import load_moe_config, tasks_by_game_experiment
from playpen.moe_runtime import apply_moe_if_requested
from playpen.moe_loss_router import (
    LossBasedExpertRouter,
    append_router_log_row,
    extract_prompt_and_first_target,
)
from playpen.adapter_bar_sequence import (
    AdapterCacheManager,
    get_game_name,
    load_router_bundle,
    predict_adapter,
)


def _install_model_registry_runtime_path_patch():
    runtime_path_raw = os.getenv("PLAYPEN_MODEL_REGISTRY_RUNTIME_PATH")
    if not runtime_path_raw:
        return

    runtime_path = Path(runtime_path_raw).expanduser()
    if getattr(ModelRegistry, "_playpen_runtime_registry_patched", False):
        return

    original = ModelRegistry.from_packaged_and_cwd_files.__func__

    @classmethod
    def _patched(cls) -> "ModelRegistry":
        configured_runtime = Path(
            os.getenv("PLAYPEN_MODEL_REGISTRY_RUNTIME_PATH", str(runtime_path))
        ).expanduser()
        if configured_runtime.exists() and configured_runtime.is_file():
            registry = cls.from_json_file(configured_runtime)
            try:
                # Keep packaged fallback entries, but do not consult cwd when an
                # explicit runtime registry was provided for this job.
                import importlib.resources as importlib_resources

                with importlib_resources.files("clemcore.backends").joinpath("model_registry.json").open("r") as f:
                    registry.register_from_list(json.load(f), lookup_source="packaged")
            except Exception:
                pass
            return registry
        return original(cls)

    ModelRegistry.from_packaged_and_cwd_files = _patched
    ModelRegistry._playpen_runtime_registry_patched = True
    ModelRegistry._playpen_runtime_registry_original = original
    print(f"[playpen eval] using runtime model registry path: {runtime_path}")


def _install_hf_local_moe_patch():
    if getattr(core_hf, "_playpen_moe_loader_patched", False):
        return

    original = core_hf.load_model

    def _patched_load_model(model_spec):
        model_config = getattr(model_spec, "model_config", {}) or {}
        if not bool(model_config.get("moe_enabled", False)):
            return original(model_spec)

        patched_spec = model_spec
        if model_config.get("moe_lora_adapter_path") and "peft_model" in model_config:
            spec_dict = model_spec.to_dict()
            spec_model_config = dict(spec_dict.get("model_config", {}))
            spec_model_config.pop("peft_model", None)
            spec_dict["model_config"] = spec_model_config
            patched_spec = ModelSpec.from_dict(spec_dict)
            core_hf.stdout_logger.info(
                "Skip generic peft_model load for MoE-enabled model %s; MoE adapter will be applied after runtime wrapping.",
                getattr(model_spec, "model_name", "model"),
            )

        model = original(patched_spec)
        return apply_moe_if_requested(model, model_spec, logger=core_hf.logger)

    core_hf.load_model = _patched_load_model
    core_hf._playpen_moe_loader_patched = True
    core_hf._playpen_moe_loader_original = original
    print("[playpen eval] installed HF local MoE load patch")


def _parse_int_env(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _install_player_call_timeout_patch():
    timeout_s = _parse_int_env("PLAYPEN_INSTANCE_CALL_TIMEOUT_SECONDS", 0)
    if timeout_s <= 0:
        return
    if not hasattr(signal, "SIGALRM"):
        return

    try:
        from clemcore.clemgame.player import Player
    except Exception:
        return

    if getattr(Player, "_playpen_timeout_patched", False):
        return

    original_call = Player.__call__

    def _timed_call(self, *args, **kwargs):
        def _handle_timeout(_signum, _frame):
            raise TimeoutError(
                f"Player call timed out after {timeout_s}s for model={getattr(getattr(self, '_model', None), 'name', 'unknown')}"
            )

        prev_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(timeout_s)
        try:
            return original_call(self, *args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)

    Player.__call__ = _timed_call
    Player._playpen_timeout_patched = True
    print(f"[playpen eval] enabled Player.__call__ timeout: {timeout_s}s")


def _install_instance_timeout_patch():
    timeout_s = _parse_int_env("PLAYPEN_INSTANCE_TIMEOUT_SECONDS", 0)
    if timeout_s <= 0:
        return
    if not hasattr(signal, "SIGALRM"):
        return

    try:
        from clemcore.clemgame.runners import sequential as _seq_runner
    except Exception:
        return

    if getattr(_seq_runner, "_playpen_instance_timeout_patched", False):
        return

    original_run = _seq_runner.run

    def _timed_run(game_benchmark, game_instances, player_models, *, callbacks):
        from tqdm import tqdm
        from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv

        callbacks.on_benchmark_start(game_benchmark)
        game_env = GameMasterEnv(game_benchmark, callbacks=callbacks)
        error_count = 0

        for row in tqdm(game_instances, desc="Playing game instances"):
            prev_handler = signal.getsignal(signal.SIGALRM)

            def _handle_timeout(_signum, _frame):
                game_id = None
                try:
                    game_id = row.get("game_instance", {}).get("game_id")
                except Exception:
                    pass
                raise TimeoutError(
                    f"Instance timed out after {timeout_s}s for game={getattr(game_benchmark, 'game_name', 'unknown')}"
                    + (f", game_id={game_id}" if game_id is not None else "")
                )

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(timeout_s)
            try:
                game_env.reset(options={
                    "player_models": player_models,
                    "experiment": row["experiment"],
                    "game_instance": row["game_instance"]
                })
                for model in player_models:
                    model.reset()
                for agent_id in game_env.agent_iter():
                    context, reward, termination, truncation, info = game_env.last(observe=True)
                    if termination or truncation:
                        response = None
                    else:
                        player = game_env.player_by_agent_id[agent_id]
                        response = player(context)
                    game_env.step(response)
            except Exception:
                message = (
                    f"{game_benchmark.game_name}: Exception for instance "
                    f"{row.get('game_instance', {}).get('game_id', 'unknown')} (but continue)"
                )
                _seq_runner.module_logger.exception(message)
                # Mark this episode as aborted so scoring/files treat it as a played failed episode,
                # not as a silently skipped one.
                try:
                    if getattr(game_env, "game_master", None) is not None:
                        try:
                            game_env.game_master.state.abort()
                        except Exception:
                            pass
                        try:
                            game_env.callbacks.on_game_end(
                                game_env.game_master,
                                row["game_instance"],
                                exception=RuntimeError("instance timeout/exception"),
                                rewards=game_env.rewards,
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
                error_count += 1
                for model in player_models:
                    model.reset()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)

        game_env.close()
        if error_count > 0:
            _seq_runner.stdout_logger.error(
                f"{game_benchmark.game_name}: '{error_count}' exceptions occurred: See clembench.log for details."
            )
        callbacks.on_benchmark_end(game_benchmark)

    _seq_runner.run = _timed_run
    _seq_runner._playpen_instance_timeout_patched = True
    _seq_runner._playpen_instance_timeout_original_run = original_run
    print(f"[playpen eval] enabled per-instance timeout: {timeout_s}s")


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


def _load_eval_dataset(dataset_name: str):
    """
    Load eval data either from local save_to_disk snapshots (preferred when provided)
    or from the default HF dataset source.
    """
    from datasets import load_dataset, load_from_disk

    local_paths = {
        "instances": os.getenv("PLAYPEN_EVAL_INSTANCES_PATH"),
        "instances-static": os.getenv("PLAYPEN_EVAL_INSTANCES_STATIC_PATH"),
    }
    local_path = local_paths.get(dataset_name)
    if local_path:
        dataset_path = Path(local_path).expanduser()
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Configured local eval dataset path does not exist for '{dataset_name}': {dataset_path}"
            )
        print(f"[eval] loading '{dataset_name}' from local path: {dataset_path}")
        return load_from_disk(str(dataset_path))

    print(f"[eval] loading '{dataset_name}' from HF dataset: colab-potsdam/playpen-data")
    return load_dataset("colab-potsdam/playpen-data", dataset_name, split="validation")


def _norm_key(value: Any) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _row_value(row: Any, *names: str):
    for nm in names:
        # Mapping-style rows.
        if isinstance(row, Mapping):
            try:
                if nm in row and row.get(nm) is not None:
                    return row.get(nm)
            except Exception:
                pass
        # Attribute-style rows.
        try:
            if hasattr(row, nm):
                val = getattr(row, nm)
                if val is not None:
                    return val
        except Exception:
            pass
        # Dict-like objects with __getitem__ only.
        try:
            val = row[nm]
            if val is not None:
                return val
        except Exception:
            pass
    return None


def _selector_row_triplet(row: Any) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Normalize different clemcore row shapes into (game, experiment, task_id).
    Supports:
    - flat rows: {"game","experiment","task_id"}
    - clemcore instances_filter rows: {"experiment": {...}, "game_instance": {...}}
    """
    # clemcore run(..., instances_filter=...) shape
    if isinstance(row, Mapping) and ("experiment" in row) and ("game_instance" in row):
        exp_obj = row.get("experiment")
        inst_obj = row.get("game_instance")
        exp_name = None
        if isinstance(exp_obj, Mapping):
            exp_name = exp_obj.get("name") or exp_obj.get("experiment")
        elif exp_obj is not None:
            exp_name = str(exp_obj)

        game_name = None
        task_val = None
        if isinstance(inst_obj, Mapping):
            game_name = (
                inst_obj.get("game")
                or inst_obj.get("game_name")
                or inst_obj.get("main_game")
            )
            task_val = (
                inst_obj.get("task_id")
                or inst_obj.get("game_id")
                or inst_obj.get("instance_id")
                or inst_obj.get("id")
            )
        try:
            task_int = int(task_val) if task_val is not None else None
        except Exception:
            task_int = None
        return (
            str(game_name) if game_name is not None else None,
            str(exp_name) if exp_name is not None else None,
            task_int,
        )

    # generic / flat rows
    game = _row_value(row, "game", "game_name", "main_game")
    experiment = _row_value(row, "experiment", "experiment_name", "exp_name")
    task_id = _row_value(
        row,
        "task_id",
        "game_id",
        "instance_id",
        "id",
        "instance_idx",
        "row_id",
    )
    try:
        task_int = int(task_id) if task_id is not None else None
    except Exception:
        task_int = None
    return (
        str(game) if game is not None else None,
        str(experiment) if experiment is not None else None,
        task_int,
    )


def _clem_run(game_selector, model_specs, gen_args: Dict, results_dir: Path, selector_fn=None):
    run_sig = inspect.signature(clem.run)
    base_kwargs = dict(gen_args=gen_args)
    if "results_dir_path" in run_sig.parameters:
        base_kwargs["results_dir_path"] = results_dir
    elif "results_dir" in run_sig.parameters:
        base_kwargs["results_dir"] = str(results_dir)
    repo_root = Path(__file__).resolve().parent.parent
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo_root)

        # No sub-selection requested: normal call.
        if selector_fn is None:
            return clem.run(game_selector, model_specs, **base_kwargs)

        def _adaptive_selector(*args):
            """
            Bridge selector callback shape differences across clemcore versions:
            - old style: selector(game_name, experiment_name) -> list[int]
            - newer style: instances_filter(row) -> bool
            """
            # Preferred/legacy style.
            if len(args) == 2:
                return selector_fn(args[0], args[1])

            # Row-filter style.
            if len(args) == 1:
                row = args[0]
                # If caller already passed a row-oriented selector, use it directly.
                try:
                    direct = selector_fn(row)
                    if isinstance(direct, bool):
                        return direct
                except TypeError:
                    pass

                game, experiment, task_id = _selector_row_triplet(row)
                if game is None or experiment is None or task_id is None:
                    return False

                selected = selector_fn(str(game), str(experiment))
                if isinstance(selected, (list, tuple, set)):
                    try:
                        selected_ids = {int(x) for x in selected}
                        return int(task_id) in selected_ids
                    except Exception:
                        return task_id in selected
                return bool(selected)

            # Unknown caller shape: fall back to direct dispatch.
            return selector_fn(*args)

        # Robust selector wiring across clemcore versions.
        # Some versions expose explicit names in the signature, others may rename selector/filter args.
        selector_candidates = ["sub_selector", "task_selector", "game_instance_filter", "instance_filter"]
        known_non_selector = {
            "game_selector",
            "game_selectors",
            "game",
            "model_selectors",
            "model_selector",
            "models",
            "gen_args",
            "results_dir",
            "results_dir_path",
        }
        for name, param in run_sig.parameters.items():
            lname = str(name).lower()
            if name in known_non_selector:
                continue
            if lname in known_non_selector:
                continue
            # Exclude positional core args like "game_selectors" / "model_selectors"
            # even if their names contain "selector".
            if "game" in lname and "selector" in lname:
                continue
            if "model" in lname and "selector" in lname:
                continue
            if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                continue
            if ("selector" in lname) or ("filter" in lname):
                selector_candidates.append(name)

        # Preserve order, remove duplicates.
        selector_candidates = list(dict.fromkeys(selector_candidates))
        first_error = None
        for selector_param in selector_candidates:
            kwargs = dict(base_kwargs)
            kwargs[selector_param] = _adaptive_selector
            try:
                print(f"[playpen eval] trying selector argument: {selector_param}")
                result = clem.run(game_selector, model_specs, **kwargs)
                print(f"[playpen eval] using selector argument: {selector_param}")
                return result
            except TypeError as e:
                msg = str(e)
                # Keep probing only for clear "unexpected kwarg" cases.
                if ("unexpected keyword argument" in msg) and (selector_param in msg):
                    if first_error is None:
                        first_error = e
                    continue
                raise

        raise RuntimeError(
            "Dataset task sub-selection requested, but no known selector argument was accepted by clem.run(). "
            "Tried: sub_selector, task_selector, game_instance_filter, instance_filter, instances_filter."
        ) from first_error
    finally:
        os.chdir(previous_cwd)


def _with_temp_model_registry_entry(entry: Dict) -> Optional[callable]:
    registry_path = Path(
        os.getenv(
            "PLAYPEN_MODEL_REGISTRY_RUNTIME_PATH",
            str(Path(__file__).resolve().parent.parent / "model_registry.json"),
        )
    ).expanduser()
    if not registry_path.exists() or not registry_path.is_file():
        return None
    lock_path = Path(f"{registry_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
    try:
        original_text = registry_path.read_text(encoding="utf-8")
        payload = json.loads(original_text)
    except Exception:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fh.close()
        return None
    if not isinstance(payload, list):
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fh.close()
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
        try:
            registry_path.write_text(original_text, encoding="utf-8")
        finally:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            finally:
                lock_fh.close()

    return restore


def _with_model_registry_file(model_registry_file: Path) -> Optional[callable]:
    requested = Path(model_registry_file).expanduser()
    target = Path(
        os.getenv(
            "PLAYPEN_MODEL_REGISTRY_RUNTIME_PATH",
            str(Path(__file__).resolve().parent.parent / "model_registry.json"),
        )
    ).expanduser()

    if not requested.exists() or not requested.is_file():
        raise FileNotFoundError(f"Model registry file not found: {requested}")

    try:
        if requested.resolve() == target.resolve():
            return None
    except Exception:
        if str(requested) == str(target):
            return None

    lock_path = Path(f"{target}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

    source_text = requested.read_text(encoding="utf-8")
    payload = json.loads(source_text)
    if not isinstance(payload, (list, dict)):
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        lock_fh.close()
        raise ValueError(
            f"Model registry file must contain a JSON object or list, got {type(payload).__name__}: {requested}"
        )
    normalized_text = json.dumps(payload, indent=2)

    had_target = target.exists() and target.is_file()
    original_text = target.read_text(encoding="utf-8") if had_target else None
    target.write_text(normalized_text, encoding="utf-8")

    def restore():
        try:
            if had_target:
                target.write_text(original_text, encoding="utf-8")
            else:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
        finally:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            finally:
                lock_fh.close()

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
                specs = payload if isinstance(payload, list) else [payload]
                for spec in specs:
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


_INSTANCE_JSON_CACHE: Dict[str, Any] = {}
_RUNTIME_PROMPT_CACHE: Dict[Tuple[str, str, int], str] = {}


def _load_json_cached(path: Path):
    key = str(path)
    if key in _INSTANCE_JSON_CACHE:
        return _INSTANCE_JSON_CACHE[key]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    _INSTANCE_JSON_CACHE[key] = payload
    return payload


def _stringify_prompt_context(x: Any, *, max_chars: int = 4000) -> str:
    parts: List[str] = []

    def walk(v: Any):
        if len("\n".join(parts)) >= max_chars:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                parts.append(s)
            return
        if isinstance(v, Mapping):
            for k in ("prompt", "instruction", "content", "text", "message", "role", "system"):
                if k in v:
                    walk(v.get(k))
            for vv in v.values():
                walk(vv)
            return
        if isinstance(v, (list, tuple)):
            for it in v:
                walk(it)
            return
        try:
            s = str(v).strip()
            if s and s != "None":
                parts.append(s)
        except Exception:
            return

    walk(x)
    out: List[str] = []
    seen = set()
    total = 0
    for p in parts:
        key = p[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        total += len(p)
        if total >= max_chars:
            break
    return "\n".join(out)[:max_chars]


def _runtime_initial_prompt_for_task(game: str, experiment: Optional[str], task_id: Optional[int]) -> str:
    if not game or task_id is None:
        return ""
    try:
        tid = int(task_id)
    except Exception:
        return ""
    exp = str(experiment) if experiment is not None else ""
    cache_key = (str(game), exp, int(tid))
    if cache_key in _RUNTIME_PROMPT_CACHE:
        return _RUNTIME_PROMPT_CACHE[cache_key]

    class _CaptureModel:
        name = "playpen-prompt-capture-dummy"
        captured_prompt = ""

        def reset(self):
            return None

        def generate_response(self, messages, *args, **kwargs):
            try:
                self.captured_prompt = _stringify_prompt_context(messages, max_chars=4000)
            except Exception:
                self.captured_prompt = ""
            return ("", {}, "")

    prompt_text = ""
    debug_runtime = str(os.getenv("PLAYPEN_RUNTIME_PROMPT_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}
    max_turns = max(1, int(str(os.getenv("PLAYPEN_RUNTIME_PROMPT_MAX_TURNS", "8")).strip() or "8"))
    try:
        from clemcore.clemgame import GameRegistry
        from clemcore.clemgame.instances import GameInstances
        from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv
        from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
        from clemcore.clemgame.benchmark import GameBenchmark

        game_registry = GameRegistry.from_directories_and_cwd_files()
        game_specs = game_registry.get_game_specs_that_unify_with(str(game), verbose=False)
        if not game_specs:
            _RUNTIME_PROMPT_CACHE[cache_key] = ""
            return ""
        game_spec = game_specs[0]

        with GameBenchmark.load_from_spec(game_spec) as game_benchmark:
            game_instances = GameInstances.from_game_spec(game_spec)

            def _row_match_strict(row):
                exp_obj = row.get("experiment")
                inst_obj = row.get("game_instance")
                row_exp = None
                if isinstance(exp_obj, Mapping):
                    row_exp = exp_obj.get("name") or exp_obj.get("experiment")
                row_tid = None
                if isinstance(inst_obj, Mapping):
                    row_tid = (
                        inst_obj.get("task_id")
                        or inst_obj.get("game_id")
                        or inst_obj.get("instance_id")
                        or inst_obj.get("id")
                    )
                try:
                    row_tid = int(row_tid) if row_tid is not None else None
                except Exception:
                    row_tid = None
                if row_tid != tid:
                    return False
                if exp and row_exp is not None:
                    return _norm_key(row_exp) == _norm_key(exp)
                return True

            def _row_match_task_only(row):
                inst_obj = row.get("game_instance")
                row_tid = None
                if isinstance(inst_obj, Mapping):
                    row_tid = (
                        inst_obj.get("task_id")
                        or inst_obj.get("game_id")
                        or inst_obj.get("instance_id")
                        or inst_obj.get("id")
                    )
                try:
                    row_tid = int(row_tid) if row_tid is not None else None
                except Exception:
                    row_tid = None
                return row_tid == tid

            filtered = game_instances.filter(_row_match_strict)
            if len(filtered) <= 0:
                filtered = game_instances.filter(_row_match_task_only)
            if len(filtered) <= 0:
                _RUNTIME_PROMPT_CACHE[cache_key] = ""
                return ""
            row = next(iter(filtered), None)
            if row is None:
                _RUNTIME_PROMPT_CACHE[cache_key] = ""
                return ""

            callbacks = GameBenchmarkCallbackList([])
            cap_model = _CaptureModel()
            env = GameMasterEnv(game_benchmark, callbacks=callbacks)
            try:
                env.reset(
                    options={
                        "player_models": [cap_model],
                        "experiment": row["experiment"],
                        "game_instance": row["game_instance"],
                    }
                )
                turns = 0
                for agent_id in env.agent_iter():
                    turns += 1
                    if turns > max_turns:
                        break
                    context, reward, termination, truncation, info = env.last(observe=True)
                    # Candidate from observed context even before model call.
                    candidate = _stringify_prompt_context(context, max_chars=4000)
                    if candidate:
                        prompt_text = candidate
                    if termination or truncation:
                        env.step(None)
                        if prompt_text:
                            break
                        continue
                    response = None
                    try:
                        player = env.player_by_agent_id[agent_id]
                        response = player(context)
                    except Exception:
                        # Even if gameplay fails, we may already have captured the outgoing prompt payload.
                        pass
                    if cap_model.captured_prompt:
                        prompt_text = cap_model.captured_prompt
                    env.step(response)
                    if prompt_text:
                        break
            finally:
                env.close()
    except Exception as e:
        if debug_runtime:
            print(
                f"[playpen runtime-prompt] failed game={game} exp={exp} task_id={tid}: {type(e).__name__}: {e}",
                flush=True,
            )
        prompt_text = ""

    _RUNTIME_PROMPT_CACHE[cache_key] = prompt_text or ""
    return _RUNTIME_PROMPT_CACHE[cache_key]


def _candidate_instance_paths(roots: Tuple[Path, ...], game_spec: Dict) -> List[Path]:
    main_game = str(game_spec.get("main_game") or game_spec.get("game_name") or "").strip()
    game_name = str(game_spec.get("game_name") or "").strip()
    instances = game_spec.get("instances") or "instances"
    instances_file = str(instances)
    if not instances_file.endswith(".json"):
        instances_file = instances_file + ".json"

    candidates: List[Path] = []
    for root in roots:
        for g in (main_game, game_name):
            if not g:
                continue
            candidates.append(root / g / "in" / instances_file)
            candidates.append(root / g / "in" / "instances.json")
            # Some benchmarks nest games under an intermediate folder (e.g. textmapworld/*).
            try:
                candidates.extend(root.glob(f"**/{g}/in/{instances_file}"))
                candidates.extend(root.glob(f"**/{g}/in/instances.json"))
            except Exception:
                pass

    dedup: List[Path] = []
    seen = set()
    for p in candidates:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        dedup.append(p)
    return dedup


def _instance_prompt_text_for_task(
    game: str,
    experiment: Optional[str],
    task_id: Optional[int],
    *,
    meta_index: Dict[str, Dict],
    roots: Tuple[Path, ...],
    max_chars: int = 4000,
) -> str:
    spec = meta_index.get(game) or {}
    if not spec:
        return ""

    exp_norm = _norm_key(experiment) if experiment is not None else ""
    for p in _candidate_instance_paths(roots, spec):
        if not (p.exists() and p.is_file()):
            continue
        payload = _load_json_cached(p)
        if not isinstance(payload, dict):
            continue
        experiments = payload.get("experiments")
        if not isinstance(experiments, list):
            continue

        exp_obj = None
        if exp_norm:
            for e in experiments:
                if not isinstance(e, dict):
                    continue
                if _norm_key(e.get("name")) == exp_norm:
                    exp_obj = e
                    break
        if exp_obj is None and experiments:
            exp_obj = experiments[0] if isinstance(experiments[0], dict) else None
        if not isinstance(exp_obj, dict):
            continue

        # Collect prompt-like fields from experiment-level config (without game_instances bulk).
        exp_shallow = {k: v for k, v in exp_obj.items() if k != "game_instances"}
        texts = _collect_prompt_strings(exp_shallow, max_chars=max_chars)

        # Add prompt-like fields from the specific task instance when available.
        if task_id is not None:
            gi = exp_obj.get("game_instances")
            if isinstance(gi, list):
                chosen = None
                for inst in gi:
                    if not isinstance(inst, dict):
                        continue
                    inst_id = inst.get("game_id")
                    try:
                        inst_id_int = int(inst_id)
                    except Exception:
                        inst_id_int = None
                    if inst_id_int is not None and inst_id_int == int(task_id):
                        chosen = inst
                        break
                if isinstance(chosen, dict):
                    texts.extend(_collect_prompt_strings(chosen, max_chars=max_chars))

        # Dedup + cap
        out: List[str] = []
        seen = set()
        total = 0
        for t in texts:
            key = t[:200]
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            total += len(t)
            if total >= max_chars:
                break
        if out:
            return "\n---\n".join(out)
    return ""


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


def _order_games_with_last(games: List[str]) -> List[str]:
    # Optional: force one or more games to the end while keeping stable order otherwise.
    raw = str(os.getenv("PLAYPEN_EVAL_GAMES_LAST", "")).strip()
    if not raw:
        return games
    last = [x.strip() for x in raw.split(",") if x.strip()]
    if not last:
        return games
    last_set = set(last)
    front = [g for g in games if g not in last_set]
    tail = [g for g in games if g in last_set]
    return front + tail


def _exclude_games(games: List[str]) -> List[str]:
    raw = str(os.getenv("PLAYPEN_EVAL_GAMES_EXCLUDE", "")).strip()
    if not raw:
        return games
    excluded = {x.strip() for x in raw.split(",") if x.strip()}
    if not excluded:
        return games
    return [g for g in games if g not in excluded]


def _apply_game_filters(games: List[str]) -> List[str]:
    return _order_games_with_last(_exclude_games(games))


def _is_truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _load_router_replay_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                item = json.loads(s)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


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
        dataset = _load_eval_dataset(dataset_name)
        tasks_by_group = tasks_by_game_experiment(dataset)
        explicit_game = None
        if isinstance(game_selector, str):
            selector = game_selector.strip()
            if selector and not selector.startswith("{") and not selector.startswith("["):
                explicit_game = selector

        if explicit_game is not None:
            target_games = sorted({g for (g, _e) in tasks_by_group.keys() if g == explicit_game})
        else:
            target_games = sorted({g for (g, _e) in tasks_by_group.keys()})
        target_games = _apply_game_filters(target_games)

        for game_name in target_games:
            by_experiment = {
                str(exp): {int(x) for x in ids}
                for (g, exp), ids in tasks_by_group.items()
                if g == game_name
            }
            if not by_experiment:
                continue

            by_experiment_norm = {_norm_key(exp): ids for exp, ids in by_experiment.items() if _norm_key(exp)}
            ids_union = set()
            for ids in by_experiment.values():
                ids_union.update(ids)
            def sub_selector_exact(*args, _g=game_name, _by_exp=by_experiment, _by_exp_norm=by_experiment_norm, _ids_union=ids_union):
                # Legacy clemcore style: selector(game, experiment) -> list[int]
                if len(args) == 2:
                    game, experiment = str(args[0]), str(args[1])
                    if game != _g:
                        return []
                    if experiment in _by_exp:
                        return sorted(_by_exp[experiment])
                    exp_norm = _norm_key(experiment)
                    if exp_norm in _by_exp_norm:
                        return sorted(_by_exp_norm[exp_norm])
                    return []

                # Newer clemcore style: instances_filter(row) -> bool
                if len(args) == 1:
                    row = args[0]
                    game_val, exp_val, task_val = _selector_row_triplet(row)

                    if game_val is not None and str(game_val) != _g:
                        return False
                    if task_val is None:
                        return False

                    candidate_ids = _ids_union
                    if exp_val is not None:
                        exp_key = str(exp_val)
                        if exp_key in _by_exp:
                            candidate_ids = _by_exp[exp_key]
                        else:
                            exp_norm = _norm_key(exp_key)
                            if exp_norm in _by_exp_norm:
                                candidate_ids = _by_exp_norm[exp_norm]
                    try:
                        return int(task_val) in candidate_ids
                    except Exception:
                        return task_val in candidate_ids

                return False

            with _eval_context_env(game=game_name, split="validation", regime="unknown"):
                _clem_run(game_name, [model_spec], gen_args, suite_results_dir, selector_fn=sub_selector_exact)
    clem.score(game_selector, str(suite_results_dir))
    clem.transcripts(game_selector, str(suite_results_dir))
    try:
        df = clem.clemeval.perform_evaluation(str(suite_results_dir), return_dataframe=True)
        return _extract_clemscore(df, model_spec.model_name)
    except KeyError as e:
        # Happens when all instances abort and clemcore cannot build the "all" aggregate column.
        print(
            f"[playpen eval] warning: clemeval missing aggregate columns for {model_spec.model_name} "
            f"(suite={suite}); returning clemscore=0.0. details={e}",
            flush=True,
        )
        return 0.0


def _extract_clemscore(df, model_name: str) -> float:
    # clemcore returns a DataFrame indexed by model name in typical usage, but keep this robust.
    def _coerce(v) -> float:
        out = float(v)
        # NaN check without extra deps
        return 0.0 if out != out else out

    try:
        return _coerce(df.loc[model_name, "-, clemscore"])
    except Exception:
        try:
            return _coerce(df["-, clemscore"].iloc[0])
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
    debug_loss_prompts = str(os.getenv("PLAYPEN_LOSS_ROUTER_DEBUG_PROMPTS", "0")).strip().lower() in {"1", "true", "yes", "on"}
    debug_loss_limit = max(0, int(str(os.getenv("PLAYPEN_LOSS_ROUTER_DEBUG_PROMPTS_LIMIT", "12")).strip() or "12"))
    debug_loss_chars = max(80, int(str(os.getenv("PLAYPEN_LOSS_ROUTER_DEBUG_PROMPT_CHARS", "1200")).strip() or "1200"))
    debug_loss_printed = 0

    routing_manifest = {
        "moe_name": moe.name,
        "default_model": moe.default_model,
        "route_by_experiment": moe.route_by_experiment,
        "routes": [r.__dict__ for r in moe.routes],
        "assignments": [],
    }

    if dataset_name is not None:
        dataset = _load_eval_dataset(dataset_name)
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
        game_names = _apply_game_filters(game_names)
        game_name_set = set(game_names)

        model_registry = ModelRegistry.from_packaged_and_cwd_files()
        def sub_selector_all(game: str, experiment: str):
            return tasks_by_group.get((game, experiment), [])

        replay_mode = _is_truthy_env("PLAYPEN_LOSS_ROUTER_REPLAY", "0")
        replay_default = os.getenv("PLAYPEN_LOSS_ROUTER_REPLAY_JSONL")
        replay_suite_key = "PLAYPEN_LOSS_ROUTER_REPLAY_CLEM_JSONL" if suite == "clem" else "PLAYPEN_LOSS_ROUTER_REPLAY_STATIC_JSONL"
        replay_path_raw = os.getenv(replay_suite_key, replay_default)
        replay_rows: List[Dict[str, Any]] = []
        if replay_mode:
            if not replay_path_raw:
                raise ValueError(
                    f"PLAYPEN_LOSS_ROUTER_REPLAY=1 but no replay file configured. "
                    f"Set {replay_suite_key} or PLAYPEN_LOSS_ROUTER_REPLAY_JSONL."
                )
            replay_path = Path(replay_path_raw).expanduser()
            if not replay_path.exists():
                raise FileNotFoundError(f"Replay JSONL not found: {replay_path}")
            replay_rows = _load_router_replay_jsonl(replay_path)
            print(
                f"[MoE-LossRouter] replay mode enabled for suite={suite}; "
                f"loaded rows={len(replay_rows)} from {replay_path}",
                flush=True,
            )

        configured_experts = list((moe.loss_router.experts if moe.loss_router is not None else ()) or [])
        if not configured_experts:
            configured_experts = sorted({r.model for r in moe.routes if getattr(r, "model", None)})
        if replay_mode and not configured_experts:
            configured_experts = sorted(
                {
                    str(r.get("selected_expert"))
                    for r in replay_rows
                    if isinstance(r, Mapping) and r.get("selected_expert")
                }
            )
        if not configured_experts:
            raise ValueError(
                "MoE requires adapter experts for loss-based routing. "
                "Configure `loss_router.experts` or route models."
            )
        configured_experts = [str(x) for x in configured_experts if str(x).strip()]
        if str(moe.default_model) in set(configured_experts):
            raise ValueError(
                f"MoE experts include default/base model '{moe.default_model}', which is disallowed."
            )

        loss_router = LossBasedExpertRouter(
            model_registry=model_registry,
            base_model_name=moe.default_model,
            experts=configured_experts,
        )
        router_log_path = (
            Path(moe.loss_router.log_path).expanduser()
            if (moe.loss_router is not None and moe.loss_router.log_path)
            else (results_dir / f"{moe.name}.{suite}.loss_router.jsonl")
        )
        oracle_field = str(moe.loss_router.oracle_field or "oracle_expert") if moe.loss_router is not None else "oracle_expert"
        id_regimes = tuple(moe.loss_router.id_regimes or ("id",)) if moe.loss_router is not None else ("id",)
        expert_to_cluster = dict(moe.loss_router.expert_to_cluster or {}) if moe.loss_router is not None else {}
        game_to_cluster = dict(moe.loss_router.game_to_cluster or {}) if moe.loss_router is not None else {}
        loss_inference_mode = str(
            (moe.loss_router.inference_mode if moe.loss_router is not None else "legacy") or "legacy"
        ).strip().lower()
        loss_routing_scope = str(
            (moe.loss_router.routing_scope if moe.loss_router is not None else "game") or "game"
        ).strip().lower()
        if loss_routing_scope not in {"game", "task"}:
            raise ValueError(
                f"Unsupported loss_router routing_scope '{loss_routing_scope}'. Use 'game' or 'task'."
            )
        if loss_inference_mode not in {"legacy", "grouped_expert"}:
            raise ValueError(
                f"Unsupported loss_router inference_mode '{loss_inference_mode}'. "
                "Use 'legacy' or 'grouped_expert'."
            )
        merge_label = merge if merge is not None else "none"
        grouped_assignments_by_expert: Dict[str, Dict[str, List[Tuple[Optional[str], int]]]] = defaultdict(lambda: defaultdict(list))

        if replay_mode and loss_inference_mode != "grouped_expert":
            raise ValueError(
                "Replay mode currently supports grouped_expert inference only. "
                "Set loss_router.inference_mode='grouped_expert'."
            )

        rows_by_game: Dict[str, List[Mapping[str, Any]]] = {}
        for row in dataset:
            game_name = str(_row_get(row, "game", "game_name", default="") or "")
            task_id = _row_get(row, "task_id", "game_id", "instance_id", default=None)
            if not game_name or task_id is None:
                continue
            if explicit_game is not None and game_name != explicit_game:
                continue
            if game_name not in game_name_set:
                continue
            rows_by_game.setdefault(game_name, []).append(row)

        if replay_mode:
            assigned = 0
            for rr in replay_rows:
                game_name = str(rr.get("game") or "").strip()
                selected_expert = rr.get("selected_expert")
                task_id = rr.get("task_id")
                experiment = rr.get("experiment")
                if not game_name or selected_expert is None or task_id is None:
                    continue
                if explicit_game is not None and game_name != explicit_game:
                    continue
                if game_name not in game_name_set:
                    continue
                try:
                    tid = int(task_id)
                except Exception:
                    continue
                expert = str(selected_expert)
                grouped_assignments_by_expert[expert][game_name].append(
                    ((str(experiment) if experiment is not None else None), tid)
                )
                routing_manifest["assignments"].append(
                    {
                        "game": game_name,
                        "experiment": str(experiment) if experiment is not None else None,
                        "task_id": tid,
                        "expert_model": expert,
                        "router": "loss_router_replay",
                        "routing_scope": loss_routing_scope,
                        "representative_task_id": tid,
                    }
                )
                assigned += 1
            print(
                f"[MoE-LossRouter] replay assignments accepted: {assigned}",
                flush=True,
            )
        else:
            routing_units: List[Tuple[str, List[Mapping[str, Any]]]] = []
            if loss_routing_scope == "game":
                routing_units = [(g, rows_by_game[g]) for g in sorted(rows_by_game.keys())]
            else:
                for g in sorted(rows_by_game.keys()):
                    for r in rows_by_game[g]:
                        routing_units.append((g, [r]))

            total_units = len(routing_units)
            for unit_idx, (game_name, game_rows) in enumerate(routing_units, start=1):
                if not game_rows:
                    continue
                print(
                    f"[MoE-LossRouter] routing {loss_routing_scope} {unit_idx}/{total_units}: {game_name} tasks={len(game_rows)}",
                    flush=True,
                )

                def _task_id_value(r):
                    return int(_row_get(r, "task_id", "game_id", "instance_id", default=10**9))

                representative = min(game_rows, key=_task_id_value)
                rep_task_id = _task_id_value(representative)
                rep_experiment = _row_get(representative, "experiment", "experiment_name", default=None)
                rep_experiment = str(rep_experiment) if rep_experiment is not None else None

                prompt_text, target_text = extract_prompt_and_first_target(representative)
                runtime_prompt_text = _runtime_initial_prompt_for_task(
                    game=str(game_name),
                    experiment=rep_experiment,
                    task_id=rep_task_id,
                ).strip()
                if runtime_prompt_text:
                    prompt_text = runtime_prompt_text
                if not prompt_text or not target_text:
                    context_text = _game_context_text(game_name, rep_experiment, game_meta_index)
                    if not prompt_text:
                        prompt_text = (context_text or f"{game_name} {rep_experiment or ''} task_id={rep_task_id}").strip()
                    if debug_loss_prompts and debug_loss_printed < debug_loss_limit:
                        prompt_preview = (prompt_text or "")[:debug_loss_chars].replace("\n", " ")
                        print(
                            f"[MoE-LossRouter][prompt-debug] mode=fallback_generate_then_loss game={game_name} "
                            f"exp={rep_experiment} task={rep_task_id} prompt_len={len(prompt_text or '')} "
                            f"target_len={len(target_text or '')} prompt={prompt_preview}",
                            flush=True,
                        )
                        debug_loss_printed += 1
                    print(
                        f"[MoE-LossRouter] warning: missing target text; routing from expert-generated initial answers "
                        f"(game={game_name}, experiment={rep_experiment}, task_id={rep_task_id})"
                    , flush=True)
                    scored = loss_router.select_expert_from_prompt(
                        prompt_text=prompt_text,
                        max_new_tokens=max(8, min(int(gen_args.get("max_tokens", 300) or 300), 96)),
                        temperature=float(gen_args.get("temperature", 0.0) or 0.0),
                    )
                else:
                    if debug_loss_prompts and debug_loss_printed < debug_loss_limit:
                        prompt_preview = (prompt_text or "")[:debug_loss_chars].replace("\n", " ")
                        target_preview = (target_text or "")[:debug_loss_chars].replace("\n", " ")
                        print(
                            f"[MoE-LossRouter][prompt-debug] mode=direct_target_loss game={game_name} "
                            f"exp={rep_experiment} task={rep_task_id} prompt_len={len(prompt_text or '')} "
                            f"target_len={len(target_text or '')} prompt={prompt_preview} target={target_preview}",
                            flush=True,
                        )
                        debug_loss_printed += 1
                    scored = loss_router.select_expert_by_loss(prompt_text=prompt_text, target_text=target_text)
                expert = str(scored["selected_expert"])
                if expert == str(moe.default_model):
                    raise ValueError(
                        f"MoE selected default/base model '{moe.default_model}', which is disallowed."
                    )

                expert_spec = _resolve_registered_model_spec(model_registry, expert)
                run_spec = _with_model_name(expert_spec, moe.name)
                extra_cfg = {"moe_expert": expert}
                if merge is not None:
                    extra_cfg["merge"] = merge
                run_spec = _with_model_config(run_spec, extra_cfg)

                regime = _regime_from_row(representative, id_regimes)
                if loss_inference_mode == "legacy":
                    restore_registry = _with_temp_model_registry_entry(run_spec.to_dict())
                    if restore_registry is not None:
                        print(f"[MoE] injected temporary model_registry entry for {moe.name}", flush=True)
                    try:
                        def sub_selector_game(game: str, experiment: str, _g=game_name):
                            if game != _g:
                                return []
                            return tasks_by_group.get((game, experiment), [])

                        t0_run = time.time()
                        print(f"[MoE-LossRouter] _clem_run start game={game_name} expert={expert}", flush=True)
                        with _eval_context_env(game=game_name, split="validation", regime=regime):
                            _clem_run(game_name, [run_spec], gen_args, suite_results_dir, selector_fn=sub_selector_game)
                        print(
                            f"[MoE-LossRouter] _clem_run done game={game_name} expert={expert} "
                            f"elapsed={time.time() - t0_run:.1f}s",
                            flush=True,
                        )
                    finally:
                        if restore_registry is not None:
                            restore_registry()

                example_id = str(_row_get(representative, "example_id", default=f"{game_name}:{rep_experiment}:{rep_task_id}"))
                oracle_expert = _row_get(representative, oracle_field, "oracle_expert", default=None)
                details = list(scored.get("details") or [])
                score_map = {
                    str(item["expert_name"]): float(item["mean_nll"])
                    for item in details
                    if isinstance(item, Mapping) and item.get("expert_name") is not None
                }
                selected_mean_nll = score_map.get(expert)
                margin = float(scored.get("margin_to_second_best", float("nan")))
                semantic_oracle_cluster = game_to_cluster.get(str(game_name))
                selected_cluster = expert_to_cluster.get(expert)
                semantic_oracle_expert = None
                semantic_oracle_nll = None
                semantic_regret = None
                semantic_match = None
                if semantic_oracle_cluster:
                    for exp_name in configured_experts:
                        if expert_to_cluster.get(str(exp_name)) == semantic_oracle_cluster:
                            semantic_oracle_expert = str(exp_name)
                            break
                    if semantic_oracle_expert is not None:
                        semantic_oracle_nll = score_map.get(semantic_oracle_expert)
                    if selected_cluster is not None:
                        semantic_match = bool(str(selected_cluster) == str(semantic_oracle_cluster))
                    if selected_mean_nll is not None and semantic_oracle_nll is not None:
                        semantic_regret = float(selected_mean_nll) - float(semantic_oracle_nll)

                log_row = {
                    "example_id": example_id,
                    "game": game_name,
                    "experiment": rep_experiment,
                    "task_id": int(rep_task_id),
                    "split": "validation",
                    "regime": regime,
                    "selected_expert": expert,
                    "oracle_expert": str(oracle_expert) if oracle_expert is not None else None,
                    "top1_minus_top2_margin": margin,
                    "selected_mean_nll": selected_mean_nll,
                    "selected_cluster": selected_cluster,
                    "semantic_oracle_cluster": semantic_oracle_cluster,
                    "semantic_oracle_expert": semantic_oracle_expert,
                    "semantic_oracle_mean_nll": semantic_oracle_nll,
                    "semantic_cluster_match": semantic_match,
                    "semantic_regret_vs_cluster_oracle": semantic_regret,
                    "final_evaluation_score": None,
                    "routing_scope": loss_routing_scope,
                }
                for expert_name in configured_experts:
                    log_row[f"mean_nll_{expert_name}"] = score_map.get(expert_name)
                append_router_log_row(router_log_path, log_row)

                print(
                    f"[MoE-LossRouter] suite={suite} game={game_name} rep_task_id={rep_task_id} "
                    f"selected={expert} margin={margin:.6f} merge={merge_label} tasks={len(game_rows)}"
                , flush=True)

                moe_info = {
                    "moe_name": moe.name,
                    "moe_type": "loss_router",
                    "default_model": moe.default_model,
                    "expert_model": expert,
                    "game": game_name,
                    "experiment": rep_experiment,
                    "task_id": int(rep_task_id),
                    "merge": merge_label,
                    "routing_scope": loss_routing_scope,
                }
                updated = _update_players_model_jsons(suite_results_dir, game_name, moe_info)
                if updated == 0:
                    print(f"[MoE] warning: no players_model.json found under {suite_results_dir} for game={game_name}", flush=True)

                for row in game_rows:
                    row_game, row_exp, row_tid = _selector_row_triplet(row)
                    if row_tid is None:
                        continue
                    # Fall back to routed game_name if row-level game is not available.
                    row_game = row_game or game_name
                    if loss_inference_mode == "grouped_expert":
                        grouped_assignments_by_expert[expert][row_game].append(
                            ((str(row_exp) if row_exp is not None else None), int(row_tid))
                        )
                    routing_manifest["assignments"].append(
                        {
                            "game": str(row_game),
                            "experiment": str(row_exp) if row_exp is not None else None,
                            "task_id": int(row_tid),
                            "expert_model": expert,
                            "router": "loss_router",
                            "routing_scope": loss_routing_scope,
                            "representative_task_id": int(rep_task_id),
                        }
                    )
        if loss_inference_mode == "grouped_expert":
            total_grouped_tasks = sum(
                len(items)
                for by_game in grouped_assignments_by_expert.values()
                for items in by_game.values()
            )
            print(
                f"[MoE-LossRouter] grouped_expert enabled: experts={len(grouped_assignments_by_expert)} "
                f"tasks={total_grouped_tasks} suite={suite}"
            , flush=True)
            for expert, by_game in sorted(grouped_assignments_by_expert.items()):
                total_for_expert = sum(len(v) for v in by_game.values())
                if total_for_expert <= 0:
                    continue
                games_in_group = sorted(by_game.keys())
                print(f"[MoE-LossRouter] running expert={expert} grouped_tasks={total_for_expert}", flush=True)
                print(f"[MoE-LossRouter] expert={expert} games={games_in_group}", flush=True)

                expert_spec = _resolve_registered_model_spec(model_registry, expert)
                skip_temp_registry = str(os.getenv("PLAYPEN_MOE_GROUPED_SKIP_TEMP_REGISTRY", "1")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if skip_temp_registry:
                    # Keep registry-resolvable model name when we do not inject temp registry entries.
                    run_spec = expert_spec
                else:
                    run_spec = _with_model_name(expert_spec, moe.name)

                extra_cfg = {"moe_expert": expert}
                if merge is not None:
                    extra_cfg["merge"] = merge
                run_spec = _with_model_config(run_spec, extra_cfg)

                restore_registry = None
                if not skip_temp_registry:
                    restore_registry = _with_temp_model_registry_entry(run_spec.to_dict())
                    if restore_registry is not None:
                        print(f"[MoE] injected temporary model_registry entry for {moe.name}", flush=True)
                else:
                    print("[MoE] grouped_expert: skipping temporary model_registry injection", flush=True)
                try:
                    total_games_for_expert = len(games_in_group)
                    for game_idx, game_name in enumerate(games_in_group, start=1):
                        exp_tid_list = by_game.get(game_name) or []
                        if not exp_tid_list:
                            continue
                        selected = {
                            (str(game_name), (str(e) if e is not None else None), int(t))
                            for (e, t) in exp_tid_list
                        }
                        if not selected:
                            continue
                        print(
                            f"[MoE-LossRouter] grouped batch {game_idx}/{total_games_for_expert} "
                            f"expert={expert} game={game_name} tasks={len(selected)}",
                            flush=True,
                        )

                        # Support both clemcore selector conventions:
                        # 1) row-filter: selector(row) -> bool
                        # 2) legacy: selector(game_name, experiment_name) -> list[int]
                        selected_by_exp = defaultdict(set)
                        selected_by_exp_norm = defaultdict(set)
                        for (gg, ee, tt) in selected:
                            selected_by_exp[(gg, ee)].add(int(tt))
                            selected_by_exp_norm[(gg, _norm_key(ee))].add(int(tt))

                        selected_task_ids = {int(tt) for (_gg, _ee, tt) in selected}

                        def sub_selector_rows(
                            *args,
                            _selected=selected,
                            _by=selected_by_exp,
                            _by_norm=selected_by_exp_norm,
                            _game_name=str(game_name),
                            _selected_task_ids=selected_task_ids,
                        ):
                            if len(args) == 1:
                                row = args[0]
                                game, experiment, task_id = _selector_row_triplet(row)
                                if task_id is None:
                                    return False
                                # clemcore row-shapes may omit game/experiment in row-filter callbacks.
                                g = str(game) if game is not None else _game_name
                                e = str(experiment) if experiment is not None else None
                                tid = int(task_id)
                                key = (g, e, tid)
                                if key in _selected:
                                    return True
                                # Fallback only when experiment is missing; otherwise do not
                                # widen selection across all experiments sharing this task_id.
                                if e is None:
                                    return (g == _game_name) and (tid in _selected_task_ids)
                                return False
                            if len(args) == 2:
                                game_name_arg, experiment_name_arg = args
                                g = str(game_name_arg) if game_name_arg is not None else None
                                if isinstance(experiment_name_arg, Mapping):
                                    exp_raw = (
                                        experiment_name_arg.get("name")
                                        or experiment_name_arg.get("experiment")
                                        or experiment_name_arg.get("experiment_name")
                                    )
                                else:
                                    exp_raw = experiment_name_arg
                                e = str(exp_raw) if exp_raw is not None else None
                                if (g, e) in _by:
                                    return sorted(_by[(g, e)])
                                e_norm = _norm_key(e)
                                if (g, e_norm) in _by_norm:
                                    return sorted(_by_norm[(g, e_norm)])
                                return []
                            return False

                        t0_run = time.time()
                        print(
                            f"[MoE-LossRouter] _clem_run start grouped expert={expert} game={game_name}",
                            flush=True,
                        )
                        with _eval_context_env(game=game_name, split="validation", regime="unknown"):
                            _clem_run(game_name, [run_spec], gen_args, suite_results_dir, selector_fn=sub_selector_rows)
                        print(
                            f"[MoE-LossRouter] _clem_run done grouped expert={expert} game={game_name} "
                            f"elapsed={time.time() - t0_run:.1f}s",
                            flush=True,
                        )
                finally:
                    if restore_registry is not None:
                        restore_registry()
        routing_manifest["loss_router_log"] = str(router_log_path)
        routing_manifest["loss_router_inference_mode"] = loss_inference_mode

        manifest_path = results_dir / f"{moe.name}.{suite}.moe.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(routing_manifest, f, indent=2)

    clem.score(game_selector, str(suite_results_dir))
    clem.transcripts(game_selector, str(suite_results_dir))
    try:
        df = clem.clemeval.perform_evaluation(str(suite_results_dir), return_dataframe=True)
        return _extract_clemscore(df, moe.name)
    except KeyError as e:
        print(
            f"[playpen eval] warning: clemeval missing aggregate columns for {moe.name} "
            f"(suite={suite}); returning clemscore=0.0. details={e}",
            flush=True,
        )
        return 0.0


def evaluate(suite: str, model_spec: ModelSpec, gen_args: Dict, results_dir: Path, game_selector: str,
             skip_gameplay: bool):
    overall_results_file = results_dir / f"{model_spec.model_name}.val.json"
    if suite in ["all", "clem"]:
        dataset_name = None if skip_gameplay else "instances"
        _game_selector = GameSpec.from_dict({"benchmark": ["3.0"]}, allow_underspecified=True) \
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
        _game_selector = GameSpec.from_dict({"benchmark": ["3.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        clem_score = evaluate_suite_moe("clem", moe, gen_args, results_dir, _game_selector, dataset_name, merge)
        store_eval_score(overall_results_file, "clemscore", clem_score)
    if suite in ["all", "static"]:
        dataset_name = None if skip_gameplay else "instances-static"
        _game_selector = GameSpec.from_dict({"benchmark": ["static_1.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        stat_score = evaluate_suite_moe("static", moe, gen_args, results_dir, _game_selector, dataset_name, merge)
        store_eval_score(overall_results_file, "statscore", stat_score)


def evaluate_suite_adapter_bar(
    suite: str,
    routing_mode: str,
    model_spec: ModelSpec,
    gen_args: Dict,
    results_dir: Path,
    game_selector,
    dataset_name: str,
    adapter_bar_router_path: str,
    adapter_bar_config: str,
    max_instances: Optional[int] = None,
):
    from playpen.adapter_bar_sequence import load_adapter_bar_mode_config

    router_type = "cluster" if routing_mode == "adapter_bar_sequence_cluster" else "game"
    bundle = load_router_bundle(adapter_bar_config, adapter_bar_router_path, router_type)
    cfg = load_adapter_bar_mode_config(adapter_bar_config, router_type)
    adapter_paths = dict(cfg.get("expert_adapter_paths") or {})
    loading_cfg = dict(cfg.get("adapter_loading") or cfg.get("adapter_loading", {}) or {})
    cache = AdapterCacheManager(
        adapter_paths=adapter_paths,
        preload_adapters=bool(loading_cfg.get("preload_adapters", False)),
        max_active_adapters=int(loading_cfg.get("max_active_adapters", 2)),
    )

    suite_results_dir = results_dir / suite
    ds = _load_eval_dataset(dataset_name) if dataset_name is not None else []
    rows = [dict(ds[i]) for i in range(len(ds))] if dataset_name is not None else []
    if max_instances is not None and max_instances > 0:
        rows = rows[: int(max_instances)]

    log_path = results_dir / "adapter_bar_routing_log.jsonl"
    usage = Counter()
    usage_by_game = defaultdict(Counter)
    confidence_vals = []
    correct = 0
    with_gold = 0
    game_gold = defaultdict(lambda: [0, 0])
    score_by_adapter = defaultdict(list)
    score_by_game = defaultdict(list)

    inference_mode = str(
        cfg.get(
            "inference_mode",
            cfg.get("adapter_bar_inference_mode", "grouped"),
        )
    ).strip().lower()
    if inference_mode not in {"legacy", "grouped"}:
        raise ValueError(f"Unsupported adapter_bar inference mode '{inference_mode}'. Use 'legacy' or 'grouped'.")

    model_registry = ModelRegistry.from_packaged_and_cwd_files()
    grouped_tasks = defaultdict(lambda: defaultdict(set))
    example_count = 0
    total_rows = len(rows)
    game_meta_index = _build_game_meta_index(_find_clembench_roots())
    roots = _find_clembench_roots()
    missing_prompt_count = 0
    prompt_source_counts = Counter()
    debug_prompts = str(os.getenv("PLAYPEN_ADAPTER_BAR_DEBUG_PROMPTS", "0")).strip().lower() in {"1", "true", "yes", "on"}
    debug_prompt_limit = max(0, int(str(os.getenv("PLAYPEN_ADAPTER_BAR_DEBUG_PROMPTS_LIMIT", "8")).strip() or "8"))
    debug_prompt_chars = max(80, int(str(os.getenv("PLAYPEN_ADAPTER_BAR_DEBUG_PROMPT_CHARS", "1200")).strip() or "1200"))
    debug_prompt_printed = 0
    verbose_steps = str(os.getenv("PLAYPEN_ADAPTER_BAR_VERBOSE_STEPS", "0")).strip().lower() in {"1", "true", "yes", "on"}
    for row in rows:
        game = get_game_name(row, training=False)
        exp_for_prompt = _row_get(row, "experiment", "experiment_name", default=None)
        tid_for_prompt = _row_get(row, "task_id", "game_id", "instance_id", default=None)
        try:
            tid_for_prompt = int(tid_for_prompt) if tid_for_prompt is not None else None
        except Exception:
            tid_for_prompt = None
        prompt = _runtime_initial_prompt_for_task(
            game=str(game or ""),
            experiment=(str(exp_for_prompt) if exp_for_prompt is not None else None),
            task_id=tid_for_prompt,
        ).strip()
        prompt_source = "runtime_initial_prompt_primary"
        if not prompt:
            prompt, _ = extract_prompt_and_first_target(row)
            prompt_source = "messages"
        if not prompt:
            prompt = str(_row_get(row, "prompt", "context", "instruction", default="") or "")
            if prompt:
                prompt_source = "row_fields"
        if not prompt:
            prompt = _instance_prompt_text_for_task(
                game=str(game or ""),
                experiment=(str(exp_for_prompt) if exp_for_prompt is not None else None),
                task_id=tid_for_prompt,
                meta_index=game_meta_index,
                roots=roots,
                max_chars=4000,
            ).strip()
            if prompt:
                prompt_source = "instance_prompt_fallback"
        if not prompt:
            prompt = _game_context_text(str(game or ""), (str(exp_for_prompt) if exp_for_prompt is not None else None), game_meta_index).strip()
            if prompt:
                prompt_source = "game_context_fallback"
        if not prompt:
            missing_prompt_count += 1
            # deterministic non-empty fallback to avoid degenerate constant routing from empty text
            g_fallback = str(_row_get(row, "game", "game_name", default="unknown_game") or "unknown_game")
            e_fallback = str(_row_get(row, "experiment", "experiment_name", default="unknown_exp") or "unknown_exp")
            t_fallback = _row_get(row, "task_id", "game_id", "instance_id", default="unknown_task")
            prompt = f"game={g_fallback}\nexperiment={e_fallback}\ntask_id={t_fallback}"
            prompt_source = "synthetic_fallback"
        prompt_source_counts[prompt_source] += 1
        if debug_prompts and debug_prompt_printed < debug_prompt_limit:
            preview = (prompt or "")[:debug_prompt_chars].replace("\n", " ")
            print(
                f"[Adapter-BAR][prompt-debug] src={prompt_source} game={_row_get(row, 'game', 'game_name', default='')} "
                f"exp={_row_get(row, 'experiment', 'experiment_name', default='')} task={_row_get(row, 'task_id', 'game_id', 'instance_id', default='')} "
                f"len={len(prompt or '')} preview={preview}",
                flush=True,
            )
            debug_prompt_printed += 1

        pred = predict_adapter(bundle, prompt)
        predicted_adapter = str(pred["predicted_adapter"])
        predicted_adapter_id = int(pred["predicted_adapter_id"])
        router_confidence = float(pred["router_confidence"])
        if verbose_steps:
            print(
                f"[Adapter-BAR] route instance {example_count + 1}/{total_rows}: "
                f"game={_row_get(row, 'game', 'game_name', default='')} "
                f"exp={_row_get(row, 'experiment', 'experiment_name', default='')} "
                f"task={_row_get(row, 'task_id', 'game_id', 'instance_id', default='')} "
                f"adapter={predicted_adapter} conf={router_confidence:.4f}",
                flush=True,
            )
        confidence_vals.append(router_confidence)
        usage[predicted_adapter] += 1
        usage_by_game[str(game)][predicted_adapter] += 1

        gold_label = None
        if game is not None:
            if router_type == "cluster":
                game_to_cluster = dict(cfg.get("game_to_cluster") or {})
                gold_label = game_to_cluster.get(game)
            else:
                gold_label = game
        if gold_label is not None:
            with_gold += 1
            game_gold[str(game)][1] += 1
            if str(gold_label) == predicted_adapter:
                correct += 1
                game_gold[str(game)][0] += 1

        cache_hit, load_secs, current_cache = cache.touch(predicted_adapter)
        adapter_path = adapter_paths.get(predicted_adapter)
        if not adapter_path:
            raise ValueError(f"No adapter path configured for predicted adapter '{predicted_adapter}'")

        game_name = str(_row_get(row, "game", "game_name", default="") or "")
        task_id = _row_get(row, "task_id", "game_id", "instance_id", default=None)
        experiment = _row_get(row, "experiment", "experiment_name", default=None)
        exp_key = str(experiment) if experiment is not None else ""
        generated_text = None
        score_value = None
        if task_id is not None and game_name:
            if inference_mode == "legacy":
                expert_model_name = f"{model_spec.model_name}-adapterbar-{predicted_adapter}"
                expert_entry = model_spec.to_dict()
                expert_entry["model_name"] = expert_model_name
                mc = dict(expert_entry.get("model_config") or {})
                mc["peft_model"] = adapter_path
                expert_entry["model_config"] = mc
                run_spec = ModelSpec.from_dict(expert_entry)
                restore_registry = _with_temp_model_registry_entry(expert_entry)
                try:
                    generate_with_selected_expert(
                        suite_results_dir=suite_results_dir,
                        game_name=game_name,
                        task_id=int(task_id),
                        run_spec=run_spec,
                        gen_args=gen_args,
                        split="validation",
                        regime="unknown",
                    )
                finally:
                    if restore_registry is not None:
                        restore_registry()
            else:
                grouped_tasks[(game_name, predicted_adapter)][exp_key].add(int(task_id))

        log_row = {
            "instance_id": int(example_count),
            "example_id": row.get("example_id") or f"{game_name}:{task_id}",
            "game": game,
            "routing_mode": routing_mode,
            "gold_label": gold_label,
            "predicted_adapter": predicted_adapter,
            "predicted_adapter_id": predicted_adapter_id,
            "router_probs": pred["router_probs"],
            "router_confidence": router_confidence,
            "adapter_path": adapter_path,
            "adapter_cache_hit": cache_hit,
            "current_adapter_cache": current_cache,
            "adapter_load_time_seconds": load_secs,
            "prompt_preview": (prompt or "")[:240],
            "generated_text": generated_text,
            "score": score_value,
            "meta": row.get("meta"),
            "experiment": experiment,
            "task_id": task_id,
        }
        append_router_log_row(log_path, log_row)
        example_count += 1
        if example_count % 200 == 0 or example_count == total_rows:
            print(
                f"[Adapter-BAR] routed {example_count}/{total_rows} rows "
                f"(groups_so_far={len(grouped_tasks)})",
                flush=True,
            )
        if score_value is not None:
            score_by_adapter[predicted_adapter].append(float(score_value))
            score_by_game[str(game)].append(float(score_value))
    if missing_prompt_count > 0:
        print(
            f"[Adapter-BAR] warning: {missing_prompt_count}/{total_rows} rows had empty prompt fields; "
            "used context fallback for routing text.",
            flush=True,
        )
    print(f"[Adapter-BAR] prompt sources: {dict(prompt_source_counts)}", flush=True)

    if inference_mode == "grouped":
        total_groups = len(grouped_tasks)
        total_tasks = sum(len(ids) for by_exp in grouped_tasks.values() for ids in by_exp.values())
        print(f"[Adapter-BAR] grouped inference enabled: groups={total_groups}, tasks={total_tasks}, suite={suite}", flush=True)
        if verbose_steps:
            print("[Adapter-BAR] grouped inference detail follows (loss-router style)", flush=True)
        for group_idx, (game_name, predicted_adapter) in enumerate(sorted(grouped_tasks.keys()), start=1):
            adapter_path = adapter_paths.get(predicted_adapter)
            if not adapter_path:
                raise ValueError(f"No adapter path configured for predicted adapter '{predicted_adapter}'")

            expert_model_name = f"{model_spec.model_name}-adapterbar-{predicted_adapter}"
            expert_entry = model_spec.to_dict()
            expert_entry["model_name"] = expert_model_name
            mc = dict(expert_entry.get("model_config") or {})
            mc["peft_model"] = adapter_path
            expert_entry["model_config"] = mc
            run_spec = ModelSpec.from_dict(expert_entry)
            restore_registry = _with_temp_model_registry_entry(expert_entry)

            by_experiment = grouped_tasks[(game_name, predicted_adapter)]
            by_experiment_norm = {_norm_key(exp): ids for exp, ids in by_experiment.items() if _norm_key(exp)}
            selected_task_ids = {int(tid) for ids in by_experiment.values() for tid in ids}
            group_task_count = sum(len(ids) for ids in by_experiment.values())
            print(
                f"[Adapter-BAR] running group {group_idx}/{total_groups} "
                f"game={game_name} adapter={predicted_adapter} tasks={group_task_count}"
            , flush=True)
            if verbose_steps:
                print(
                    f"[Adapter-BAR] grouped batch {group_idx}/{total_groups} "
                    f"adapter={predicted_adapter} game={game_name} tasks={group_task_count}",
                    flush=True,
                )

            try:
                def sub_selector_group(
                    *args,
                    _g=str(game_name),
                    _by=by_experiment,
                    _by_norm=by_experiment_norm,
                    _selected_task_ids=selected_task_ids,
                ):
                    # row-filter style: selector(row) -> bool
                    if len(args) == 1:
                        row = args[0]
                        game, experiment, task_id = _selector_row_triplet(row)
                        if task_id is None:
                            return False
                        g = str(game) if game is not None else _g
                        if g != _g:
                            return False
                        tid = int(task_id)
                        e = str(experiment) if experiment is not None else None
                        if e is not None and e in _by and tid in _by[e]:
                            return True
                        e_norm = _norm_key(e) if e is not None else ""
                        if e_norm and e_norm in _by_norm and tid in _by_norm[e_norm]:
                            return True
                        # Fallback only when experiment is missing/empty in row callbacks.
                        if e is None:
                            return tid in _selected_task_ids
                        return False

                    # legacy style: selector(game_name, experiment_name) -> list[int]
                    if len(args) == 2:
                        game_name_arg, experiment_name_arg = args
                        g = str(game_name_arg) if game_name_arg is not None else None
                        if g != _g:
                            return []
                        if isinstance(experiment_name_arg, Mapping):
                            exp_raw = (
                                experiment_name_arg.get("name")
                                or experiment_name_arg.get("experiment")
                                or experiment_name_arg.get("experiment_name")
                            )
                        else:
                            exp_raw = experiment_name_arg
                        exp = str(exp_raw) if exp_raw is not None else ""
                        if exp in _by:
                            return sorted(_by[exp])
                        exp_norm = _norm_key(exp)
                        if exp_norm in _by_norm:
                            return sorted(_by_norm[exp_norm])
                        return []

                    return False

                with _eval_context_env(game=game_name, split="validation", regime="unknown"):
                    t0_run = time.time()
                    print(
                        f"[Adapter-BAR] _clem_run start group {group_idx}/{total_groups} "
                        f"game={game_name} adapter={predicted_adapter}",
                        flush=True,
                    )
                    _clem_run(game_name, [run_spec], gen_args, suite_results_dir, selector_fn=sub_selector_group)
                    print(
                        f"[Adapter-BAR] _clem_run done group {group_idx}/{total_groups} "
                        f"game={game_name} adapter={predicted_adapter} elapsed={time.time() - t0_run:.1f}s",
                        flush=True,
                    )
            finally:
                if restore_registry is not None:
                    restore_registry()

    clem.score(game_selector, str(suite_results_dir))
    clem.transcripts(game_selector, str(suite_results_dir))
    try:
        df = clem.clemeval.perform_evaluation(str(suite_results_dir), return_dataframe=True)
        clem_score = _extract_clemscore(df, model_spec.model_name)
    except KeyError as e:
        print(
            f"[playpen eval] warning: clemeval missing aggregate columns for {model_spec.model_name} "
            f"(suite={suite}, routing_mode={routing_mode}); returning clemscore=0.0. details={e}",
            flush=True,
        )
        clem_score = 0.0

    total = max(1, sum(usage.values()))
    summary = {
        "total_instances": int(sum(usage.values())),
        "routing_mode": routing_mode,
        "inference_mode": inference_mode,
        "adapter_usage_counts": dict(usage),
        "adapter_usage_percentages": {k: float(v) / total for k, v in usage.items()},
        "routing_accuracy_if_gold_available": (float(correct) / with_gold) if with_gold > 0 else None,
        "per_game_adapter_usage": {g: dict(c) for g, c in usage_by_game.items()},
        "per_game_routing_accuracy": {g: (v[0] / max(1, v[1])) for g, v in game_gold.items()},
        "average_router_confidence": (sum(confidence_vals) / len(confidence_vals)) if confidence_vals else 0.0,
        "downstream_score_by_predicted_adapter": {
            k: (sum(v) / len(v) if v else None) for k, v in score_by_adapter.items()
        },
        "downstream_score_by_game": {k: (sum(v) / len(v) if v else None) for k, v in score_by_game.items()},
        "suite_score": clem_score,
    }
    (results_dir / "adapter_bar_routing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return clem_score


def evaluate_adapter_bar(
    suite: str,
    routing_mode: str,
    model_spec: ModelSpec,
    gen_args: Dict,
    results_dir: Path,
    game_selector,
    skip_gameplay: bool,
    adapter_bar_router_path: str,
    adapter_bar_config: str,
    max_instances: Optional[int] = None,
):
    overall_results_file = results_dir / f"{model_spec.model_name}.val.json"
    if suite in ["all", "clem"]:
        dataset_name = None if skip_gameplay else "instances"
        _game_selector = GameSpec.from_dict({"benchmark": ["3.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        clem_score = evaluate_suite_adapter_bar(
            "clem",
            routing_mode,
            model_spec,
            gen_args,
            results_dir,
            _game_selector,
            dataset_name,
            adapter_bar_router_path,
            adapter_bar_config,
            max_instances=max_instances,
        )
        store_eval_score(overall_results_file, "clemscore", clem_score)
    if suite in ["all", "static"]:
        dataset_name = None if skip_gameplay else "instances-static"
        _game_selector = GameSpec.from_dict({"benchmark": ["static_1.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        stat_score = evaluate_suite_adapter_bar(
            "static",
            routing_mode,
            model_spec,
            gen_args,
            results_dir,
            _game_selector,
            dataset_name,
            adapter_bar_router_path,
            adapter_bar_config,
            max_instances=max_instances,
        )
        store_eval_score(overall_results_file, "statscore", stat_score)


def cli(args: argparse.Namespace):
    _install_model_registry_runtime_path_patch()
    _install_hf_local_moe_patch()
    _install_player_call_timeout_patch()
    _install_instance_timeout_patch()
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
            if args.routing_mode in {"adapter_bar_sequence_cluster", "adapter_bar_sequence_game"}:
                if not args.adapter_bar_router_path:
                    raise ValueError("--adapter_bar_router_path is required for Adapter-BAR routing modes.")
                model_spec = ModelSpec.from_string(args.model)
                evaluate_adapter_bar(
                    args.suite,
                    args.routing_mode,
                    model_spec,
                    gen_args,
                    args.results_dir,
                    args.game,
                    args.skip_gameplay,
                    args.adapter_bar_router_path,
                    args.adapter_bar_config,
                    max_instances=args.max_instances,
                )
            elif args.moe is not None:
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
                                  " Default: {\"benchmark\": [\"3.0\"]} (clem suite)"
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
    eval_parser.add_argument(
        "--routing_mode",
        type=str,
        default="default",
        choices=["default", "adapter_bar_sequence_cluster", "adapter_bar_sequence_game"],
        help="(Optional) sequence-level Adapter-BAR routing mode.",
    )
    eval_parser.add_argument(
        "--adapter_bar_router_path",
        type=str,
        default=None,
        help="(Optional) Path to sequence-level Adapter-BAR router checkpoint (router.pt).",
    )
    eval_parser.add_argument(
        "--adapter_bar_config",
        type=str,
        default="configs/adapter_bar_sequence.yaml",
        help="(Optional) Path to sequence-level Adapter-BAR config.",
    )
    eval_parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="(Optional) limit number of instances for smoke tests.",
    )
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
