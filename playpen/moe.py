import ast
import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MoeRoute:
    model: str
    game: str = "*"
    experiment: Optional[str] = None
    keywords: Tuple[str, ...] = ()
    min_keyword_hits: int = 1

    def matches(self, game: str, experiment: Optional[str], context_text: str) -> bool:
        if not fnmatchcase(game, self.game):
            return False
        if self.experiment is None:
            experiment_ok = True
        else:
            if experiment is None:
                return False
            experiment_ok = fnmatchcase(experiment, self.experiment)

        if not experiment_ok:
            return False

        if not self.keywords:
            return True
        if not context_text:
            return False

        text = context_text.lower()
        hits = 0
        for kw in self.keywords:
            if not kw:
                continue
            if kw.lower() in text:
                hits += 1
                if hits >= max(1, int(self.min_keyword_hits)):
                    return True
        return False


@dataclass(frozen=True)
class MoeConfig:
    name: str
    default_model: str
    routes: Tuple[MoeRoute, ...]
    route_by_experiment: bool = False
    router: Optional["TextRouterConfig"] = None

    def select_model(self, game: str, experiment: Optional[str] = None, *, context_text: str = "") -> str:
        for route in self.routes:
            if route.matches(game, experiment, context_text):
                return route.model

        if self.router is not None and context_text:
            router = load_text_router(self.router.path)
            predicted, confidence = router.predict(context_text)
            if predicted is not None:
                if confidence is None or confidence >= self.router.min_confidence:
                    return predicted

        return self.default_model


@dataclass(frozen=True)
class TextRouterConfig:
    path: str
    min_confidence: float = 0.0


class NaiveBayesTextRouter:
    """
    Simple multinomial Naive Bayes bag-of-words router.
    """

    def __init__(
        self,
        labels: Sequence[str],
        log_priors: Sequence[float],
        vocab: Sequence[str],
        log_likelihoods: Sequence[Sequence[float]],
        unk_log_likelihoods: Sequence[float],
    ):
        self.labels = list(labels)
        self._log_priors = list(log_priors)
        self._vocab = list(vocab)
        self._token_to_idx = {t: i for i, t in enumerate(self._vocab)}
        self._log_likelihoods = [list(row) for row in log_likelihoods]
        self._unk_log_likelihoods = list(unk_log_likelihoods)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Very simple tokenizer; good enough for a routing signal.
        out: List[str] = []
        buf: List[str] = []
        for ch in (text or "").lower():
            if ch.isalnum() or ch in ("_", "-"):
                buf.append(ch)
            else:
                if buf:
                    out.append("".join(buf))
                    buf = []
        if buf:
            out.append("".join(buf))
        return out

    def _log_posteriors(self, text: str) -> List[float]:
        tokens = self.tokenize(text)
        scores = list(self._log_priors)
        for token in tokens:
            idx = self._token_to_idx.get(token)
            if idx is None:
                for c in range(len(scores)):
                    scores[c] += self._unk_log_likelihoods[c]
            else:
                for c in range(len(scores)):
                    scores[c] += self._log_likelihoods[c][idx]
        return scores

    @staticmethod
    def _softmax(logits: Sequence[float]) -> List[float]:
        if not logits:
            return []
        import math
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps) or 1.0
        return [e / s for e in exps]

    def predict(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        if not text:
            return None, None
        scores = self._log_posteriors(text)
        probs = self._softmax(scores)
        if not probs:
            return None, None
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return self.labels[best_idx], float(probs[best_idx])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "nb_bow_v1",
            "labels": self.labels,
            "log_priors": self._log_priors,
            "vocab": self._vocab,
            "log_likelihoods": self._log_likelihoods,
            "unk_log_likelihoods": self._unk_log_likelihoods,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "NaiveBayesTextRouter":
        if (d.get("type") or "") != "nb_bow_v1":
            raise ValueError("Unsupported router type (expected 'nb_bow_v1').")
        return NaiveBayesTextRouter(
            labels=d["labels"],
            log_priors=d["log_priors"],
            vocab=d["vocab"],
            log_likelihoods=d["log_likelihoods"],
            unk_log_likelihoods=d["unk_log_likelihoods"],
        )


_TEXT_ROUTER_CACHE: Dict[str, NaiveBayesTextRouter] = {}


def load_text_router(path: str) -> NaiveBayesTextRouter:
    path = str(Path(path))
    cached = _TEXT_ROUTER_CACHE.get(path)
    if cached is not None:
        return cached
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    router = NaiveBayesTextRouter.from_dict(obj)
    _TEXT_ROUTER_CACHE[path] = router
    return router


def train_nb_text_router(examples: Sequence[Tuple[str, str]]) -> NaiveBayesTextRouter:
    """
    Train multinomial NB with add-one smoothing.
    `examples`: list of (text, label).
    """
    if not examples:
        raise ValueError("No training examples provided.")

    # Build label set
    labels = sorted({lbl for _, lbl in examples})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    # Count tokens per class
    token_counts: List[Dict[str, int]] = [dict() for _ in labels]
    total_tokens: List[int] = [0 for _ in labels]
    doc_counts: List[int] = [0 for _ in labels]
    vocab_set = set()

    for text, label in examples:
        c = label_to_idx[label]
        doc_counts[c] += 1
        for tok in NaiveBayesTextRouter.tokenize(text):
            vocab_set.add(tok)
            token_counts[c][tok] = token_counts[c].get(tok, 0) + 1
            total_tokens[c] += 1

    vocab = sorted(vocab_set)
    vocab_idx = {t: i for i, t in enumerate(vocab)}

    # Priors
    n_docs = sum(doc_counts) or 1
    import math
    log_priors = []
    for c in range(len(labels)):
        prior = doc_counts[c] / n_docs
        # avoid log(0)
        log_priors.append(float("-inf") if prior <= 0 else float(math.log(prior)))

    # Likelihoods with add-one smoothing
    log_likelihoods: List[List[float]] = [[0.0 for _ in vocab] for _ in labels]
    unk_log_likelihoods: List[float] = [0.0 for _ in labels]
    vsize = len(vocab)
    for c in range(len(labels)):
        denom = total_tokens[c] + vsize  # +1 per vocab token
        unk_log_likelihoods[c] = math.log(1.0 / denom)
        counts = token_counts[c]
        for tok, cnt in counts.items():
            i = vocab_idx.get(tok)
            if i is None:
                continue
            log_likelihoods[c][i] = math.log((cnt + 1.0) / denom)
        # tokens not seen in class -> log(1/denom), already 0.0; fill them
        default = math.log(1.0 / denom)
        for i in range(len(vocab)):
            if log_likelihoods[c][i] == 0.0:
                log_likelihoods[c][i] = default

    return NaiveBayesTextRouter(
        labels=labels,
        log_priors=log_priors,
        vocab=vocab,
        log_likelihoods=log_likelihoods,
        unk_log_likelihoods=unk_log_likelihoods,
    )


def _parse_config_payload(payload: str) -> Any:
    payload = payload.strip()
    if not payload:
        raise ValueError("Empty MoE config.")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            # Allow single quotes and python-style dicts: "{'benchmark': ['2.0']}"
            return ast.literal_eval(payload)
        except Exception as e:
            raise ValueError(
                "MoE config must be a JSON file path or a JSON/Python-literal dict."
            ) from e


def load_moe_config(moe: str, *, default_name: str, default_model: str) -> MoeConfig:
    """
    Load a MoE routing config from:
      - a JSON file path
      - an inline JSON / python-literal dict string

    Supported schema:
      {
        "name": "llama3-8b-moe",              # optional, defaults to default_name
        "default_model": "llama3-8b",         # optional, defaults to default_model
        "route_by_experiment": false,         # optional, defaults to false
        "game_map": {                         # optional, shorthand for exact game_name routing
          "adventuregame": "llama3-8b-exploration_navigation",
          "wordle": "llama3-8b-wordguessing"
        },
        "routes": [
          {"game": "wordle*", "model": "llama3-8b-wordguessing"},
          {"game": "adventuregame", "model": "llama3-8b-exploration_navigation"},
          {"game": "*", "model": "llama3-8b-cooperation"}
        ]
      }
    Patterns use shell-style globs (fnmatch).
    """
    moe = (moe or "").strip()
    if not moe:
        raise ValueError("Missing MoE config.")

    path = Path(moe)
    if path.exists() and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = _parse_config_payload(moe)

    if not isinstance(payload, Mapping):
        raise ValueError("MoE config must be an object/dict at the top level.")

    name = str(payload.get("name") or default_name)
    default = str(payload.get("default_model") or default_model)
    route_by_experiment = bool(payload.get("route_by_experiment", False))
    router_cfg = payload.get("router")
    router: Optional[TextRouterConfig] = None
    if router_cfg is not None:
        if not isinstance(router_cfg, Mapping):
            raise ValueError("MoE config field 'router' must be an object/dict.")
        router_path = router_cfg.get("path")
        if not router_path:
            raise ValueError("MoE router config requires field 'path'.")
        router = TextRouterConfig(
            path=str(router_path),
            min_confidence=float(router_cfg.get("min_confidence", 0.0) or 0.0),
        )

    game_map = payload.get("game_map") or payload.get("game_name_map") or payload.get("game_to_model")
    game_map_routes: List[Dict[str, str]] = []
    if game_map is not None:
        if not isinstance(game_map, Mapping):
            raise ValueError("MoE config field 'game_map' must be an object/dict.")
        for game_name, model_name in game_map.items():
            if not model_name:
                raise ValueError(f"MoE game_map entry for '{game_name}' is missing a model.")
            game_map_routes.append({"game": str(game_name), "model": str(model_name)})

    raw_routes = payload.get("routes", [])
    if raw_routes is None:
        raw_routes = []
    if not isinstance(raw_routes, Sequence):
        raise ValueError("MoE config field 'routes' must be a list.")
    if game_map_routes:
        raw_routes = game_map_routes + list(raw_routes)

    routes: List[MoeRoute] = []
    for idx, raw in enumerate(raw_routes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"MoE route #{idx} must be an object/dict.")
        model = raw.get("model")
        if not model:
            raise ValueError(f"MoE route #{idx} is missing required field 'model'.")
        game = raw.get("game")
        if game is None and "game_name" in raw:
            game = raw.get("game_name")
        game = str(game or "*")
        experiment = raw.get("experiment")
        if experiment is not None:
            experiment = str(experiment)
        keywords_raw = raw.get("keywords") or []
        if keywords_raw is None:
            keywords_raw = []
        if isinstance(keywords_raw, str):
            keywords = (keywords_raw,)
        else:
            if not isinstance(keywords_raw, Sequence):
                raise ValueError(f"MoE route #{idx} field 'keywords' must be a string or list of strings.")
            keywords = tuple(str(k) for k in keywords_raw if k is not None)
        min_keyword_hits = int(raw.get("min_keyword_hits", 1) or 1)
        routes.append(
            MoeRoute(
                model=str(model),
                game=game,
                experiment=experiment,
                keywords=keywords,
                min_keyword_hits=min_keyword_hits,
            )
        )

    # Auto-enable experiment routing if any route specifies 'experiment'
    if not route_by_experiment and any(r.experiment is not None for r in routes):
        route_by_experiment = True

    return MoeConfig(
        name=name,
        default_model=default,
        routes=tuple(routes),
        route_by_experiment=route_by_experiment,
        router=router,
    )


def tasks_by_game_experiment(dataset: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, str], List[int]]:
    tasks: Dict[Tuple[str, str], List[int]] = {}
    for row in dataset:
        game = row.get("game")
        experiment = row.get("experiment")
        task_id = row.get("task_id")
        if game is None or experiment is None or task_id is None:
            continue
        key = (str(game), str(experiment))
        tasks.setdefault(key, []).append(int(task_id))
    return tasks
