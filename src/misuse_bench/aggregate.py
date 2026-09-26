from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean

import pandas as pd
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

from misuse_bench.config import ModelVariant


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased probability of at least one success in k of n attempts."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("pass@k requires 0 <= c <= n and 1 <= k <= n")
    return 1 - comb(n - c, k) / comb(n, k)


def challenge_scores(log: EvalLog) -> dict[str | int, list[int]]:
    """Read Cybench's includes scorer, retaining individual epoch outcomes."""
    if log.status != "success" or not log.samples:
        raise ValueError("A successful log with sample scores is required")
    outcomes: dict[str | int, list[int]] = defaultdict(list)
    seen = set()
    for sample in log.samples:
        key = (sample.id, sample.epoch)
        if key in seen:
            raise ValueError(f"Duplicate challenge/epoch: {key}")
        seen.add(key)
        if sample.error or not sample.scores or "includes" not in sample.scores:
            raise ValueError(f"Missing valid score for challenge/epoch: {key}")
        value = sample.scores["includes"].value
        if value not in ("C", "I", 0, 1):
            raise ValueError(f"Non-binary score for challenge/epoch: {key}: {value}")
        outcomes[sample.id].append(int(value in ("C", 1)))
    expected_epochs = set(range(1, (log.eval.config.epochs or 1) + 1))
    for challenge_id in outcomes:
        epochs = {epoch for sample_id, epoch in seen if sample_id == challenge_id}
        if epochs != expected_epochs:
            raise ValueError(f"Incomplete epochs for challenge {challenge_id}")
    return dict(outcomes)


def variant_scores(outcomes: dict[str | int, list[int]], k: int) -> dict:
    """Give every challenge equal weight, regardless of its epoch count."""
    if not outcomes or any(not values for values in outcomes.values()):
        raise ValueError("At least one scored epoch per challenge is required")
    return {
        "challenges": len(outcomes),
        "attempts": sum(len(values) for values in outcomes.values()),
        "success_rate": mean(mean(values) for values in outcomes.values()),
        "k": k,
        "pass_at_k": mean(
            pass_at_k(len(values), sum(values), k) for values in outcomes.values()
        ),
    }


def ceilings(scores: pd.DataFrame) -> pd.DataFrame:
    """Select the best whole variant for each metric and base model."""
    rows = []
    for base_model, group in scores.groupby("base_model", sort=True):
        row = {"base_model": base_model, "variants": len(group), "k": group.k.iloc[0]}
        for metric in ("success_rate", "pass_at_k"):
            best = group.loc[group[metric].idxmax()]
            row[metric] = best[metric]
            row[f"{metric}_model_id"] = best.model_id
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(
    models: dict[str, ModelVariant],
    *,
    k: int = 1,
    log_dir: Path = Path("logs"),
    results_dir: Path = Path("results"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise each variant's latest successful run and write both CSVs."""
    if k < 1:
        raise ValueError("k must be positive")
    rows = []
    challenges_by_base = {}
    for model_id, model in sorted(models.items()):
        directory = log_dir / model_id
        if not directory.exists():
            continue
        completed = []
        for file in list_eval_logs(str(directory), recursive=False):
            header = read_eval_log(file, header_only=True)
            if header.status == "success":
                completed.append((header.eval.created, file.name))
        if not completed:
            continue
        _, filename = max(completed)
        log = read_eval_log(filename)
        if (log.eval.metadata or {}).get("model_variant") != model.model_dump(
            mode="json"
        ):
            raise ValueError(
                f"{filename}: model config differs from the logged variant"
            )
        outcomes = challenge_scores(log)
        challenge_ids = set(outcomes)
        previous = challenges_by_base.setdefault(model.base_model, challenge_ids)
        if previous != challenge_ids:
            raise ValueError(f"Challenge sets differ for base model {model.base_model}")
        rows.append(
            {
                "model_id": model_id,
                "hf_id": model.hf_id,
                "base_model": model.base_model,
                "variant": model.variant,
                **variant_scores(outcomes, k),
                "log": filename,
            }
        )
    if not rows:
        raise ValueError(f"No successful model logs found in {log_dir}")
    scores = pd.DataFrame(rows)
    ceiling = ceilings(scores)
    results_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(results_dir / "cybench_scores.csv", index=False)
    ceiling.to_csv(results_dir / "cybench_ceiling.csv", index=False)
    return scores, ceiling
