from pathlib import Path

from inspect_ai import eval
from inspect_ai.log import EvalLog
from inspect_evals.cybench import cybench

from misuse_bench.config import ModelVariant


def run_cybench(
    model: ModelVariant,
    *,
    epochs: int = 1,
    limit: int | None = None,
    message_limit: int | None = None,
    token_limit: int | None = None,
    max_connections: int | None = None,
    log_dir: Path = Path("logs"),
) -> list[EvalLog]:
    """Evaluate one variant on Cybench's default hard challenges."""
    for name, value in {
        "epochs": epochs,
        "limit": limit,
        "message_limit": message_limit,
        "token_limit": token_limit,
        "max_connections": max_connections,
    }.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")

    logs = eval(
        cybench(sandbox_type="docker"),
        model=f"vllm/{model.hf_id}",
        model_args=model.model_args,
        epochs=epochs,
        limit=limit,
        message_limit=message_limit,
        token_limit=token_limit,
        max_connections=max_connections,
        log_dir=str(log_dir / model.id),
        log_samples=True,
        metadata={"model_variant": model.model_dump(mode="json")},
    )
    if not logs or any(log.status != "success" for log in logs):
        raise RuntimeError(
            f"Evaluation failed for {model.id}; see {log_dir / model.id}"
        )
    return logs
