import hashlib
import os
from pathlib import Path

from dotenv import dotenv_values
from inspect_ai import eval
from inspect_ai.agent import AgentAttempts, react
from inspect_ai.log import EvalLog
from inspect_ai.tool import bash, python
from inspect_evals.cybench import cybench
from inspect_evals.cybench.cybench import (
    DEFAULT_CONTINUE_MESSAGE,
    DEFAULT_INCORRECT_MESSAGE,
    DEFAULT_SYSTEM_MESSAGE,
)

from misuse_bench.config import ModelVariant


def _load_endpoint_api_key(env_file: Path = Path(".env")) -> None:
    if os.environ.get("ENDPOINT_API_KEY"):
        return
    if env_file.is_file():
        key = dotenv_values(env_file).get("ENDPOINT_API_KEY")
        if key:
            os.environ["ENDPOINT_API_KEY"] = key


def _load_prompts(
    prompt_dir: Path,
) -> tuple[str | None, dict[str, str], str | None, str | None, str]:
    if not prompt_dir.is_dir():
        raise ValueError(f"Prompt directory does not exist: {prompt_dir}")

    paths = [
        prompt_dir / "system.txt",
        prompt_dir / "continue.txt",
        prompt_dir / "incorrect.txt",
        *sorted((prompt_dir / "challenges").glob("*.txt")),
    ]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise ValueError(f"No prompt files found in {prompt_dir}")

    contents = {
        path.relative_to(prompt_dir).as_posix(): path.read_text().strip()
        for path in paths
    }
    if any(not content for content in contents.values()):
        raise ValueError(f"Prompt files must not be empty: {prompt_dir}")

    digest = hashlib.sha256()
    for name, content in sorted(contents.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(content.encode())
        digest.update(b"\0")

    system = contents.pop("system.txt", None)
    continue_message = contents.pop("continue.txt", None)
    incorrect_message = contents.pop("incorrect.txt", None)
    challenges = {Path(name).stem: content for name, content in contents.items()}
    return system, challenges, continue_message, incorrect_message, digest.hexdigest()


def run_cybench(
    model: ModelVariant,
    *,
    epochs: int = 1,
    limit: int | None = None,
    message_limit: int | None = None,
    token_limit: int | None = None,
    max_connections: int | None = None,
    log_dir: Path = Path("logs"),
    prompt_dir: Path | None = None,
) -> list[EvalLog]:
    """Evaluate one variant on Cybench's hard challenges."""
    for name, value in {
        "epochs": epochs,
        "limit": limit,
        "message_limit": message_limit,
        "token_limit": token_limit,
        "max_connections": max_connections,
    }.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")

    if model.model.startswith("openai-api/endpoint/"):
        _load_endpoint_api_key()

    prompt_config = _load_prompts(prompt_dir) if prompt_dir is not None else None
    task = cybench(sandbox_type="docker")
    metadata = {"model_variant": model.model_dump(mode="json")}
    if prompt_config is not None:
        system, challenges, continue_message, incorrect_message, prompt_hash = (
            prompt_config
        )
        if challenges:
            available = {sample.metadata["eval_name"] for sample in task.dataset}
            unknown = sorted(challenges.keys() - available)
            if unknown:
                raise ValueError(
                    f"Unknown Cybench challenges in {prompt_dir}: {', '.join(unknown)}"
                )
            task.dataset = task.dataset.flat_map(
                lambda sample: [
                    sample.model_copy(
                        deep=True,
                        update={"input": challenges[sample.metadata["eval_name"]]},
                    )
                    if sample.metadata["eval_name"] in challenges
                    else sample
                ]
            )
        if any(
            message is not None
            for message in (system, continue_message, incorrect_message)
        ):
            # Keep Cybench's tools and retry count while replacing supplied text.
            task.solver = react(
                prompt=system or DEFAULT_SYSTEM_MESSAGE,
                tools=[bash(timeout=180), python(timeout=180)],
                attempts=AgentAttempts(
                    attempts=3,
                    incorrect_message=incorrect_message or DEFAULT_INCORRECT_MESSAGE,
                ),
                on_continue=continue_message or DEFAULT_CONTINUE_MESSAGE,
            )
        metadata.update({"prompt_set": prompt_dir.name, "prompt_sha256": prompt_hash})

    logs = eval(
        task,
        model=model.model,
        model_base_url=model.base_url,
        model_args=model.model_args,
        epochs=epochs,
        limit=limit,
        message_limit=message_limit,
        token_limit=token_limit,
        max_connections=max_connections,
        log_dir=str(log_dir / model.id),
        log_samples=True,
        metadata=metadata,
    )
    if not logs or any(log.status != "success" for log in logs):
        raise RuntimeError(
            f"Evaluation failed for {model.id}; see {log_dir / model.id}"
        )
    return logs
