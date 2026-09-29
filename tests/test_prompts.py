from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate
from inspect_cyber.dataset import AgenticEvalDataset
from typer.testing import CliRunner

import misuse_bench.cli as cli_module
import misuse_bench.run as run_module
from misuse_bench.config import ModelVariant


def model() -> ModelVariant:
    return ModelVariant(
        id="test-model",
        hf_id="owner/model",
        model="mockllm/model",
        base_model="model",
        variant="original",
    )


def test_prompt_dir_replaces_only_named_challenge_and_system(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "field-assessment-v1"
    (prompt_dir / "challenges").mkdir(parents=True)
    (prompt_dir / "system.txt").write_text("  Custom system instruction.  \n")
    (prompt_dir / "continue.txt").write_text("  Keep going.  \n")
    (prompt_dir / "incorrect.txt").write_text("  Try again.  \n")
    (prompt_dir / "challenges" / "robust_cbc.txt").write_text(
        "Connect to victim:1337.\n"
    )
    samples = [
        Sample(
            id="robust_cbc (hard)",
            input="Connect to victim:9999.",
            target="secret-a",
            metadata={"eval_name": "robust_cbc"},
        ),
        Sample(
            id="rpgo (hard)",
            input="Original instruction.",
            target="secret-b",
            metadata={"eval_name": "rpgo"},
        ),
    ]
    calls = {}

    def fake_cybench(*, sandbox_type):
        assert sandbox_type == "docker"
        return Task(
            dataset=AgenticEvalDataset(tmp_path, samples),
            solver=generate(),
            scorer=includes(),
        )

    def fake_react(**kwargs):
        calls["react"] = kwargs
        return generate()

    def fake_eval(task, **kwargs):
        calls["task"] = task
        calls["eval"] = kwargs
        return [SimpleNamespace(status="success")]

    monkeypatch.setattr(run_module, "cybench", fake_cybench)
    monkeypatch.setattr(run_module, "react", fake_react)
    monkeypatch.setattr(run_module, "eval", fake_eval)

    run_module.run_cybench(model(), prompt_dir=prompt_dir)

    changed, unchanged = calls["task"].dataset
    assert changed.input == "Connect to victim:1337."
    assert changed.target == "secret-a"
    assert changed.metadata == {"eval_name": "robust_cbc"}
    assert unchanged.input == "Original instruction."
    assert unchanged.target == "secret-b"
    assert calls["react"]["prompt"] == "Custom system instruction."
    assert calls["react"]["on_continue"] == "Keep going."
    assert calls["react"]["attempts"].incorrect_message == "Try again."
    assert len(calls["react"]["tools"]) == 2
    assert calls["react"]["attempts"].attempts == 3
    assert calls["eval"]["metadata"]["prompt_set"] == "field-assessment-v1"
    original_hash = calls["eval"]["metadata"]["prompt_sha256"]
    assert len(original_hash) == 64

    (prompt_dir / "continue.txt").write_text("Different follow-up.\n")
    run_module.run_cybench(model(), prompt_dir=prompt_dir)
    assert calls["eval"]["metadata"]["prompt_sha256"] != original_hash


def test_unknown_challenge_prompt_is_rejected(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    (prompt_dir / "challenges").mkdir(parents=True)
    (prompt_dir / "challenges" / "typo.txt").write_text("New instruction.\n")
    monkeypatch.setattr(
        run_module,
        "cybench",
        lambda **kwargs: Task(
            dataset=AgenticEvalDataset(
                tmp_path,
                [Sample(input="Original", metadata={"eval_name": "robust_cbc"})],
            ),
            solver=generate(),
            scorer=includes(),
        ),
    )
    with pytest.raises(ValueError, match="Unknown Cybench challenges.*typo"):
        run_module.run_cybench(model(), prompt_dir=prompt_dir)


def test_missing_prompt_dir_fails_before_building_task(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_module, "cybench", lambda **kwargs: pytest.fail("task constructed")
    )
    with pytest.raises(ValueError, match="Prompt directory does not exist"):
        run_module.run_cybench(model(), prompt_dir=tmp_path / "missing")


def test_cli_accepts_prompt_dir(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": [model().model_dump(mode="json")]}))
    calls = {}

    def fake_run(model_variant, **kwargs):
        calls["model"] = model_variant
        calls.update(kwargs)

    monkeypatch.setattr(cli_module, "run_cybench", fake_run)
    prompt_dir = tmp_path / "prompts"
    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            "--model",
            "test-model",
            "--config",
            str(config),
            "--prompt-dir",
            str(prompt_dir),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls["model"].id == "test-model"
    assert calls["prompt_dir"] == Path(prompt_dir)
