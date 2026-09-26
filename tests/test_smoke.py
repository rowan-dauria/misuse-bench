import os
import traceback

import pandas as pd
import pytest
import yaml
from inspect_ai import Task, eval
from inspect_ai._util import appdirs
from inspect_ai.dataset import Sample
from inspect_ai.model._providers.mockllm import MockLLM
from inspect_ai.scorer import includes
from inspect_ai.solver import generate
from typer.testing import CliRunner

import misuse_bench.run as run_module
from misuse_bench.cli import app


def test_cli_smoke_without_gpu_docker_or_downloads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CYBENCH_ACKNOWLEDGE_RISKS", raising=False)
    # Keep Inspect state local and avoid tiktoken's first-use vocabulary download.
    monkeypatch.setattr(appdirs, "user_data_path", lambda _: tmp_path / "data")
    monkeypatch.setattr(appdirs, "user_cache_path", lambda _: tmp_path / "cache")

    async def count_text_tokens(self, text):
        return len(text.split())

    monkeypatch.setattr(MockLLM, "count_text_tokens", count_text_tokens)
    models = [
        {
            "id": "original",
            "hf_id": "owner/original",
            "model": "openai-api/endpoint/deployed-original",
            "base_url": "https://endpoint.example/v1",
            "base_model": "base",
            "variant": "original",
            "model_args": {"strict_tools": False},
        },
        {
            "id": "abliterated",
            "hf_id": "owner/abliterated",
            "model": "mockllm/model",
            "base_model": "base",
            "variant": "abliterated",
            "model_args": {"strict_tools": False},
        },
    ]
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": models}))
    calls = []

    def tiny_cybench(*, sandbox_type):
        assert sandbox_type == "docker"
        return Task(
            dataset=[
                Sample(id="success", input="Say hello", target="Default output"),
                Sample(id="failure", input="Say hello", target="absent flag"),
            ],
            solver=generate(),
            scorer=includes(),
        )

    def mock_eval(task, **kwargs):
        calls.append(kwargs.copy())
        kwargs["model"] = "mockllm/model"
        return eval(task, display="none", **kwargs)

    monkeypatch.setattr(run_module, "cybench", tiny_cybench)
    monkeypatch.setattr(run_module, "eval", mock_eval)
    runner = CliRunner()
    for model in models:
        result = runner.invoke(
            app,
            [
                "run",
                "--model",
                model["id"],
                "--config",
                str(config),
                "--epochs",
                "3",
                "--limit",
                "2",
                "--message-limit",
                "10",
                "--token-limit",
                "1000",
                "--max-connections",
                "1",
            ],
        )
        assert result.exit_code == 0, (
            result.output,
            "".join(traceback.format_exception(*result.exc_info))
            if result.exc_info
            else "",
        )
    assert calls[0]["model"] == "openai-api/endpoint/deployed-original"
    assert calls[1]["model"] == "mockllm/model"
    assert calls[0]["model_base_url"] == "https://endpoint.example/v1"
    assert calls[1]["model_base_url"] is None
    assert calls[0]["model_args"] == {"strict_tools": False}
    assert calls[0]["message_limit"] == 10
    assert calls[0]["token_limit"] == 1000
    assert calls[0]["max_connections"] == 1
    assert calls[0]["log_dir"] == "logs/original"
    result = runner.invoke(app, ["aggregate", "--config", str(config), "--k", "2"])
    assert result.exit_code == 0, (result.output, result.exception)
    scores = pd.read_csv("results/cybench_scores.csv")
    ceiling = pd.read_csv("results/cybench_ceiling.csv")
    assert len(scores) == 2
    assert scores.challenges.tolist() == [2, 2]
    assert scores.attempts.tolist() == [6, 6]
    assert scores.success_rate.tolist() == [0.5, 0.5]
    assert scores.pass_at_k.tolist() == [0.5, 0.5]
    assert ceiling.base_model.tolist() == ["base"]
    assert ceiling.success_rate.tolist() == [0.5]
    assert ceiling.pass_at_k_model_id.tolist() == ["abliterated"]
    assert len(list((tmp_path / "logs").rglob("*.eval"))) == 2
    assert "CYBENCH_ACKNOWLEDGE_RISKS" not in os.environ


def test_unknown_model_and_invalid_options(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text("models: []\n")
    monkeypatch.setattr(
        run_module, "cybench", lambda **kwargs: pytest.fail("task constructed")
    )
    runner = CliRunner()
    for options in (["--model", "missing"], ["--model", "missing", "--epochs", "0"]):
        result = runner.invoke(app, ["run", "--config", str(config), *options])
        assert result.exit_code == 2
