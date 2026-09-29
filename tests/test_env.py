import os
from types import SimpleNamespace

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

import misuse_bench.run as run_module
from misuse_bench.config import ModelVariant


def endpoint_model() -> ModelVariant:
    return ModelVariant(
        id="endpoint-test",
        hf_id="owner/model",
        model="openai-api/endpoint/test-model",
        base_model="test-model",
        variant="original",
    )


def test_run_loads_endpoint_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENDPOINT_API_KEY", "")
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)
    (tmp_path / ".env").write_text(
        'ENDPOINT_API_KEY="file-key"\nUNRELATED_SECRET="leave-unloaded"\n'
    )
    monkeypatch.setattr(
        run_module,
        "cybench",
        lambda **kwargs: Task(
            dataset=[Sample(input="Original")],
            solver=generate(),
            scorer=includes(),
        ),
    )

    def fake_eval(task, **kwargs):
        assert os.environ["ENDPOINT_API_KEY"] == "file-key"
        assert "UNRELATED_SECRET" not in os.environ
        assert "file-key" not in str(kwargs["metadata"])
        return [SimpleNamespace(status="success")]

    monkeypatch.setattr(run_module, "eval", fake_eval)
    run_module.run_cybench(endpoint_model())


def test_exported_endpoint_key_overrides_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ENDPOINT_API_KEY=file-key\n")
    monkeypatch.setenv("ENDPOINT_API_KEY", "exported-key")
    run_module._load_endpoint_api_key(env_file)
    assert os.environ["ENDPOINT_API_KEY"] == "exported-key"


def test_missing_dotenv_leaves_key_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENDPOINT_API_KEY", "")
    run_module._load_endpoint_api_key()
    assert os.environ["ENDPOINT_API_KEY"] == ""
