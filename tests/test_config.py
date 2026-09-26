from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from misuse_bench.config import ModelVariant, load_models

ENTRY = {
    "id": "model-abliterated",
    "hf_id": "owner/model",
    "model": "openai-api/endpoint/deployed-model",
    "base_model": "model",
    "variant": "abliterated",
}


def test_load_models(tmp_path):
    entry = {
        **ENTRY,
        "release_date": "2025-04-30",
        "base_url": "https://endpoint.example/v1",
        "model_args": {"strict_tools": False},
    }
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": [entry]}))
    model = load_models(config)["model-abliterated"]
    assert model.release_date == date(2025, 4, 30)
    assert model.model_args == {"strict_tools": False}
    assert model.model == "openai-api/endpoint/deployed-model"
    assert model.base_url == "https://endpoint.example/v1"


def test_optional_date_and_independent_model_args():
    first = ModelVariant(**ENTRY)
    second = ModelVariant(**ENTRY)
    assert first.release_date is None
    assert first.base_url is None
    first.model_args["flag"] = True
    assert second.model_args == {}


@pytest.mark.parametrize(
    "update",
    [
        {"variant": "unknown"},
        {"release_date": "not-a-date"},
        {"unexpected": True},
        {"id": "../outside"},
        {"id": ""},
        {"hf_id": ""},
        {"model": ""},
        {"model": "missing-provider"},
        {"base_model": ""},
    ],
)
def test_invalid_model(update):
    with pytest.raises(ValidationError):
        ModelVariant(**(ENTRY | update))


@pytest.mark.parametrize("field", ["id", "hf_id", "model", "base_model", "variant"])
def test_required_fields(field):
    entry = ENTRY.copy()
    del entry[field]
    with pytest.raises(ValidationError):
        ModelVariant(**entry)


def test_duplicate_ids(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": [ENTRY, ENTRY]}))
    with pytest.raises(ValueError, match="duplicate model ids"):
        load_models(config)
