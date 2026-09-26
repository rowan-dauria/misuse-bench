from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIG = Path("configs/models.yaml")


class ModelVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    hf_id: str
    base_model: str
    variant: Literal["original", "abliterated", "fine-tuned"]
    release_date: date | None
    model_args: dict[str, Any] = {}


def load_models(path: Path = DEFAULT_CONFIG) -> dict[str, ModelVariant]:
    """Load model variants from YAML, keyed by id."""
    raw = yaml.safe_load(path.read_text())
    variants = [ModelVariant.model_validate(m) for m in raw["models"]]
    ids = [v.id for v in variants]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate model ids: {duplicates}")
    return {v.id: v for v in variants}
