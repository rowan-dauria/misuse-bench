from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG = Path("configs/models.yaml")


class ModelVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    hf_id: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    variant: Literal["original", "abliterated", "fine-tuned"]
    release_date: date | None = None
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
