from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)

from region import Region


class Stage(StrEnum):
    DATA = "data"
    FEATURES = "features"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    LEARNING = "learning"

    @property
    def order(self) -> int:
        return list(Stage).index(self)


CUT = "cut"
CleanTarget = Literal["grib", "cut", "extracted"]


def _parse_date_value(value: datetime | date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)

    raise ValueError(f"Unsupported date value: {value!r}")


def parse_dates(dates_entry: list | str | dict | None) -> list[date]:
    """
    Parses a date configuration entry, which can be a single date item or a list of items.
    """

    if not dates_entry:
        return []

    if not isinstance(dates_entry, list):
        dates_entry = [dates_entry]

    parsed_dates = set()

    for item in dates_entry:
        if isinstance(item, (datetime, date, str)):
            parsed_dates.add(_parse_date_value(item))
        elif isinstance(item, dict):
            start = _parse_date_value(item.get("start"))
            end = _parse_date_value(item.get("end"))

            curr = start
            step = item.get("step", 1)
            while curr <= end:
                parsed_dates.add(curr)
                curr += timedelta(days=step)
        else:
            raise TypeError(f"Unsupported dates entry: {item!r}")

    return sorted(parsed_dates)


class FoldRmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["one_vs_rest", "multiclass", "ordinal"] = "one_vs_rest"
    ratio: float = Field(0.7, gt=0, le=1)
    split: Literal["date", "row"] = "date"
    test_ratio: float = Field(0.3, gt=0, lt=1)
    seed: int = 42
    min_support: int = Field(0, ge=0)
    gpu: bool = False


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str | list[float] | dict[str, Any] = "FVG"
    cities: dict[str, dict[str, float]] | None = None
    dates: list[date]
    output_path: Path
    workers: PositiveInt = 12
    clustering: bool = False
    save_images: bool = False
    debug: bool = False
    force: Stage | None = None
    stop_after: Stage | Literal["cut"] = Stage.LEARNING
    clean: list[CleanTarget] = []
    fold_rm: FoldRmConfig = FoldRmConfig()

    @field_validator("dates", mode="before")
    @classmethod
    def _parse_dates(cls, value: Any) -> list[date]:
        try:
            dates = parse_dates(value)
        except TypeError as e:  # pydantic only reports ValueErrors as field errors
            raise ValueError(e) from e
        if not dates:
            raise ValueError("at least one date is required")
        return dates

    @model_validator(mode="after")
    def _check_region(self) -> RunConfig:
        self.build_region()
        return self

    def build_region(self) -> Region:
        return Region.from_config(self.region, cities=self.cities)

    def forces(self, stage: Stage) -> bool:
        """True if `stage` must be recomputed even when cached results exist."""
        return self.force is not None and stage.order >= self.force.order

    def runs(self, stage: Stage) -> bool:
        """True if `stage` is reached before the run stops (see `stop_after`)."""
        if self.stop_after == CUT:
            return stage == Stage.DATA
        return stage.order <= self.stop_after.order


def _merge(base: dict, override: dict) -> dict:
    merged = {**base, **override}
    if isinstance(base.get("fold_rm"), dict) and isinstance(
        override.get("fold_rm"), dict
    ):
        merged["fold_rm"] = {**base["fold_rm"], **override["fold_rm"]}
    return merged


def load_runs(
    config_path: Path, cli_overrides: dict[str, Any] | None = None
) -> dict[str, RunConfig]:
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: expected a mapping of run names to runs")

    run_keys = set(RunConfig.model_fields) & set(raw)
    if run_keys:
        raise ValueError(f"{config_path}: {sorted(run_keys)} must be inside a run")

    defaults = raw.pop("defaults", None) or {}
    if not raw:
        raise ValueError(f"{config_path}: no runs found")

    runs = {}
    errors = []
    for name, block in raw.items():
        if not isinstance(block, dict):
            errors.append(f"[{name}] expected a mapping of options")
            continue
        data = _merge(_merge(defaults, block), cli_overrides or {})
        data.setdefault("output_path", Path("runs") / name)
        try:
            runs[name] = RunConfig.model_validate(data)
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(map(str, err["loc"]))
                msg = err["msg"].removeprefix("Value error, ")
                errors.append(f"[{name}] {loc}: {msg}" if loc else f"[{name}] {msg}")

    if errors:
        raise ValueError("Invalid config:\n" + "\n".join(errors))

    return runs
