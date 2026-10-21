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
    EXTRACTION = "extraction"
    FEATURES = "features"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    LEARNING = "learning"

    @property
    def order(self) -> int:
        return list(Stage).index(self)


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
            unknown = set(item) - {"start", "end", "step"}
            if unknown:
                raise ValueError(f"Unknown date range keys {sorted(unknown)}")
            start = _parse_date_value(item.get("start"))
            end = _parse_date_value(item.get("end"))
            if end < start:
                raise ValueError(f"Date range ends before it starts: {item!r}")

            curr = start
            step = item.get("step", 1)
            # a zero or fractional step would never move past `end`
            if not isinstance(step, int) or isinstance(step, bool) or step < 1:
                raise ValueError(
                    f"Date range step must be a positive integer: {step!r}"
                )
            while curr <= end:
                parsed_dates.add(curr)
                curr += timedelta(days=step)
        else:
            raise TypeError(f"Unsupported dates entry: {item!r}")

    return sorted(parsed_dates)


class CleaningConfig(BaseModel):
    """Options of cleaning.clean_rules, applied to every learned model."""

    model_config = ConfigDict(extra="forbid")

    min_support: int = Field(0, ge=0)
    min_exception_support: int = Field(0, ge=0)
    # MDL pruning, lambda per 1000 minority-side training rows (None = off)
    mdl_lambda: float | None = Field(None, ge=0)
    # thresholds to round values of their unit
    snap: bool = False


class FoldRmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["one_vs_rest", "multiclass", "ordinal", "ordinal_count"] = (
        "one_vs_rest"
    )
    ratio: float = Field(0.7, gt=0)
    split: Literal["homogeneous", "date", "row"] = "homogeneous"
    test_ratio: float = Field(0.3, gt=0, lt=1)
    seed: int = 42
    cleaning: CleaningConfig = CleaningConfig()
    # target -> feature column prefixes left out of its models
    exclude_features: dict[str, list[str]] = {}
    gpu: bool = False
    output: str = "fold_rm"

    @model_validator(mode="after")
    def _check_cleaning(self) -> FoldRmConfig:
        # one head label per model, as only the ordinal strategies learn
        uses_rules = self.cleaning.mdl_lambda is not None or self.cleaning.snap
        if uses_rules and not self.strategy.startswith("ordinal"):
            raise ValueError("cleaning.mdl_lambda / snap need an ordinal strategy")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str | list[float] | dict[str, Any] = "FVG"
    dates: list[date]
    output_path: Path
    workers: PositiveInt = 12
    clustering: bool = False
    save_images: bool = False
    debug: bool = False
    force: Stage | None = None
    stop_after: Stage = Stage.LEARNING
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
        if isinstance(self.region, dict):
            unknown = set(self.region) - {"name", "bounds", "radius", "cities"}
            if unknown:
                raise ValueError(f"region: unknown keys {sorted(unknown)}")
            if {"name", "bounds"} <= set(self.region):
                raise ValueError("region: set either 'name' or 'bounds', not both")
        try:
            self.build_region()
        except (KeyError, TypeError) as e:  # e.g. a city without lat/lon
            raise ValueError(f"region: invalid value ({e!r})") from e
        return self

    def build_region(self) -> Region:
        return Region.from_config(self.region)

    def forces(self, stage: Stage) -> bool:
        """True if `stage` must be recomputed even when cached results exist."""
        return self.force is not None and stage.order >= self.force.order

    def runs(self, stage: Stage) -> bool:
        """True if `stage` is reached before the run stops (see `stop_after`)."""
        return stage.order <= self.stop_after.order


def _merge(base: dict, override: dict) -> dict:
    """Recursive merge, `override` wins. A run's region cities and dates replace
    the default ones, and a region name or bounds replaces both the default name
    and bounds."""
    merged = {**base}
    for key, value in override.items():
        base_value = merged.get(key)
        if key in ("cities", "dates") or not (
            isinstance(value, dict) and isinstance(base_value, dict)
        ):
            merged[key] = value
            continue
        if key == "region" and {"name", "bounds"} & set(value):
            base_value = {
                k: v for k, v in base_value.items() if k not in ("name", "bounds")
            }
        merged[key] = _merge(base_value, value)
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
