#!/usr/bin/env python3
"""Generate channel statistics used to normalize the dynamic-model dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_LIB = REPO_ROOT / "models" / "lib"
if str(MODELS_LIB) not in sys.path:
    sys.path.insert(0, str(MODELS_LIB))

from Utils.New_features import divergence, meshgrid, vorticity  # noqa: E402


@dataclass
class RunningStatistics:
    """Numerically stable, mergeable statistics for one model channel."""

    count: int = 0
    minimum: float = np.inf
    maximum: float = -np.inf
    total: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return

        batch_count = int(finite.size)
        batch_total = float(np.sum(finite, dtype=np.float64))
        batch_mean = batch_total / batch_count
        batch_m2 = float(np.sum((finite - batch_mean) ** 2, dtype=np.float64))

        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            combined_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / combined_count
            self.m2 += batch_m2 + delta**2 * self.count * batch_count / combined_count

        self.count += batch_count
        self.total += batch_total
        self.minimum = min(self.minimum, float(np.min(finite)))
        self.maximum = max(self.maximum, float(np.max(finite)))

    def as_row(self, variable: str, level: int) -> dict[str, float | int | str]:
        if self.count == 0:
            raise ValueError(f"No finite values were found for {variable} at level {level}")

        variance = max(self.m2 / self.count, 0.0)
        return {
            "variable": variable,
            "level": level,
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "sum": self.total,
            "mean": self.mean,
            "variance": variance,
            "std": float(np.sqrt(variance)),
        }


def load_model_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    try:
        return config["DYNAMIC_MODEL_DATASET"]
    except KeyError as exc:
        raise KeyError(f"DYNAMIC_MODEL_DATASET is missing from {config_path}") from exc


def resolve_config_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def channel_specs(model_config: dict) -> list[tuple[str, int]]:
    pressure_levels = [int(level) for level in model_config["PRESS_LEVEL"]]
    specs = [(variable, 0) for variable in model_config["SINGLE_VAR"]]
    specs.extend(
        (variable, level)
        for variable in model_config["PRESS_VAR"]
        for level in pressure_levels
    )
    specs.extend(
        (variable, level)
        for variable in model_config.get("ADD_VAR", [])
        for level in pressure_levels
    )
    if len(specs) != len(set(specs)):
        raise ValueError("DYNAMIC_MODEL_DATASET defines duplicate variable/level channels")
    return specs


def surface_array(dataset: xr.Dataset, variable: str) -> np.ndarray:
    if variable not in dataset.variables:
        raise KeyError(f"Variable {variable!r} is missing")
    values = np.asarray(dataset.variables[variable].data).squeeze()
    if values.ndim != 2:
        raise ValueError(f"Surface variable {variable} has shape {values.shape}; expected 2-D")
    return values


def pressure_array(dataset: xr.Dataset, variable: str, level_count: int) -> np.ndarray:
    if variable not in dataset.variables:
        raise KeyError(f"Variable {variable!r} is missing")
    values = np.asarray(dataset.variables[variable].data).squeeze()
    if values.ndim != 3 or values.shape[0] < level_count:
        raise ValueError(
            f"Pressure variable {variable} has shape {values.shape}; "
            f"expected at least {level_count} vertical levels"
        )
    return values[:level_count]


def warn_if_pressure_levels_differ(
    dataset: xr.Dataset,
    pressure_levels: list[int],
) -> None:
    """Warn when config labels differ from the positional levels used by training."""
    coordinate = "isobaricInhPa"
    if coordinate not in dataset.coords:
        warnings.warn(
            f"Pressure coordinate {coordinate!r} is missing. Statistics will follow "
            "Merra2_full and use the first configured number of array levels.",
            stacklevel=2,
        )
        return

    actual = np.asarray(dataset.coords[coordinate].data).squeeze()
    expected = np.asarray(pressure_levels)
    if actual.size < expected.size or not np.allclose(actual[: expected.size], expected):
        warnings.warn(
            f"The first pressure levels in the data are "
            f"{actual[:expected.size].tolist()}, but config.json labels them as "
            f"{pressure_levels}. Statistics will follow Merra2_full exactly: use "
            "the first configured number of levels by position and write the "
            "config labels to the workbook.",
            stacklevel=2,
        )


def model_channels(
    dataset: xr.Dataset,
    model_config: dict,
) -> Iterator[tuple[tuple[str, int], np.ndarray]]:
    pressure_levels = [int(level) for level in model_config["PRESS_LEVEL"]]
    level_count = len(pressure_levels)
    warn_if_pressure_levels_differ(dataset, pressure_levels)

    for variable in model_config["SINGLE_VAR"]:
        yield (variable, 0), surface_array(dataset, variable)

    pressure_data: dict[str, np.ndarray] = {}
    for variable in model_config["PRESS_VAR"]:
        values = pressure_array(dataset, variable, level_count)
        pressure_data[variable] = values
        for index, level in enumerate(pressure_levels):
            yield (variable, level), values[index]

    derived_variables = list(model_config.get("ADD_VAR", []))
    unsupported = set(derived_variables) - {"VOR", "DIV"}
    if unsupported:
        raise ValueError(f"Unsupported derived variables: {sorted(unsupported)}")
    if not derived_variables:
        return

    u_wind = pressure_data.get("U")
    if u_wind is None:
        u_wind = pressure_array(dataset, "U", level_count)
    v_wind = pressure_data.get("V")
    if v_wind is None:
        v_wind = pressure_array(dataset, "V", level_count)

    longitude = np.asarray(dataset.coords["longitude"].data)
    latitude = np.asarray(dataset.coords["latitude"].data)[::-1]
    latitude_grid, longitude_grid = meshgrid(latitude, longitude, level_count)

    derived_data = {
        "VOR": vorticity(u_wind, v_wind, latitude_grid, longitude_grid),
        "DIV": divergence(u_wind, v_wind, latitude_grid, longitude_grid),
    }
    for variable in derived_variables:
        for index, level in enumerate(pressure_levels):
            yield (variable, level), derived_data[variable][index]


def training_paths(train_csv: Path) -> list[Path]:
    dataframe = pd.read_csv(train_csv)
    required_columns = {"Path", "Label"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{train_csv} is missing columns: {sorted(missing_columns)}")

    dataframe = dataframe[dataframe["Label"].notna()]
    if dataframe.empty:
        raise ValueError(f"{train_csv} contains no labeled training samples")

    paths = []
    for value in dataframe["Path"]:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = train_csv.parent / path
        paths.append(path.resolve())
    return paths


def calculate_statistics(
    paths: list[Path],
    model_config: dict,
    progress_every: int,
) -> pd.DataFrame:
    specs = channel_specs(model_config)
    statistics = {spec: RunningStatistics() for spec in specs}

    for index, path in enumerate(paths, start=1):
        if not path.is_file():
            raise FileNotFoundError(f"Training sample does not exist: {path}")
        try:
            with xr.open_dataset(path) as dataset:
                observed_specs = set()
                for spec, values in model_channels(dataset, model_config):
                    statistics[spec].update(values)
                    observed_specs.add(spec)
        except Exception as exc:
            raise RuntimeError(f"Failed to process training sample {path}") from exc

        missing_specs = set(specs) - observed_specs
        if missing_specs:
            raise ValueError(f"Sample {path} is missing model channels: {sorted(missing_specs)}")
        if progress_every > 0 and (index % progress_every == 0 or index == len(paths)):
            print(f"Processed {index}/{len(paths)} training samples")

    rows = [statistics[spec].as_row(*spec) for spec in specs]
    return pd.DataFrame(
        rows,
        columns=["variable", "level", "count", "min", "max", "sum", "mean", "variance", "std"],
    )


def write_excel(dataframe: pd.DataFrame, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        dataframe.to_excel(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate normalization statistics from labeled training NetCDF samples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config.json",
        help="Configuration file (default: repository config.json)",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--train-csv",
        type=Path,
        help="Training CSV; defaults to DYNAMIC_MODEL_DATASET.TRAIN_PATH",
    )
    input_group.add_argument(
        "--inp-dir",
        type=Path,
        help="Directory containing train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output workbook; defaults to DYNAMIC_MODEL_DATASET.STATISTIC_PATH",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N samples; use 0 to disable (default: 100)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output workbook",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    model_config = load_model_config(config_path)

    if args.train_csv is not None:
        train_csv = args.train_csv.expanduser().resolve()
    elif args.inp_dir is not None:
        train_csv = (args.inp_dir.expanduser() / "train.csv").resolve()
    else:
        train_csv = resolve_config_path(model_config["TRAIN_PATH"], config_path)

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
    else:
        output_path = resolve_config_path(model_config["STATISTIC_PATH"], config_path)

    paths = training_paths(train_csv)
    print(f"Training CSV: {train_csv}")
    print(f"Labeled samples: {len(paths)}")
    print(f"Output workbook: {output_path}")

    dataframe = calculate_statistics(paths, model_config, args.progress_every)
    write_excel(dataframe, output_path, args.overwrite)
    print(f"Wrote {len(dataframe)} channel-statistic rows to {output_path}")


if __name__ == "__main__":
    main()
