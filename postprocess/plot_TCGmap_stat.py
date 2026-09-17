import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from config_loader import CONFIG

OUTPUT_DIR = "./output/postprocess"

def forecast_step(column_name):
    """Return the numeric lead time from a Score_step<N> column name."""
    return int(column_name.removeprefix("Score_step"))


def configured_map_bounds():
    """Return plotting bounds padded by half the configured sampling interval."""
    area = CONFIG.SLICING_WINDOW.AREA
    child_resolution = CONFIG.SLICING_WINDOW.CHILD_AREA.RESOLUTION
    number_of_steps = CONFIG.SLICING_WINDOW.NUM_STEP

    latitude_spacing = float(child_resolution.LAT_STEP) * int(number_of_steps.LAT_NSTEP)
    longitude_spacing = float(child_resolution.LON_STEP) * int(number_of_steps.LON_NSTEP)

    return (
        float(area.MIN_LAT) - latitude_spacing / 2,
        float(area.MAX_LAT) + latitude_spacing / 2,
        float(area.MIN_LON) - longitude_spacing / 2,
        float(area.MAX_LON) + longitude_spacing / 2,
    )

def coordinate_ticks(minimum, maximum, interval=10):
    start = np.floor(minimum / interval) * interval
    stop = np.ceil(maximum / interval) * interval
    return np.arange(start, stop + interval, interval)

def plot_all_case_statistics():
    """Plot mean TCG probability at each grid point across all input cases."""
    csv_file = CONFIG.POSTPROCESS.PREDICT_CSV_FILE
    dataframe = pd.read_csv(csv_file)

    required_columns = {"Datetime", "Point"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{csv_file} is missing columns: {sorted(missing_columns)}")

    score_columns = sorted(
        (column for column in dataframe.columns if column.startswith("Score_step")),
        key=forecast_step,
    )
    if not score_columns:
        raise ValueError(f"{csv_file} does not contain any Score_step<N> columns")

    point_coordinates = dataframe["Point"].str.extract(
        r"^(?P<Lat>-?\d+(?:\.\d+)?)_(?P<Lon>-?\d+(?:\.\d+)?)$"
    )
    if point_coordinates.isna().any(axis=None):
        invalid_points = dataframe.loc[point_coordinates.isna().any(axis=1), "Point"].unique()
        raise ValueError(f"Invalid Point values in {csv_file}: {invalid_points[:5].tolist()}")

    dataframe[["Lat", "Lon"]] = point_coordinates.astype(float)
    dataframe["Datetime"] = pd.to_datetime(dataframe["Datetime"], errors="raise")

    latitude_values = np.sort(dataframe["Lat"].unique())
    longitude_values = np.sort(dataframe["Lon"].unique())
    longitude_grid, latitude_grid = np.meshgrid(longitude_values, latitude_values)
    lat_min, lat_max, lon_min, lon_max = configured_map_bounds()
    case_count = dataframe["Datetime"].nunique()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_paths = []
    for score_column in score_columns:
        step = forecast_step(score_column)
        mean_probability = dataframe.pivot_table(
            index="Lat",
            columns="Lon",
            values=score_column,
            aggfunc="mean",
        ).reindex(index=latitude_values, columns=longitude_values)

        figure = plt.figure(figsize=(9, 5))
        map_ax = Basemap(
            projection="cyl",
            llcrnrlat=lat_min,
            urcrnrlat=lat_max,
            llcrnrlon=lon_min,
            urcrnrlon=lon_max,
            resolution="c",
        )
        map_ax.fillcontinents(color="lightgray", lake_color="white", zorder=2)
        map_ax.drawcountries(linewidth=0.8, zorder=3)
        map_ax.drawcoastlines(linewidth=0.8, zorder=3)
        map_ax.drawparallels(
            coordinate_ticks(lat_min, lat_max),
            labels=[1, 0, 0, 0],
            linewidth=0.5,
        )
        map_ax.drawmeridians(
            coordinate_ticks(lon_min, lon_max),
            labels=[0, 0, 0, 1],
            linewidth=0.5,
        )

        color_mesh = map_ax.pcolormesh(
            longitude_grid,
            latitude_grid,
            np.ma.masked_invalid(mean_probability.to_numpy()),
            shading="auto",
            cmap="coolwarm",
            vmin=0,
            vmax=0.3,
            zorder=1,
        )
        figure.colorbar(
            color_mesh,
            ax=plt.gca(),
            orientation="vertical",
            shrink=0.8,
            pad=0.05,
            label="Mean TCG probability",
        )
        plt.title(f"All-case mean TCG probability — forecast lead {step} h\n{case_count} cases")
        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, f"stat_step{step}.pdf")
        figure.savefig(output_path, bbox_inches="tight", format="pdf")
        plt.close(figure)
        output_paths.append(output_path)
        print(f"Saved: {output_path}")

    return output_paths

# Preserve the original callable name for existing imports.
plot_dynamic = plot_all_case_statistics
