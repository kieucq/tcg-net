from .__prototype__ import *
from config_loader import CONFIG

class Ibtracs(HurricaneTrack):
    SUPPORTED_BASINS = ("WP", "NA")

    def __init__(self):
        self.DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
        self.TIME_POINTS_HOURS = [0,6,12,18]
        self.TIME_POINTS_MINS = [0]
        self.TIME_POINTS_SECS = [0]
        self.DATETIME_COL_INX = ["ISO_TIME"]
        self.SKIP_ROWS = [1]
        self.SEPERATOR = ","
        self.COLUMNS = ["SID", "ISO_TIME", "LAT", "LON", "BASIN","SUBBASIN"]
        self.FILTERS = []

    @staticmethod
    def _normalize_longitudes(longitudes: pd.Series, min_lon: float, max_lon: float) -> pd.Series:
        """Convert IBTrACS longitudes to the convention used by PRE_DOMAIN."""
        longitudes = pd.to_numeric(longitudes, errors="coerce")
        if 0 <= min_lon <= 360 and 0 <= max_lon <= 360:
            return longitudes % 360
        if -180 <= min_lon <= 180 and -180 <= max_lon <= 180:
            return ((longitudes + 180) % 360) - 180
        return longitudes

    @staticmethod
    def _longitude_in_domain(longitudes: pd.Series, min_lon: float, max_lon: float) -> pd.Series:
        if min_lon <= max_lon:
            return longitudes.between(min_lon, max_lon, inclusive="both")
        return (longitudes >= min_lon) | (longitudes <= max_lon)

    def _select_basin_from_domain(self, dataframe: pd.DataFrame) -> str:
        min_lat = float(CONFIG.PRE_DOMAIN.MIN_LAT)
        max_lat = float(CONFIG.PRE_DOMAIN.MAX_LAT)
        min_lon = float(CONFIG.PRE_DOMAIN.MIN_LON)
        max_lon = float(CONFIG.PRE_DOMAIN.MAX_LON)

        if min_lat > max_lat:
            raise ValueError(
                f"Invalid PRE_DOMAIN latitude range: MIN_LAT={min_lat} exceeds MAX_LAT={max_lat}"
            )

        basin_candidates = dataframe[dataframe["BASIN"].isin(self.SUPPORTED_BASINS)]
        in_domain = basin_candidates[
            basin_candidates["LAT"].between(min_lat, max_lat, inclusive="both")
            & self._longitude_in_domain(basin_candidates["LON"], min_lon, max_lon)
        ]
        basin_counts = in_domain.groupby("BASIN")["SID"].nunique()
        matching_basins = basin_counts[basin_counts > 0]

        if matching_basins.empty:
            raise ValueError(
                "PRE_DOMAIN does not contain the first track center of any supported "
                f"IBTrACS basin {self.SUPPORTED_BASINS}: "
                f"latitude=[{min_lat}, {max_lat}], longitude=[{min_lon}, {max_lon}]"
            )
        if len(matching_basins) > 1:
            raise ValueError(
                "PRE_DOMAIN overlaps multiple supported IBTrACS basins; narrow the "
                f"domain to select one basin. Storm counts: {matching_basins.to_dict()}"
            )

        basin = str(matching_basins.index[0])
        self.FILTERS = [("BASIN", [basin])]
        print(
            f"IBTrACS basin selected from PRE_DOMAIN: {basin} "
            f"({int(matching_basins.iloc[0])} storm centers inside domain)"
        )
        return basin

    # This function is used to load raw *.csv from data provider
    # then merge them as one panda.dataframe
    def LoadRawCSVs(self, paths:list[str]) -> pd.DataFrame:
        df_list = [
            pd.read_csv(
                filepath_or_buffer=p,
                sep=self.SEPERATOR,
                date_format=self.DATETIME_FORMAT,
                parse_dates=self.DATETIME_COL_INX,
                index_col=False,
                low_memory=False,
                skiprows=self.SKIP_ROWS,
                keep_default_na=False,
                na_values=["", " "],
            )
            for p in paths
        ]
        dataframe = pd.concat(df_list)
        return dataframe

    # This function is used to process the loadded raw dataframe
    def ProcessRaw(self, dataframe:pd.DataFrame, filter_first:bool=True) -> pd.DataFrame:
        temp_df = dataframe.copy()
        temp_df["ISO_TIME"] = pd.to_datetime(temp_df["ISO_TIME"], errors="coerce")
        temp_df["LAT"] = pd.to_numeric(temp_df["LAT"], errors="coerce")
        temp_df["LON"] = pd.to_numeric(temp_df["LON"], errors="coerce")
        temp_df["BASIN"] = temp_df["BASIN"].astype("string").str.strip()

        # Remove time points
        temp_df = temp_df.where(
            temp_df["ISO_TIME"].dt.time.isin([datetime.time(h, m, s) for h in self.TIME_POINTS_HOURS for m in self.TIME_POINTS_MINS for s in self.TIME_POINTS_SECS])
        )
        # Get only the first record
        if filter_first:
            temp_df = temp_df.sort_values(by="ISO_TIME", ascending=False).groupby("SID").tail(1).sort_values(by="SID", ascending=True)

        min_lon = float(CONFIG.PRE_DOMAIN.MIN_LON)
        max_lon = float(CONFIG.PRE_DOMAIN.MAX_LON)
        temp_df["LON"] = self._normalize_longitudes(temp_df["LON"], min_lon, max_lon)
        self._select_basin_from_domain(temp_df)

        # Select the basin whose track centers overlap PRE_DOMAIN.
        for f in self.FILTERS:
            temp_df = temp_df.where(
                temp_df[f[0]].isin(f[1])
            )
        # Get only columns of interest
        if (self.COLUMNS):
            temp_df = temp_df[self.COLUMNS]
        temp_df.dropna(inplace=True)
        return temp_df
