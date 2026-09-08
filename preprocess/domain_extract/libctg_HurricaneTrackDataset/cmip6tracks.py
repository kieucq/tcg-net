from .__prototype__ import *

class Cmip6Tracks(HurricaneTrack):
    def __init__(self):
        self.DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
        self.SEPERATOR = ","
        self.RAW_COLUMNS = ["SID", "YEAR", "DAY", "LON", "LAT", "PMIN", "VMAX", "LIFETIME"] 
        self.DROP_COLUMNS = ["YEAR", "DAY", "PMIN", "VMAX", "LIFETIME"]

    # This function is used to load raw *.txt from data provider
    # then merge them as one panda.dataframe
    def LoadRawTxts(self, paths:list[str]) -> pd.DataFrame:
        data = [
            line.strip().split()
            for path in paths
            for line in open(path, "r")
        ]
        return pd.DataFrame(data, columns=self.RAW_COLUMNS)

    # This function is used to process the loadded raw dataframe
    def ProcessRaw(self, dataframe:pd.DataFrame, filter_first:bool=True) -> pd.DataFrame:
        temp_df = dataframe.copy(deep=True)
        # Convert YEAR and DAY to ISO_TIME
        temp_df["ISO_TIME"] = pd.to_datetime(temp_df["YEAR"].astype(str) + temp_df["DAY"].astype(str), format='%Y%j')
        temp_df["ISO_TIME"] = temp_df["ISO_TIME"].dt.strftime(self.DATETIME_FORMAT)
        # Drop unnessary columns
        temp_df.drop(columns=self.DROP_COLUMNS, inplace=True)
        # Get only the first record
        temp_df["SID"] = pd.to_numeric(temp_df["SID"])
        if filter_first:
            temp_df = temp_df.sort_values(by="ISO_TIME", ascending=False).groupby("SID").tail(1).sort_values(by="SID", ascending=True)
        # Only in West Pacific WP
        # Convert LAT, LON to numeric values
        temp_df["LAT"] = pd.to_numeric(temp_df["LAT"])
        temp_df["LON"] = pd.to_numeric(temp_df["LON"])   
        
        # Filter based on LAT and LON range
        temp_df = temp_df[(temp_df["LAT"] <= 70) & (temp_df["LAT"] >= -50) & (temp_df["LON"] <= 220) & (temp_df["LON"] >= 60)]

        temp_df.dropna(inplace=True)
        return temp_df
