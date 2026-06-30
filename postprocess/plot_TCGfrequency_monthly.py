import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from config_loader import CONFIG

def load_ibtracs(file_ibtracs, start_year=None, end_year=None):
    """Đọc và xử lý dữ liệu IBTrACS"""
    df = pd.read_csv(file_ibtracs)

    # Chuyển đổi Datetime
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors="coerce")
    df[['LAT', 'LON']] = df[['LAT', 'LON']].astype(float)

    # Lọc theo vùng
    lat_min = CONFIG.SLICING_WINDOW.AREA.MIN_LAT #0
    lat_max = CONFIG.SLICING_WINDOW.AREA.MAX_LAT #30
    lon_min = CONFIG.SLICING_WINDOW.AREA.MIN_LON #100
    lon_max = CONFIG.SLICING_WINDOW.AREA.MAX_LON #150
    df = df[df['LAT'].between(lat_min, lat_max, inclusive='both') &
            df['LON'].between(lon_min, lon_max, inclusive='both')]

    # Chuẩn hóa tên cột
    df = df.rename(columns={'LAT': 'Lat',
                            'LON': 'Lon',
                            'ISO_TIME': 'Datetime'})
    df = df[['Datetime', 'Lat', 'Lon']]

    # Lọc theo năm (nếu có)
    if start_year and end_year:
        df = df[df['Datetime'].dt.year.between(start_year, end_year, inclusive='both')]

    # Thêm cột tháng
    df['Month'] = df['Datetime'].dt.month

    # Đếm số bão theo tháng
    df_truth = df.groupby('Month').size().reset_index(name='IBTrACS')

    # Đảm bảo đủ 12 tháng (nếu không có dữ liệu tháng nào thì fill 0)
    all_months = pd.DataFrame({'Month': range(1, 13)})
    df_truth = pd.merge(all_months, df_truth, on='Month', how='left').fillna(0)

    print(df_truth)

    return df_truth


def plot_yearly(storm_counts, score_cols, xticklabels, df_ibtracs, plot_ibtracs=False):
    """Vẽ và lưu biểu đồ theo từng năm riêng lẻ"""
    # normalize per year
    yearly_totals = storm_counts.groupby("Year")[score_cols].transform("sum")
    storm_counts[score_cols] = storm_counts[score_cols].div(yearly_totals).fillna(0)
    for year, yearly_data in storm_counts.groupby("Year"):
        plt.figure(figsize=(14, 7))

        for col in score_cols:
            plt.plot(
                yearly_data["Month"] - 1, 
                yearly_data[col], 
                marker="o", 
                label=col, 
                zorder=3
            )

        # IBTrACS cho từng năm
        print(f"plot_ibtracs = {plot_ibtracs}")
        if plot_ibtracs and df_ibtracs is not None:
            df_truth_year = load_ibtracs(df_ibtracs, year, year)
            print("df_truth_year")
            print(df_truth_year)
            print(df_truth_year["IBTrACS"].sum())
            df_truth_year["IBTrACS"] = df_truth_year["IBTrACS"] / df_truth_year["IBTrACS"].sum()
            print("df_truth_year normalized")
            print(df_truth_year)
            plt.plot(
                df_truth_year["Month"] - 1,
                df_truth_year["IBTrACS"],
                marker="s",
                color="black",
                linewidth=3,
                markersize=10,
                label="IBTrACS",
                zorder=5
            )


        # Cấu hình biểu đồ
        plt.xticks(range(12), labels=xticklabels, fontsize=20, rotation=30)
        plt.yticks(fontsize=20)
        plt.ylabel('Density of Positive prediction\n', fontsize=26)
        plt.title(f"Storms Distribution – {year}", fontsize=26)

        legend = plt.legend(
            title='Lead time forecast',
            ncol=1,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=20,
            borderaxespad=0
        )
        legend.get_title().set_fontsize(24)

        plt.grid(axis='y', zorder=0, alpha=0.7)
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.gca().spines[['left', 'bottom']].set_linewidth(1)

        # Lưu hình cho từng năm
        plt.savefig(f"{CONFIG.TCG_FREQUENCY.OUTPUT_FREQUENCY}/TCGfrequency_{year}.eps", bbox_inches='tight', format='eps')
        plt.savefig(f"{CONFIG.TCG_FREQUENCY.OUTPUT_FREQUENCY}/TCGfrequency_{year}.pdf", bbox_inches='tight', format='pdf')

        plt.close()
        print(f"Đã lưu biểu đồ cho năm {year}")


def plot_duration(storm_counts, score_cols, xticklabels, start_year, end_year, df_ibtracs, plot_ibtracs=False):
    """Vẽ và lưu biểu đồ tổng theo một khoảng năm"""
    data = storm_counts[(storm_counts["Year"] >= start_year) & (storm_counts["Year"] <= end_year)]
    duration_sum = data.groupby("Month")[score_cols].sum().reset_index()
    duration_sum[score_cols] = duration_sum[score_cols].div(duration_sum[score_cols].sum()).fillna(0)

    plt.figure(figsize=(14, 7))

    for col in score_cols:
        plt.plot(
            duration_sum["Month"] - 1,
            duration_sum[col],
            marker="o",
            label=col,
            zorder=3
        )

    # IBTrACS cho giai đoạn
    if plot_ibtracs and df_ibtracs is not None:
        df_truth_duration = load_ibtracs(df_ibtracs, start_year, end_year)
        plt.plot(
            df_truth_duration["Month"] - 1,
            df_truth_duration["IBTrACS"],
            marker="s",
            color="black",
            linewidth=3,
            label="IBTrACS",
            zorder=5
        )

    # Cấu hình biểu đồ
    plt.xticks(range(12), labels=xticklabels, fontsize=20, rotation=30)
    plt.yticks(fontsize=20)
    plt.ylabel('Density of Positive prediction\n', fontsize=26)
    plt.title(f"Storms Distribution – {start_year}-{end_year}", fontsize=26)

    legend = plt.legend(
        title='Lead time forecast',
        ncol=1,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=20,
        borderaxespad=0
    )
    legend.get_title().set_fontsize(24)

    plt.grid(axis='y', zorder=0, alpha=0.7)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.gca().spines[['left', 'bottom']].set_linewidth(1)

    # Lưu hình cho khoảng năm
    plt.savefig(f"{CONFIG.TCG_FREQUENCY.OUTPUT_FREQUENCY}/TCGfrequency_{start_year}_{end_year}.eps", bbox_inches='tight', format='eps')
    plt.savefig(f"{CONFIG.TCG_FREQUENCY.OUTPUT_FREQUENCY}/TCGfrequency_{start_year}_{end_year}.pdf", bbox_inches='tight', format='pdf')

    plt.close()
    print(f"Đã lưu biểu đồ cho giai đoạn {start_year}-{end_year}")


def main():
    csv_file = CONFIG.TCG_FREQUENCY.PREDICT_CSV_FILE

    # Tự động tìm file IBTrACS
    ibtracs_folder = CONFIG.TCG_FREQUENCY.IBTRACS_FOLDER
    ibtracs_file = None
    if ibtracs_folder and os.path.isdir(ibtracs_folder):
        matches = glob.glob(os.path.join(ibtracs_folder, "FIRST_*.csv"))
        if matches:
            ibtracs_file = matches[0]
            print(f"Dùng IBTrACS file: {ibtracs_file}")
        else:
            print(f"Không tìm thấy file FIRST_*.csv trong {ibtracs_folder}")

    year_duration = CONFIG.TCG_FREQUENCY.YEAR_DURATION
    plot_ibtracs = CONFIG.TCG_FREQUENCY.PLOT_IBTRACS
    plot_duration_flag = CONFIG.TCG_FREQUENCY.PLOT_DURATION

    print(f"plot_ibtracs = {plot_ibtracs}, plot_duration = {plot_duration_flag}, year_duration = {year_duration}")

    df = pd.read_csv(csv_file)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Year"] = df["Datetime"].dt.year
    df["Month"] = df["Datetime"].dt.month

    score_cols = [col for col in df.columns if col.startswith("Score_step")]
    if not score_cols:
        print("Không tìm thấy cột nào dạng Score_stepx trong file CSV!")
        return

    # Tạo lưới đầy đủ
    all_years = sorted(df["Year"].unique())
    all_months = range(1, 13)
    full_index = pd.MultiIndex.from_product([all_years, all_months], names=["Year", "Month"])

    # Đếm bão
    storm_counts = (
        df.groupby(["Year", "Month"])[score_cols]
        .apply(lambda g: (g > 0.5).sum())
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    xticklabels = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    if plot_duration_flag and year_duration:
        try:
            start_year, end_year = map(int, year_duration.split("-"))
            plot_duration(storm_counts, score_cols, xticklabels,
                          start_year, end_year, ibtracs_file, plot_ibtracs)
        except Exception:
            print("Sai định dạng YEAR_DURATION. Dùng: YYYY-YYYY")
    else:
        plot_yearly(storm_counts, score_cols, xticklabels, ibtracs_file, plot_ibtracs)


if __name__ == "__main__":    
    main()

