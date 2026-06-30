import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from config_loader import CONFIG

csv_file = CONFIG.TCG_FREQUENCY.PREDICT_CSV_FILE
os.makedirs(f"./output/postprocess/", exist_ok=True)

def plot_dynamic():
    df = pd.read_csv(csv_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Lat'] = df['Point'].str.split('_').str[0].astype(float)
    df['Lon'] = df['Point'].str.split('_').str[1].astype(float)

#    for name, grp in df.groupby('Datetime'):
    for step in range(0, 20, 2):
            plt.figure(figsize=(6, 5))

            # Unique sorted coords
            lon_vals = np.sort(df['Lon'].unique())
            lat_vals = np.sort(df['Lat'].unique())

            # Build grid
            Lon, Lat = np.meshgrid(lon_vals, lat_vals)

            # Pivot to align values correctly
            if step == 0:
                arr = df.pivot_table(index='Lat', columns='Lon', values='Score_step2').values
            else:
                arr = df.pivot_table(index='Lat', columns='Lon', values=f'Score_step{step}').values

            # Basemap settings
            lat_min, lat_max = -2.5, 32.5
            lon_min, lon_max = 97.5, 152.5
            map_ax = Basemap(projection='cyl',
                             llcrnrlat=lat_min, urcrnrlat=lat_max,
                             llcrnrlon=lon_min, urcrnrlon=lon_max,
                             resolution='c')
            map_ax.drawcountries(linewidth=0.8)
            map_ax.drawcoastlines(linewidth=0.8)
            map_ax.drawparallels(np.arange(-60, 61, 10), labels=[1,0,0,0],linewidth=0.5)
            map_ax.drawmeridians(np.arange(-180, 181, 10), labels=[0,0,0,1],linewidth=0.5)

            # Plot data
            cs = map_ax.pcolormesh(Lon, Lat, arr, shading='auto', cmap='coolwarm')

            # Add colorbar
            plt.colorbar(cs, orientation='vertical', shrink=0.8, pad=0.05, label='TCG probability')
            map_ax.fillcontinents(color='lightgray', lake_color=None, alpha=1.0)

            plt.title(f"Forecast lead time (step) {step}", fontsize=10)
            plt.tight_layout()
            filename = "stat" #"".join(str(name).split())
            plt.savefig(f'./output/postprocess/{filename}_step{step}.pdf',
                        bbox_inches='tight', format='pdf')
            plt.close()
            print(f"Saved: {filename}_step{step}_prediction.pdf")

