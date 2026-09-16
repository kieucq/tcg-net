import numpy as np

omega = 7.29 * 1e-5

def vorticity(u, v, lat, lon):
    x = lon# * 111 * 1000
    y = lat# * 111 * 1000
    
    y[y == 0] = 0.5
    
    lat = np.deg2rad(lat)
    
    # vx = np.divide(v, x, out=np.full_like(v, np.nan, dtype=np.float64), where=(x != 0))
    # uy = np.divide(u, y, out=np.full_like(u, np.nan, dtype=np.float64), where=(y != 0))
    vx = v / x
    uy = u / y
    
    return vx - uy + 2 * omega * np.sin(lat) * 111 * 1000

def divergence(u, v, lat, lon):
    x = lon# * 111 * 1000
    y = lat# * 111 * 1000
    
    y[y == 0] = 0.5
    
    # ux = np.divide(u, x, out=np.full_like(u, np.nan, dtype=np.float64), where=(x != 0))
    # vy = np.divide(v, y, out=np.full_like(v, np.nan, dtype=np.float64), where=(y != 0))
    ux = u / x
    vy = v / y
    
    return ux + vy

def meshgrid(lat, lon, lvl = 1):
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lat_grid = np.repeat(lat_grid[np.newaxis, :, :], lvl, axis=0)
    lon_grid = np.repeat(lon_grid[np.newaxis, :, :], lvl, axis=0)
    
    return lat_grid, lon_grid

if __name__ == '__main__':
    
    print(vorticity(
        np.array([30, 30, -30, -30]),
        np.array([30, -30, 30, -30]),
        np.array([90, 90, 90, 90]),
        np.array([90, 90, 90, 90])
    ))