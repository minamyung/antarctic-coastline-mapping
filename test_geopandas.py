import geopandas as gpd
from matplotlib import pyplot as plt
# Clips the reference coastline file to the size of the image border(inner border)
ref = gpd.read_file('C:/Users/myung/Documents/CSC8099/Data/add_coastline_high_res_polygon_v7_4_dissolved.shp')
border = gpd.read_file('C:/Users/myung/Documents/CSC8099/Data/polygons/polygon_layer.shp')
clipped = gpd.clip(ref, border)
clipped.plot()
plt.show()

