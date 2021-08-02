import geopandas as gpd
from matplotlib import pyplot as plt
import shapely.geometry
# Clips the reference coastline file to the size of the image border(inner border)
# ref = gpd.read_file('C:/Users/myung/Documents/CSC8099/Data/add_coastline_high_res_polygon_v7_4_dissolved.shp')
# border = gpd.read_file('C:/Users/myung/Documents/CSC8099/Data/polygons/polygon_layer.shp')
# clipped = gpd.clip(ref, border)
# clipped.plot()
# plt.show()

# Open polygonized shp
poly_gs = gpd.GeoSeries.from_file('C:/Users/myung/Documents/CSC8099/Data/polygons/polygon_layer_0.05area_nomask_open10_1.shp')
valid_list = (poly_gs.is_valid) # List of boolean values indicating if each geometry in the geoseries is valid
for i in range(0,len(valid_list)):
    if valid_list[i] == False:
        # If invalid, save to a shapefile
        inv_poly_list = list(poly_gs[i])
        
        for j in range(0, len(inv_poly_list)):
            inv_poly = gpd.GeoSeries(inv_poly_list[j], poly_gs.crs)
            # inv_poly.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/invalid/{name1}{name2}.shp'.format(name1=i, name2=j))


# new_poly_list = shapely.geometry.MultiPolygon(((-1573818.3483661159 -555583.87230170844,-1573818.3483661159 -555603.87230170844,-1573838.3483661159 -555603.87230170844,-1573838.3483661159 -555623.87230170844,-1573878.3483661159 -555623.87230170844,-1573878.3483661159 -555643.87230170844,-1573898.3483661159 -555643.87230170844,-1573898.3483661159 -555663.87230170844,-1573918.3483661159 -555663.87230170844,-1573918.3483661159 -555683.87230170844,-1573938.3483661159 -555683.87230170844,-1573938.3483661159 -555723.87230170844,-1573958.3483661159 -555723.87230170844,-1573958.3483661159 -555763.87230170844,-1573978.3483661159 -555763.87230170844,-1573978.3483661159 -555783.87230170844,-1573998.3483661159 -555783.87230170844,-1573998.3483661159 -555803.87230170844,-1574018.3483661159 -555803.87230170844,-1574018.3483661159 -555823.87230170844,-1574038.3483661159 -555823.87230170844,-1574038.3483661159 -555843.87230170844,-1574058.3483661159 -555843.87230170844,-1574058.3483661159 -556043.87230170844,-1574038.3483661159 -556043.87230170844,-1574038.3483661159 -556063.87230170844,-1573978.3483661159 -556063.87230170844,-1573978.3483661159 -556083.87230170844,-1573918.3483661159 -556083.87230170844,-1573918.3483661159 -556063.87230170844,-1573758.3483661159 -556063.87230170844,-1573758.3483661159 -556043.87230170844,-1573698.3483661159 -556043.87230170844,-1573698.3483661159 -556023.87230170844,-1573638.3483661159 -556023.87230170844,-1573638.3483661159 -556003.87230170844,-1573618.3483661159 -556003.87230170844,-1573618.3483661159 -555983.87230170844,-1573598.3483661159 -555983.87230170844,-1573598.3483661159 -555963.87230170844,-1573578.3483661159 -555963.87230170844,-1573578.3483661159 -555943.87230170844,-1573558.3483661159 -555943.87230170844,-1573558.3483661159 -555923.87230170844,-1573538.3483661159 -555923.87230170844,-1573538.3483661159 -555783.87230170844,-1573558.3483661159 -555783.87230170844,-1573558.3483661159 -555743.87230170844,-1573578.3483661159 -555743.87230170844,-1573578.3483661159 -555723.87230170844,-1573598.3483661159 -555723.87230170844,-1573598.3483661159 -555623.87230170844,-1573618.3483661159 -555623.87230170844,-1573618.3483661159 -555603.87230170844,-1573638.3483661159 -555603.87230170844,-1573638.3483661159 -555583.87230170844,-1573818.3483661159 -555583.87230170844)),((-1574118.3483661159 -556243.87230170844,-1574118.3483661159 -556263.87230170844,-1574098.3483661159 -556263.87230170844,-1574098.3483661159 -556243.87230170844,-1574038.3483661159 -556243.87230170844,-1574038.3483661159 -556223.87230170844,-1574018.3483661159 -556223.87230170844,-1574018.3483661159 -556203.87230170844,-1573998.3483661159 -556203.87230170844,-1573998.3483661159 -556103.87230170844,-1574018.3483661159 -556103.87230170844,-1574018.3483661159 -556083.87230170844,-1574038.3483661159 -556083.87230170844,-1574038.3483661159 -556063.87230170844,-1574178.3483661159 -556063.87230170844,-1574178.3483661159 -556083.87230170844,-1574198.3483661159 -556083.87230170844,-1574198.3483661159 -556223.87230170844,-1574178.3483661159 -556223.87230170844,-1574178.3483661159 -556243.87230170844,-1574118.3483661159 -556243.87230170844))))
new_poly_gs = gpd.GeoSeries.from_wkt('MULTIPOLYGON (((-1573818.3483661159 -555583.87230170844,-1573818.3483661159 -555603.87230170844,-1573838.3483661159 -555603.87230170844,-1573838.3483661159 -555623.87230170844,-1573878.3483661159 -555623.87230170844,-1573878.3483661159 -555643.87230170844,-1573898.3483661159 -555643.87230170844,-1573898.3483661159 -555663.87230170844,-1573918.3483661159 -555663.87230170844,-1573918.3483661159 -555683.87230170844,-1573938.3483661159 -555683.87230170844,-1573938.3483661159 -555723.87230170844,-1573958.3483661159 -555723.87230170844,-1573958.3483661159 -555763.87230170844,-1573978.3483661159 -555763.87230170844,-1573978.3483661159 -555783.87230170844,-1573998.3483661159 -555783.87230170844,-1573998.3483661159 -555803.87230170844,-1574018.3483661159 -555803.87230170844,-1574018.3483661159 -555823.87230170844,-1574038.3483661159 -555823.87230170844,-1574038.3483661159 -555843.87230170844,-1574058.3483661159 -555843.87230170844,-1574058.3483661159 -556043.87230170844,-1574038.3483661159 -556043.87230170844,-1574038.3483661159 -556063.87230170844,-1573978.3483661159 -556063.87230170844,-1573978.3483661159 -556083.87230170844,-1573918.3483661159 -556083.87230170844,-1573918.3483661159 -556063.87230170844,-1573758.3483661159 -556063.87230170844,-1573758.3483661159 -556043.87230170844,-1573698.3483661159 -556043.87230170844,-1573698.3483661159 -556023.87230170844,-1573638.3483661159 -556023.87230170844,-1573638.3483661159 -556003.87230170844,-1573618.3483661159 -556003.87230170844,-1573618.3483661159 -555983.87230170844,-1573598.3483661159 -555983.87230170844,-1573598.3483661159 -555963.87230170844,-1573578.3483661159 -555963.87230170844,-1573578.3483661159 -555943.87230170844,-1573558.3483661159 -555943.87230170844,-1573558.3483661159 -555923.87230170844,-1573538.3483661159 -555923.87230170844,-1573538.3483661159 -555783.87230170844,-1573558.3483661159 -555783.87230170844,-1573558.3483661159 -555743.87230170844,-1573578.3483661159 -555743.87230170844,-1573578.3483661159 -555723.87230170844,-1573598.3483661159 -555723.87230170844,-1573598.3483661159 -555623.87230170844,-1573618.3483661159 -555623.87230170844,-1573618.3483661159 -555603.87230170844,-1573638.3483661159 -555603.87230170844,-1573638.3483661159 -555583.87230170844,-1573818.3483661159 -555583.87230170844)),((-1574118.3483661159 -556243.87230170844,-1574118.3483661159 -556263.87230170844,-1574098.3483661159 -556263.87230170844,-1574098.3483661159 -556243.87230170844,-1574038.3483661159 -556243.87230170844,-1574038.3483661159 -556223.87230170844,-1574018.3483661159 -556223.87230170844,-1574018.3483661159 -556203.87230170844,-1573998.3483661159 -556203.87230170844,-1573998.3483661159 -556103.87230170844,-1574018.3483661159 -556103.87230170844,-1574018.3483661159 -556083.87230170844,-1574038.3483661159 -556083.87230170844,-1574038.3483661159 -556063.87230170844,-1574178.3483661159 -556063.87230170844,-1574178.3483661159 -556083.87230170844,-1574198.3483661159 -556083.87230170844,-1574198.3483661159 -556223.87230170844,-1574178.3483661159 -556223.87230170844,-1574178.3483661159 -556243.87230170844,-1574118.3483661159 -556243.87230170844)))')
new_poly_gs.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/invalid/fixed.shp')