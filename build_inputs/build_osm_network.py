# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
#parent_dir = f"{parent_dir}/prediction_validation/"  # A utiliser sur .ipynb notebook
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

import osmnx as ox
import geopandas as gpd
import os 

data_folder = '../../../../../data/rrochas/raw_data'
osm_shp_shapefile= f"{data_folder}/lyon_osm_network"
print(f'shapefile in {osm_shp_shapefile} :\n{os.listdir(osm_shp_shapefile)}')

# Liste des communes de l'agglomération lyonnaise
communes = [
    "Lyon, France",
    "Villeurbanne, France",
    "Bron, France",
    "Vaulx-en-Velin, France",
    "Caluire-et-Cuire, France",
    "Décines-Charpieu, France",
    "Vénissieux, France"
]

# Load OSM Network with all possible type of links:
G = ox.graph_from_place(communes, network_type='drive')
ox.save_graph_geopackage(G, filepath=f"{data_folder}/lyon_osm_network")
gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G, nodes=True, edges=True)


### Ring Road / Highway: 
highway_types = ["motorway", "motorway_link", "trunk", "trunk_link"]
highway_gdf = gdf_edges[gdf_edges["highway"].isin(highway_types)]
# ...


### Urban road: 
# Convert 'speed' in numerical format:
gdf_edges["maxspeed"] = gdf_edges["maxspeed"].apply(lambda x: float(x) if (x is not None and str(x).isdigit()) else None)

urban_gdf = gdf_edges[
    (gdf_edges["maxspeed"].notna()) &           # Filter NaN speed
    (gdf_edges["maxspeed"] <= 50) &             # Filter speed > 50 km/h
    (~gdf_edges["highway"].isin(highway_types)) # Filter highway/ringroad
]
# ...

## Convert List to Str and save: 
for gdf,name_shp in zip([highway_gdf,urban_gdf],['highway_gdf','urban_gdf']):
    for c in gdf.columns:
        if c != 'geometry':
            gdf[c] = gdf[c].apply(lambda x: ",".join(str(i) for i in x) if isinstance(x, list) else str(x))
    gdf.to_file(f"{osm_shp_shapefile}/{name_shp}.shp")
## ...


if __name__ == '__main__':
    import geopandas as gpd
    data_folder = '../../../../../data/rrochas/raw_data'
    osm_shp_shapefile= f"{data_folder}/lyon_osm_network"

    # Load Highway:
    gpd.read_file(f"{osm_shp_shapefile}/highway_gdf.shp")
    # Load Urban Road Network:
    gpd.read_file(f"{osm_shp_shapefile}/urban_gdf.shp")
    # Load Edges full network:
    gpd.read_file(f"{osm_shp_shapefile}/edges.shp")
    # Load Nodes:
    gpd.read_file(f"{osm_shp_shapefile}/nodes.shp")
