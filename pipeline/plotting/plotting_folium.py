import os 
import folium 
import geopandas as gpd 
import branca.colormap as cm
import pandas as pd 
from IPython.display import display

import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.plotting.plotting import set_attention_weights_agregated_per_daily_period
from pipeline.utils.calendar_class import get_temporal_mask

def load_subway_gdf(FOLDER_PATH,sptial_unit):
    ref_subway = pd.read_csv(f"{FOLDER_PATH}/ref_subway.csv",index_col =0 ).rename(columns={'MEAN_X':'lon','MEAN_Y':'lat','LIB_STA_SIFO':'Nom'})[['lon','lat','COD_TRG','Nom']]
    ref_subway['geometry'] = gpd.points_from_xy(ref_subway.lon,ref_subway.lat)
    ref_subway = gpd.GeoDataFrame(ref_subway,geometry='geometry',crs='EPSG:4326')[['COD_TRG','Nom','geometry']]
    ref_subway.set_index('COD_TRG',inplace=True)
    ref_subway=ref_subway.reindex(sptial_unit)
    ref_subway.reset_index(inplace=True)
    return ref_subway

def plot_folium_map(FOLDER_PATH, gdf, spatial_unit, station_i=0, VMIN=None, VMAX=None, app_tag_mode = None, bike_mode = None):

    # ---- Load subway gdf: 
    ref_subway = load_subway_gdf(FOLDER_PATH,spatial_unit)
    
    # ---- Init Folium Map ----
    gdf_map = gdf.to_crs(epsg=4326)
    m = folium.Map(location=[45.78,4.85], zoom_start=12, tiles='CartoDB positron')
    # ----

    # ---- Coropleth on Attention Weights ----
    if app_tag_mode is not None:
        added_legend = f"with {' '.join(app_tag_mode.split('_')[:-2])} app ({app_tag_mode.split('_')[-1]}) "
        column = app_tag_mode
    elif bike_mode is not None:
        added_legend = f"using {bike_mode}"
        column = bike_mode
    colormap = cm.LinearColormap(
        colors=['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000'], # Colors of 'OrRd' palette
        vmin=VMIN,
        vmax=VMAX,
        caption=f"Attention weights Distribution of IRIS zones {added_legend} at Station {spatial_unit[station_i]}"
    )

    # A function to determine the color for each feature based on its value
    data_dict = gdf_map.set_index('NOM_IRIS')[column].to_dict()
    def style_function(feature):
        # Get the value from the data dictionary using the 'NOM_IRIS' key
        iris_name = feature['properties']['NOM_IRIS']
        value = data_dict.get(iris_name, None)

        # Use the colormap to get the corresponding color
        if pd.isna(value):
            color = '#808080'
        elif VMIN > value: 
            color = 'white'
        else:
            color = colormap(value)



        return {
            'fillColor': color,
            'color': 'black',  # Line color
            'weight': 1,       # Line weight (thickness)
            'fillOpacity': 0.7,
            'line_opacity': 0.3
        }

    # Add the GeoJson layer to the map
    # We use the style_function to color each polygon
    geojson = folium.GeoJson(
        gdf_map.to_json(),
        name='choropleth',
        style_function=style_function,
        highlight_function=lambda x: {'weight': 3, 'color': 'black'}, # Highlight on hover
        tooltip=folium.GeoJsonTooltip(fields=['NOM_IRIS', column,f"{column}_channel_spatial"])
    ).add_to(m)

    # Add the legend from the colormap to the map
    colormap.add_to(m)
    # ----



    # ---- Add interactive tools 
    # Interactive Keyword args 
    style_function = lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.1, 'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1}

    tooltip_layer = folium.features.GeoJson(
        gdf_map,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['NOM_IRIS', column,f"{column}_channel_spatial"],
            aliases=['IRIS Name: ', 'Value: ', 'Channel-Spatial Unit:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
            sticky=True
        )
    )
    m.add_child(tooltip_layer)
    m.keep_in_front(tooltip_layer)
    # ----

    # ---- Add Subway Stations ----
    for index, station in ref_subway.iterrows():
        lat = station.geometry.y
        lon = station.geometry.x
        
        # Par défaut, une croix grise
        svg_cross = """
        <svg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" style="transform: rotate(45deg);">
        <line x1="9" y1="3" x2="9" y2="15" stroke="grey" stroke-width="2.5" />
        <line x1="3" y1="9" x2="15" y2="9" stroke="grey" stroke-width="2.5" />
        </svg>
        """

        # Si l'index correspond à la station spéciale
        if index == station_i:
            svg_cross ="""
            <svg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" style="transform: rotate(45deg);">
        <line x1="10" y1="0" x2="10" y2="20" stroke="purple" stroke-width="4.5" />
        <line x1="0" y1="10" x2="20" y2="10" stroke="purple" stroke-width="4.5" />
        </svg>
        """

        # Création du marqueur avec l'icône personnalisée
        # Use DivIcon to wrap the SVG
        cross_icon = folium.DivIcon(
            icon_size=(20, 20),
            icon_anchor=(10, 10), # Anchors the icon at its center
            html=svg_cross
        )
        folium.Marker(
            location=[lat, lon],
            tooltip=station['Nom'], # Tooltip qui s'affiche au survol
            icon=cross_icon
        ).add_to(m)
    # ----


    # 5. Add a layer control panel
    folium.LayerControl().add_to(m)

    # To display the map in a Jupyter Notebook or similar environment, simply have 'm' as the last line.
    return m 




def plot_attn_weight_projected_on_map(dict_attn_weights,station,ds,args,folder_path,temporal_aggs,vmax_coeff = 4):
    station_i = list(ds.spatial_unit).index(station)
    s_dates = ds.tensor_limits_keeper.df_verif_test.iloc[:,-1].reset_index(drop=True)

    if hasattr(args,'contextual_kwargs') and len(args.contextual_kwargs) > 0:
        contextual_datasets = list(args.contextual_kwargs.keys())
        for contextual_dataset in contextual_datasets:

            agg_iris_target_n = args.contextual_kwargs[contextual_dataset]['agg_iris_target_n']
            if agg_iris_target_n is not None: 
                gdf = gpd.read_file(f"{folder_path}/lyon_iris_agg{agg_iris_target_n}")
            else:
                gdf = gpd.read_file(f"{folder_path}/lyon_iris_shapefile")

            NetMob_attn_weights = dict_attn_weights[contextual_dataset]

            # -- Set parameters for plotting:
            uniform_weights = 1/NetMob_attn_weights.size(-1)
            VMIN = 0.5*uniform_weights
            VMAX = vmax_coeff*uniform_weights #2*uniform_weights
            CMAP = 'OrRd'
            # --

            # -- Init Correspondence between Attention Weights Size and GDF index :
            kept_zones,list_correspondence,dict_label2agg = None,None,None
            if 'kept_zones' in args.contextual_kwargs[contextual_dataset].keys():
                kept_zones = args.contextual_kwargs[contextual_dataset]['kept_zones']
            elif 'list_correspondence' in args.contextual_kwargs[contextual_dataset].keys():
                list_correspondence = args.contextual_kwargs[contextual_dataset]['list_correspondence']
                dict_label2agg = args.contextual_kwargs[contextual_dataset]['dict_label2agg']
            else:
                raise ValueError(f"Contextual dataset {contextual_dataset} does not have 'kept_zones' or 'list_correspondence' defined in args.contextual_kwargs")
            # ---

            for head in range(NetMob_attn_weights.size(1)):
                for temporal_agg in temporal_aggs:

                    mask = get_temporal_mask(s_dates,temporal_agg = temporal_agg,city=ds.city)   # mask_morning  # mask_evening # mask_off_peak # mask_7 # # mask_8 # mask_9 # mask_16 # mask_17 # mask_18 # mask_21 # mask_22 # mask_23

                    gdf_copy = set_attention_weights_agregated_per_daily_period(gdf,NetMob_attn_weights, 
                                                        station_i,head, mask, agg_iris_target_n,
                                                        dict_label2agg= dict_label2agg,list_correspondence=list_correspondence,
                                                        kept_zones = kept_zones, contextual_dataset = contextual_dataset)

                    print(f"Plotting Attention Weights at HEAD {head} for {contextual_dataset} at {station} ({temporal_agg})") 
                    # ---- Plotting ----
                    if list_correspondence is not None:
                        for app_tag_mode in list_correspondence:
                            folium_map = plot_folium_map(folder_path, gdf_copy, 
                                                spatial_unit = list(ds.spatial_unit), station_i=station_i, VMIN=VMIN, VMAX=VMAX,
                                                app_tag_mode = app_tag_mode)
                    else:
                        folium_map = plot_folium_map(folder_path, gdf_copy, 
                                                spatial_unit = list(ds.spatial_unit), station_i=station_i, VMIN=VMIN, VMAX=VMAX,
                                                bike_mode = contextual_dataset)
                    # ---
                    display(folium_map)