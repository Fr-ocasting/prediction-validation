


IN_bdc = 'midnightblue'  # Inflow - Business Day Color
OUT_bdc = 'indianred' # Outflow - Business Day Color
IN_nbdc = 'cadetblue' # Inflow - Non Business Day Color
OUT_nbdc = 'peru' # Outflow - Non Business Day Color



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import bokeh
import folium 
import geopandas as gpd 

# --------------------------------------------------
#  Distribution des volumes et valeurs manquantes 

def get_histogram_per_day_type(ts):
    """ Retourne l'histogramme des volumes 
    en business day et en non business day.
    Les deux histogrammes sont superposée en transparence.

    Args:
        ts (pd.Series): time-serie de volumes d'un seul signal. 
        Exemple: 
            ts est la somme des volumes sur toutes les unités spatiales 
          
    Convert: 
        ts_bd (pd.Series): vecteur de volumes en business day.
        ts_nbd (pd.Series): vecteur de volumes en non business day.

    Returns:    

    """

    ts_bd, ts_nbd = 


def get_boxplot_per_spatial_unit_per_day_type(df):
    """
    Retourne les boxplot des volumes par stations en business days et en non business days. Les deux boxplots sont superposés en transparence.
    
    Exemple:
        df (pd.DataFrame): dataframe des volumes avec des timestamp en index et une colonne par unité spatiale.
    """

    df_bd, df_nbd = 



# --------------------------------------------------
# Identification des particularités:

def heatmap_normalized(df):
    """ Retourne une heatmap des volumes normalisés par unité spatiale et par day type """

    df_bd, df_nbd =

    # get stats 
    for df in [df_bd, df_nbd]:
        df_mean = df.mean()
        df_std = df.std()
        df_normalized = (df - df_mean) / df_std

        # plot heatmap 


        
    
    