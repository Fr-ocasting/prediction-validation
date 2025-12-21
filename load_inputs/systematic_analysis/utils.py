


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import bokeh
import folium 
import geopandas as gpd 
import datetime 


# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...


from pipeline.clustering.clustering import filter_by_temporal_agg
from pipeline.plotting.plotting import plot_coverage_matshow,add_calendar_columns




IN_bdc = 'midnightblue'  # Inflow - Business Day Color
OUT_bdc = 'indianred' # Outflow - Business Day Color
IN_nbdc = '#708090'  # slategray (bleu grisé)
OUT_nbdc = '#bc8f8f' # rosybrown (rouge/brun grisé)

# IN_nbdc = 'cadetblue' # Inflow - Non Business Day Color
# OUT_nbdc = 'peru' # Outflow - Non Business Day Color

# --------------------------------------------------
#  Utils  

def get_time_bounds(period):
    if period == 'all_day':
        start = datetime.time(0, 00)
        end=datetime.time(23, 59)
    if period == 'morning_peak':
        start = datetime.time(7, 00)
        end=datetime.time(10, 00)
    if period == 'evening_peak':
        start = datetime.time(16, 30)
        end=datetime.time(19, 00)
    return start, end

def convert_to_str(period):
    if period == 'all_day':
        return 'All Day'
    if period == 'morning_peak':
        return 'Morning Peak (7am-10am)'
    if period == 'evening_peak':
        return 'Evening Peak (4:30pm-7pm)'



def filter_per_day_type(df,period,city,start = None,end=None):
    if (start is None) and (end is None):
        start, end = get_time_bounds(period)

    ts_bd = filter_by_temporal_agg(df = df,
                                    temporal_agg = 'business_day',
                                    city = city,
                                    start = start,
                                    end=end)

    ts_nbd = filter_by_temporal_agg(df = df,
                                    temporal_agg = 'non_business_day',
                                    city = city,
                                    start = start,
                                    end=end)
    
    return ts_bd, ts_nbd



def get_inflow_outflow(df_raw_in_no_agg,df_raw_out_no_agg,filter_q,city,period='all_day',day_type = None,index_name= 'idstation'):
    if filter_q is not None:
        # Apply quantile filter
        df_raw_in_f = df_raw_in_no_agg[df_raw_in_no_agg < df_raw_in_no_agg.quantile(filter_q)]
        df_raw_out_f = df_raw_out_no_agg[df_raw_out_no_agg < df_raw_out_no_agg.quantile(filter_q)]
    else:
        df_raw_in_f = df_raw_in_no_agg
        df_raw_out_f = df_raw_out_no_agg

    df_bd_in, df_nbd_in = filter_per_day_type(df_raw_in_f,period,city)
    df_bd_out, df_nbd_out = filter_per_day_type(df_raw_out_f,period,city)

    if day_type is not None:
        if day_type == 'business_day':
            df_raw_in_f = df_bd_in
            df_raw_out_f = df_bd_out
        elif day_type == 'non_business_day':
            df_raw_in_f = df_nbd_in
            df_raw_out_f = df_nbd_out
        else:
            raise ValueError(f"day_type must be 'business_day' or 'non_business_day', got {day_type}")
    else:
        df_raw_in_f = pd.concat([df_bd_in,df_nbd_in],axis=0)
        df_raw_out_f = pd.concat([df_bd_out,df_nbd_out],axis=0)
            


    # Inflow 
    inflow = df_raw_in_f.sum(axis = 0)
    inflow.name = 'Inflow'
    inflow.index.name = index_name

    # Outflow 
    outflow = df_raw_out_f.sum(axis = 0)
    outflow.name = 'Outflow'
    outflow.index.name = index_name
    return inflow, outflow


def normalize(df_raw,norm,normtype,normalized_based_on= None,city= None):
    df = df_raw.copy()
    if norm:
        if normalized_based_on is not None: 
            spatial_units = df_raw.columns.tolist()
            df['datetime'] = df.index
            df = add_calendar_columns(df, business_day = True,city = city)
            group_stats = list(set(normalized_based_on))

            
            # mean and std 
            df_stats = df.groupby(group_stats)[spatial_units].agg(['mean','std','min','max'])

            # Normalize df based on business days mean and std
            df_means = df_stats.xs('mean', level=1, axis=1)
            df_stds = df_stats.xs('std', level=1, axis=1)
            df_mins = df_stats.xs('min', level=1, axis=1)
            df_maxs = df_stats.xs('max', level=1, axis=1)

            if normtype == 'zscore':
                df_normalized = (df.set_index(['datetime']+group_stats)[spatial_units] - df_means)/ df_stds
            elif normtype == 'minmax':
                df_normalized = (df.set_index(['datetime']+group_stats)[spatial_units] - df_mins) / (df_maxs - df_mins)
            else:
                raise ValueError("normtype must be 'zscore' or 'minmax'")

        else:
            df_means = df.mean()
            df_stds = df.std()
            df_mins = df.min()
            df_maxs = df.max()

            if normtype == 'zscore':
                df_normalized = (df - df_means) /  df_stds
            elif normtype == 'minmax':
                df_normalized = (df - df_mins) / (df_maxs - df_mins)
            else:
                raise ValueError("normtype must be 'zscore' or 'minmax'")
    else:
        df_normalized = df
    return df_normalized

# --------------------------------------------------
#  Distribution des volumes et valeurs manquantes 

def get_histogram_per_day_type(df,city,period,stats,palette,n_bins = 30):
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
    if type(df) == pd.core.frame.DataFrame:
        df = df.sum(axis=1)


    # filter by day type
    ts_bd, ts_nbd = filter_per_day_type(df,period,city)

    
    # histogram plot avec sns : 
    plt.figure(figsize=(10,6))
    # set histogram with same bins width :
    mini,maxi = min(ts_bd.min(),ts_nbd.min()), max(ts_bd.max(),ts_nbd.max())
    bins = np.linspace(mini, maxi, n_bins)
    sns.histplot(ts_bd, color=palette[0], label='Business Days', kde=False, stat=stats, alpha=0.6, bins=bins)
    sns.histplot(ts_nbd, color=palette[1], label='Non-Business Days', kde=False, stat=stats, alpha=0.6, bins=bins)
    plt.xlabel('Volume')
    plt.ylabel('Density')
    plt.title('Histogram of Volumes by day type across all Spatial Units and {} period'.format(convert_to_str(period)))
    plt.legend()
    plt.show()
    return ts_bd, ts_nbd





def preprocess_df(df,ascending=None,period= None,city= None,
                  norm = False,filter_q = None,
                  normtype='zscore',
                  normalized_based_on= None,
                  start = None,
                  end = None):
    """
    Preprocess the dataframe by organizing columns, filtering by day type, applying quantile filtering, and normalization.
    Args:
        ascending (bool): if True, sort columns in ascending order of median values.
        df (pd.DataFrame): dataframe of volumes with timestamps as index and a column per spatial unit.
        period (str): period of the day to analyze. Choices are 'all_day', 'morning_peak', 'evening_peak'.
        city (str): city to analyze.
        norm (bool): if True, normalize volumes before further processing.
        filter_q (float): quantile for filtering extreme volumes before further processing. If None, no filtering.
        normtype (str): type of normalization to apply if norm is True. Choices are 'zscore' and 'minmax'.
    Returns:
        df_bd (pd.DataFrame): processed dataframe for business days.
        df_nbd (pd.DataFrame): processed dataframe for non-business days.
    """
    if ascending is not None: 
        # Organise columns in an ascending order of the median values:
        if ascending:
            median_values = df.median().sort_values()   
        else:
            median_values = df.median().sort_values(ascending=False)   
        df = df[median_values.index]

    # filter by day type
    if period is not None:
        if city is None:
            raise ValueError("city must be provided when period is specified")
        df_bd, df_nbd = filter_per_day_type(df,period,city,start = start,end = end)

        if filter_q is not None:
            df_bd = df_bd[df_bd <= df_bd.quantile(filter_q)]
            df_nbd = df_nbd[df_nbd <= df_nbd.quantile(filter_q)]
            
        # Normalize if needed 
        df_bd = normalize(df_bd,norm,normtype,normalized_based_on,city=city)
        df_nbd = normalize(df_nbd,norm,normtype,normalized_based_on,city=city)
        return df_bd, df_nbd
    
    else:
        if start is not None or end is not None:
            df = filter_by_temporal_agg(df = df,
                                    temporal_agg = None,
                                    city = city,
                                    start = start,
                                    end=end)
        if filter_q is not None:
            df = df[df <= df.quantile(filter_q)]
        df = normalize(df,norm,normtype,normalized_based_on,city=city)
        return df
    
def get_add_title(norm,filter_q):
    add_title1 = 'Normalized ' if norm else ''
    add_title2 = '(Filtered at {} quantile)'.format(filter_q) if filter_q else ''
    return add_title1,add_title2

def get_boxplot_per_spatial_unit_per_day_type(df,period,city,palette,norm = False,filter_q = None,normtype='zscore',
                                              figsize=(12, 6)):
    """
    Retourne les boxplot des volumes par stations en business days et en non business days. Les deux boxplots sont superposés en transparence.
    
    Exemple:
        df (pd.DataFrame): dataframe des volumes avec des timestamp en index et une colonne par unité spatiale.

    Args:
        df (pd.DataFrame): dataframe des volumes avec des timestamp en index et une colonne par unité spatiale.
        period (str): période de la journée à analyser. Choix entre 'all_day', 'morning_peak', 'evening_peak'.
        city (str): ville à analyser.
        palette (list): liste de deux couleurs pour les boxplots [business_day_color, non_business_day_color].
        norm (bool): si True, normalise les volumes avant de tracer les boxplots.
        filter_q (float): quantile pour filtrer les volumes extrêmes avant de tracer les boxplots. Si None, pas de filtrage.
        normtype (str): type de normalisation à appliquer si norm est True. Choix entre 'zscore' et 'minmax'.
    """
    # Organise columns in an ascending order of the median values:
    df_bd,df_nbd = preprocess_df(df,ascending=True,period= period,city= city,norm = norm,filter_q = filter_q,normtype=normtype)

    ts_bd = df_bd.stack()
    ts_bd.name = 'Volume'
    ts_bd = ts_bd.reset_index()
    ts_bd['day_type'] = 'Business Day'

    ts_nbd = df_nbd.stack()
    ts_nbd.name = 'Volume'
    ts_nbd = ts_nbd.reset_index()
    ts_nbd['day_type'] = 'Non-Business Day'
    df_combined = pd.concat([ts_bd, ts_nbd], axis=0)

    # 1. CALCUL DES INDICATEURS SUPPLEMENTAIRES A AFFICHER
    df_q = df_combined.groupby(['Spatial Unit ID', 'day_type'])['Volume'].agg(
                                        # q90=lambda x: x.quantile(0.75),
                                        # q95=lambda x: x.quantile(0.95),
                                        q99=lambda x: x.quantile(0.99),
                                    ).reset_index()
    # Transformation en format long pour l'affichage (une ligne par valeur)
    df_q = df_q.melt(id_vars=['Spatial Unit ID', 'day_type'], value_name='Volume').drop(columns='variable')

    # B. Calcul des Max pour les afficher comme cercles vides
    df_max = df_combined.groupby(['Spatial Unit ID', 'day_type'])['Volume'].max().reset_index()

    # boxplot plot avec sns :
    # 2. AFFICHAGE DU GRAPHIQUE
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)

    # COUCHE 1 : Le Boxplot principal (Moustaches à 5% et 95%)
    sns.boxplot(
        data=df_combined,
        x='Spatial Unit ID',
        y='Volume',
        hue="day_type",
        palette={'Business Day': palette[0], 'Non-Business Day': palette[1]},
        showfliers=False,    # False   # On cache les outliers classiques
        showcaps=True,
        showmeans=True,
        meanline=True,
        meanprops={"color": "green", "linestyle": "-", "linewidth": 2},
        medianprops={"color": "yellow", "linewidth": 2},
        ax=ax
    )

    # COUCHE 2 : Les traits horizontaux pour 10% et 90%
    # On utilise le marqueur "_" (underscore) qui fait une ligne horizontale
    sns.stripplot(
        data=df_q,
        x='Spatial Unit ID',
        y='Volume',
        hue="day_type",
        dodge=True,             # Indispensable pour s'aligner sur les boîtes décalées
        jitter=False,
        marker='_',             # Le secret pour avoir une barre horizontale
        size=10,                # Taille de la barre (largeur)
        linewidth=1,            # Epaisseur du trait
        # color='red',          # Couleur des traits 10/90
        palette={'Business Day': palette[0], 'Non-Business Day': palette[1]},
        legend=False,
        ax=ax,
        zorder=10               # S'assure que c'est dessiné au-dessus
    )
    
    df_max['Spatial Unit ID'] = df_max['Spatial Unit ID'].astype(str)
    sns.scatterplot(data=df_max, x='Spatial Unit ID', y='Volume', hue="day_type", marker="$\circ$", ec="face", s=30,palette = { 'Business Day': palette[0], 'Non-Business Day': palette[1]}, ax=ax, legend=False, zorder=11)


    add_title1,add_title2 = get_add_title(norm,filter_q)

    plt.xticks(rotation=90)
    plt.xlabel('Spatial Units')
    plt.ylabel(f'{add_title1}Volume')

    # Set Ttile: 
    plt.title(f'Boxplot of {add_title1}Volumes {add_title2} by Spatial Units and {convert_to_str(period)} period')
  

    # Affiche la légende des : 
    #  - business day 
    #  - non business day
    #  - mean (ligne verte)
    #  - median (ligne jaune)
    #  - quantile 99% (traits rouges)
    #  - max (cercles gris vides)
    #  - boxplots quantile 5%-95%
    handles = [plt.Line2D([0], [0], color=palette[0], lw=4, label='Business Days'),
                plt.Line2D([0], [0], color=palette[1], lw=4, label='Non-Business Days'),
                plt.Line2D([0], [0], color='green', lw=2, label='Mean'),
                plt.Line2D([0], [0], color='yellow', lw=2, label='Median'),
                plt.Line2D([0], [0], color=palette[0], marker='_', markersize=10, linestyle='None', label='99th Percentile (Business Days)'),  
                plt.Line2D([0], [0], color=palette[1], marker='_', markersize=10, linestyle='None', label='99th Percentile (Non-Business Days)'),  
                plt.Line2D([0], [0], color=palette[0], marker='o', markersize=6, markerfacecolor='None', linestyle='None', label='Max (Business Days)'),
                plt.Line2D([0], [0], color=palette[1], marker='o', markersize=6, markerfacecolor='None', linestyle='None', label='Max (Non-Business Days')
                ]
    
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    # Affiche la legende hors du cadre du plot, à droite: 

    plt.show()
    return df_bd, df_nbd



# --------------------------------------------------
# Identification des particularités:


def heatmap(df,city,norm=False,filter_q = None,normtype= 'zscore',cmap = 'RdYlBu',bool_reversed = False,
            index = 'month_year',columns = 'day_date',
            normalized_based_on = None, vmin = None, vmax = None,figsize = None):
    """ Retourne une heatmap des volumes normalisés par unité spatiale et par day type """

    # Sum through spatial units: 
    if filter_q is not None:
        df = df[df <= df.quantile(filter_q)]

    ts = df.sum(axis=1)
    ts.name = 'Volume'
    ts.index.name = 'datetime'
    ts = pd.DataFrame(ts)

    # sum through day:
    if norm:
        ts = ts.resample('D').sum()
        ts.loc[ts['Volume'] < 1000, 'Volume'] = np.nan

    processed_df = preprocess_df(ts,norm = norm,filter_q = None,normtype=normtype,city=city,normalized_based_on=normalized_based_on)
    processed_df['datetime'] = processed_df.index.levels[0] if isinstance(processed_df.index, pd.MultiIndex) else processed_df.index
    processed_df = processed_df.reset_index(drop=True)

    df_agg = add_calendar_columns(processed_df,freq='D')
    # df_agg['year']= df_agg.apply(lambda row: row['year']+1 if (row['datetime'].month==12 and row['woy']==1) else row['year'], axis=1)
    # df_agg['week_of_year']= df_agg.apply(lambda row: datetime.datetime.strptime(f"{row['year']}-W{row['woy']}-1", "%G-W%V-%u").strftime("%Y-%m-%d"), axis=1)

    df_agg = df_agg.pivot(index = index,columns =columns,values = 'Volume') # .fillna(0)


    if index == 'month_year':
        df_agg.index = [dt.strftime('%B %Y') for dt in df_agg.index]
        yaxis = "Months - Years"
    if index == 'monday_date':
        yaxis = "1st day of week"

    if columns == 'day_date':
        xaxis = "Days of Month"
    if columns == 'day_of_week':
        xaxis = "Days of Week"
        dict_dow = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
        df_agg.columns = [dict_dow[dt] for dt in df_agg.columns]

    # heatmap plot
    ax = plot_coverage_matshow(df_agg,cmap=cmap, save=None, bool_reversed=bool_reversed, 
                        v_min=vmin, v_max=vmax, 
                        #   display_values=True, 
                        cbar_magic_args = True,
                        xaxis = xaxis,
                        yaxis = yaxis,  
                        figsize = figsize
        )
    add_title1,add_title2 = get_add_title(norm,filter_q)

    ax.set_title(f"HeatMap of Daily {add_title1}Volumes {add_title2}")
    return df_agg



        
    
    