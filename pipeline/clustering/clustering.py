
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Optional
import os 
# Visualization and Clustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from pipeline.calendar_class import get_temporal_mask
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import geopandas as gpd
import itertools

def filter_by_temporal_agg(df: pd.DataFrame, temporal_agg: str, city : str,
        start = datetime.time(7, 00),
        end=datetime.time(21, 00),
        start_morning_peak = datetime.time(7,30), 
        end_morning_peak = datetime.time(9,0),
        start_evening_peak = datetime.time(17,0), 
        end_evening_peak = datetime.time(19,0)
        ) -> pd.DataFrame:
    s_dates = df.index.to_series()
    filter_mask = get_temporal_mask(s_dates=s_dates,
                                    start=start,
                                    end=end, 
                                    temporal_agg=temporal_agg,
                                    city = city,
                                     start_morning_peak=start_morning_peak,end_morning_peak=end_morning_peak,
                                  start_evening_peak=start_evening_peak,end_evening_peak=end_evening_peak
                                    )
  
        
    df_filtered = df[filter_mask]
    print('Number of remaining time-slots after filtering', len(df_filtered))
    return df_filtered




class TimeSeriesClusterer:
    """
    A class to perform clustering on time series data.

    This class provides a structured workflow:
    1. Initialize with a DataFrame whose row represents timestamps and columns represent spatial units
    2. Preprocess the data (e.g., filter by a temporal window).
    3. Run a clustering algorithm (Agglomerative or GMM).
    4. Plot the results.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the clusterer with the source DataFrame.

        Args:
            df (pd.DataFrame): The initial DataFrame with time series as columns
                               and a DatetimeIndex.
        """
        self.df_init: pd.DataFrame = df.copy()
        self.df_preprocessed: Optional[pd.DataFrame] = None
        self.df_normalized: Optional[pd.DataFrame] = None
        self.labels: Optional[np.ndarray] = None
        self.clusters: Optional[Dict[int, List[str]]] = None
        self.model = None
        self.method: Optional[str] = None
        self.min_samples: Optional[int] = None
        self.bool_plot: bool = True  # Whether to show plots or not
        self.folder_path: Optional[str] = None
        self.save_name: Optional[str] = None

    def normalize(self, normalisation_type: str = 'minmax' ):
        """
        Normalizes the preprocessed DataFrame using the specified method.

        Args:
            normalisation_type (str): The type of normalization to apply ('minmax', 'zscore', etc.).

        Returns:
            self: The instance itself for method chaining.
        """
        if self.df_preprocessed is None:
            raise RuntimeError("Preprocessing must be run before normalization. Call .preprocess()")

        if normalisation_type == 'minmax':
            self.df_normalized = (self.df_preprocessed - self.df_preprocessed.min()) / (self.df_preprocessed.max() - self.df_preprocessed.min())
        elif normalisation_type == 'zscore':
            self.df_normalized = (self.df_preprocessed - self.df_preprocessed.mean()) / self.df_preprocessed.std()
        else:
            raise ValueError(f"Unknown normalization type: {normalisation_type}")

        return self

    def preprocess(self, temporal_agg: str, 
                        normalisation_type: str = 'minmax',
                        index: str = 'Station',
                        city: str = 'Lyon',
                        start = datetime.time(7, 00),
                        end=datetime.time(21, 00),
                        start_morning_peak= datetime.time(7,30),
                        end_morning_peak = datetime.time(9,0), 
                        start_evening_peak= datetime.time(17,0),
                        end_evening_peak= datetime.time(19,0),
                        ):
        """
        Filters the initial DataFrame based on a temporal window.
        The result is stored in `self.df_preprocessed`.
        Then normalizes the preprocessed DataFrame using the specified method.


        Args:
            temporal_agg (str): The temporal period to filter on (e.g., 'morning_peak').
            city (str): The city for holiday calendars.
            normalisation_type (str): The type of normalization to apply ('minmax' or 'zscore').
            index (str): The column name used for creating representative profiles (default is 'Station').

        Returns:
            self: The instance itself for method chaining.
        """
        self.df_preprocessed = filter_by_temporal_agg(self.df_init, 
                                                      temporal_agg=temporal_agg, 
                                                      city=city,
                                                      start=start,
                                                      end=end,
                                                    start_morning_peak=start_morning_peak,
                                                    end_morning_peak=end_morning_peak,
                                                    start_evening_peak=start_evening_peak,
                                                    end_evening_peak=end_evening_peak)
        self.normalize(normalisation_type)

        # --- Build Temporal Profil 
        # Filrering sequences with NaN values : 
        df_copy = self.df_normalized.copy()
        df_copy.index = df_copy.index.set_names('datetime') 
        df_copy=df_copy.stack()
        df_copy.name = 'values'
        df_copy=df_copy.reset_index(level=1)
        df_copy['date'] = df_copy.index.date
        df_copy['time'] = df_copy.index.time
        max_group = df_copy.groupby([index, df_copy.index.date]).count().max().max()
        df_copy = df_copy.groupby([index, df_copy.index.date]).filter(lambda group: len(group) == max_group)

        # Pivot into daily profile
        self.representative_profiles = df_copy.pivot_table(index = index,columns = 'time', values = 'values', aggfunc = 'mean')
        print(f"Number of representative profiles: {len(self.representative_profiles)}")
        # --- 

        # - Build correlation matrix: 
        self.corr_matrix = self.df_normalized.corr()

        # - Build correlation-based distance matrix: 
        self.dist_matrix = 1 - np.abs(self.corr_matrix)

    
        return self

    def _format_output(self):
        """Private helper to format the final cluster dictionary."""
        if self.labels is not None:
            unique,counts = np.unique(self.labels,return_counts=True)
            dict_label2count  = dict(zip(unique,counts))

            clusters = {}
            for i in range(len(unique)):
                if (self.min_samples is not None) and (dict_label2count[i] >= self.min_samples):
                    clusters[i] = []
                else:
                    clusters[-1] = []


            sorted_indices = np.argsort(self.labels)
            sorted_series_names = self.df_init.columns[sorted_indices]
            sorted_labels = self.labels[sorted_indices]

            for series_name, label in zip(sorted_series_names, sorted_labels):
                if label in clusters.keys():
                     clusters[label].append(series_name)
                else:
                    clusters[-1].append(series_name)

            for k,label in enumerate(self.labels):
                if label not in clusters.keys():
                    self.labels[k] = -1  # Assign to the 'other' cluster if not in clusters

            self.clusters = clusters

    def run_agglomerative(self, n_clusters: int, 
                                linkage_method: str = 'complete',
                                metric: str ='precomputed',
                                distance_threshold: Optional[float] = None,
                                min_samples: Optional[int] = None
                                ):
        """
        Runs agglomerative clustering on the preprocessed data.

        Args:
            n_clusters (int): The number of clusters to form.
            linkage_method (str): The linkage criterion to use.

        Returns:
            self: The instance itself for method chaining.
        """
        if self.df_normalized is None:
            raise RuntimeError("Preprocessing must be run before clustering. Call .preprocess()")

        self.method = 'agglomerative'
        self.min_samples = min_samples
        
        # --- Distance Matrix Calculation ---
        dist_matrix = 1 - np.abs(self.corr_matrix)
        
        # --- Clustering ---
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters, metric=metric, linkage=linkage_method,distance_threshold = distance_threshold
        )

        ### OLD 
        # self.labels = self.model.fit_predict(dist_matrix)
        ### NEW
        raw_labels = self.model.fit_predict(dist_matrix.values)
        station_to_label_map = dict(zip(dist_matrix.index, raw_labels))
        self.labels = np.array([station_to_label_map[station] for station in self.corr_matrix.columns])
        # ----

        self._format_output()
        print(f"Agglomerative clustering completed with {n_clusters} clusters.")

        return self

    def run_gmm(self, n_clusters: int, covariance_type: str = 'full',index: str = 'Station',random_state: int = 42,min_samples: Optional[int] = None):
        """
        Runs Gaussian Mixture Model clustering on the preprocessed data.
        This method works on the average temporal profile of each series.

        Args:
            n_clusters (int): The number of mixture components (clusters).
            covariance_type (str): The type of covariance parameters to use.
            index (str): The index to use for creating representative profiles (default is 'Station').
            random_state (int): Random seed for reproducibility.

        Returns:
            self: The instance itself for method chaining.
        """
        if self.df_normalized is None:
            raise RuntimeError("Preprocessing must be run before clustering. Call .preprocess()")

        self.method = 'gmm'
        self.min_samples = min_samples

        
        # --- Clustering ---
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            random_state=random_state
        )
        self.labels = self.model.fit_predict(self.representative_profiles.values)
        self._format_output()
        print(f"GMM clustering completed with {n_clusters} clusters.")
        return self

    def plot_clusters(self,heatmap: bool = True, daily_profile: bool = False, 
                      dendrogram: bool = False,
                      bool_plot: bool = True,
                      folder_path: Optional[str] = None,
                      save_name: Optional[str] = None,
                      spatial_unit_distance_to_zones: Optional[gpd.GeoDataFrame] = None,
                      vmax = 1.0,
                      vmin = 0.0,
                      cmap = 'viridis',
                      heatmap_title = 'Correlation Matrix Heatmap (Ordered by Cluster)'):
        """
        Visualizes the clustering results based on the method used.
        """
        if self.labels is None or self.method is None:
            raise RuntimeError("Clustering must be run before plotting. Call a .run_...() method.")
        
        print(f"Plotting results for {self.method} clustering...")
        self.bool_plot = bool_plot
        self.folder_path = folder_path
        self.save_name = save_name
        self.vmax = vmax
        self.vmin =vmin
        self.cmap =cmap
        self.spatial_unit_distance_to_zones = spatial_unit_distance_to_zones
        self.heatmap_title = heatmap_title
        if self.method == 'agglomerative':
            self._plot_agglomerative(heatmap,daily_profile,dendrogram)
        elif self.method == 'gmm':
            self._plot_gmm(heatmap,daily_profile)

    def _plot_agglomerative(self,heatmap: bool = True, daily_profile: bool = False, dendrogram_bool: bool = False):
        """Generates plots for agglomerative clustering (dendrogram and heatmap)."""
        
        # --- Dendrogram Visualization ---
        if dendrogram_bool:
            self._plot_dendrogram()

        # --- Heatmap Visualization ---
        if heatmap:
            self._plot_heatmap_corr()

        # --- Daily Profil Visualization ---
        if daily_profile:
            self._plot_daily_profile()

    def _plot_gmm(self,heatmap: bool = True, daily_profile: bool = False):
        """Generates plots for GMM clustering (average profiles)."""

        # --- Heatmap Visualization ---
        if heatmap:
            self._plot_heatmap_corr()

        # --- Daily Profil Visualization ---
        if daily_profile:
            self._plot_daily_profile()

    def _plot_dendrogram(self):
        linked = linkage(self.dist_matrix.values, method=self.model.linkage)
        plt.figure(figsize=(14, 4))
        dendrogram(linked, orientation='top', labels=self.corr_matrix.index, distance_sort='descending')
        plt.title('Dendrogram of Time Series')
        plt.ylabel('Distance (1 - |correlation|)')
        plt.tight_layout()
        if self.bool_plot:
            plt.show()
        if (self.folder_path is not None) and (self.save_name is not None):
            assert os.path.exists(self.folder_path), f"Folder path {self.folder_path} does not exist."
            if not os.path.exists(f"{self.folder_path}/dendrogram"):
                os.makedirs(f"{self.folder_path}/dendrogram")
            plt.savefig(f"{self.folder_path}/dendrogram/{self.save_name}_dendrogram.pdf", bbox_inches='tight')

    def _personnalise_ticks(self,tick_label,cmap_red_to_gray,norm):
        station_name = tick_label.get_text()
        station_info = self.spatial_unit_distance_to_zones.loc[station_name]
        distance = station_info['min_dist_to_polygons_m']
        is_hub = station_info['hub']
        
        # Appliquer la couleur en fonction de la distance
        color = cmap_red_to_gray(norm(distance))
        tick_label.set_color(color)
        
        # Mettre en gras si c'est un hub
        if is_hub:
            tick_label.set_bbox(dict(boxstyle='square,pad=0.2',
                            facecolor='none',
                            edgecolor='black',
                            linestyle='--'))

    def _change_ticks_color(self,g):
        cmap_red_to_gray = mcolors.LinearSegmentedColormap.from_list('RedToGray', ['red', 'darkgray'])
        norm = mcolors.Normalize(vmin=0, vmax=500)
        g.ax_heatmap.tick_params(axis='both', which='both', length=0)   # Remove tick marks
        ax = g.ax_heatmap

        # Personnalise Yticks : 
        for tick_label in ax.get_yticklabels():
            self._personnalise_ticks(tick_label,cmap_red_to_gray,norm)
        # Personnalise Xticks : 
        for tick_label in ax.get_xticklabels():
            self._personnalise_ticks(tick_label,cmap_red_to_gray,norm)
        
        
        # --- Ajout d'une barre de couleur pour les étiquettes ---
        cbar_ax = g.fig.add_axes([0.8, 1, 0.2, 0.02]) # Position [left, below, width, height]
        cb = ColorbarBase(cbar_ax, cmap=cmap_red_to_gray, norm=norm, orientation='horizontal')
        cb.set_label('Distance to Activity Zone (m)')
        
        # Améliorer la lisibilité
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])   # suptitle area 
    def _add_cluster_separators(self, g):
        ax = g.ax_heatmap 
        
        # Get cluster boundaries and labels from self.clusters
        cluster_sizes = [len(v) for v in self.clusters.values()]
        boundaries = np.cumsum(cluster_sizes)  
        label_positions = boundaries - np.array(cluster_sizes) / 2  
        cluster_labels = [f"Cluster {k}" for k in self.clusters.keys()]  
        
        # Draw separator lines on the heatmap
        for pos in boundaries[:-1]:  
            ax.axhline(y=pos, color='white', linewidth=2.5, linestyle='--')  
            ax.axvline(x=pos, color='white', linewidth=2.5, linestyle='--')  
            
        # Add cluster labels on a new y-axis (right side)
        ax_labels_y = ax.twinx()  
        ax_labels_y.set_ylim(ax.get_ylim())  
        ax_labels_y.set_yticks(label_positions)  
        ax_labels_y.set_yticklabels(cluster_labels, fontsize=12, va='center')  
        ax_labels_y.tick_params(axis='y', length=0, pad=5)  
        
        # Add cluster labels on a new x-axis (top side)
        ax_labels_x = ax.twiny()  
        ax_labels_x.set_xlim(ax.get_xlim())  
        ax_labels_x.set_xticks(label_positions)  
        ax_labels_x.set_xticklabels(cluster_labels, rotation=90, fontsize=12, ha='center')  
        ax_labels_x.tick_params(axis='x', length=0, pad=5)  

    def _plot_heatmap_corr(self):
        clustered_series_order = [series_names for  _,series_names in self.clusters.items()]
        clustered_series_order = list(itertools.chain.from_iterable(clustered_series_order))

        ordered_corr_matrix = self.corr_matrix.loc[clustered_series_order, clustered_series_order]
        g = sns.clustermap(ordered_corr_matrix,
                           row_cluster=False, # False, #True, 
                           col_cluster=False, # False, #True, 
                           cmap=self.cmap, 
                            vmin=self.vmin,
                            vmax=self.vmax,
                           figsize=(10, 10))
        plt.suptitle(self.heatmap_title, y=1.02, fontsize=16)
        
        self._add_cluster_separators(g)
     

        if self.spatial_unit_distance_to_zones is not None:
            self._change_ticks_color(g)

        if self.bool_plot:
            plt.show()

        if (self.folder_path is not None) and (self.save_name is not None):
            assert os.path.exists(self.folder_path), f"Folder path {self.folder_path} does not exist."
            if not os.path.exists(f"{self.folder_path}/heatmap"):
                os.makedirs(f"{self.folder_path}/heatmap")
            plt.savefig(f"{self.folder_path}/heatmap/{self.save_name}_heatmap_corr.pdf", bbox_inches='tight')

    def _plot_daily_profile(self):
        n_clusters = len(self.clusters)
        # --- Plot 1: Individual Profiles per Cluster (Original Plot) ---
        fig, axes = plt.subplots((n_clusters+1), 1, figsize=(14, 4 * (n_clusters+1))) #, sharex=True, squeeze=False)
        axes = axes.flatten()

        # Dictionary to store the mean profile of each cluster for the next plot
        self.cluster_mean_profiles = {}

        for i, (label, series_names) in enumerate(self.clusters.items()):
            ax = axes[i]
            # Plot profiles for stations in the current cluster
            profiles_to_plot = self.representative_profiles.T[series_names]
            ax.plot(profiles_to_plot.index.astype(str), profiles_to_plot, alpha=0.6,label=series_names)
            ax.set_title(f'Cluster {label} ({self.method}-based Profiles)')
            ax.set_ylabel('Average Value')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.3)

            # Calculate and store the mean profile for this cluster
            self.cluster_mean_profiles[label] = profiles_to_plot.mean(axis=1)

        # --- Agregated plot: overall Mean Profile for Each Cluster ---
        ax = axes[-1]
        for label, mean_profile in sorted(self.cluster_mean_profiles.items()):
            ax.plot(mean_profile.index.astype(str), mean_profile, label=f'Cluster {label} Mean', linewidth=2.5, alpha = 1 if label != -1 else 0.3)
        
        ax.set_title('Comparison of Mean Profiles for Each Cluster')
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Average Value')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)


        plt.tight_layout()
        if self.bool_plot:
            plt.show()

        if (self.folder_path is not None) and (self.save_name is not None):
            assert os.path.exists(self.folder_path), f"Folder path {self.folder_path} does not exist."
            if not os.path.exists(f"{self.folder_path}/daily_profiles"):
                os.makedirs(f"{self.folder_path}/daily_profiles")
            plt.savefig(f"{self.folder_path}/daily_profiles/{self.save_name}_daily_profiles.pdf", bbox_inches='tight')







if __name__ == "__main__":

    train_df = pd.read_csv('train_df_1year_subway_out.csv',index_col=0)
    train_df.index = pd.to_datetime(train_df.index)
    train_df.columns.name = 'Station'

    # -- Clustering based business days and Agglomerative clustering:
    clusterer = TimeSeriesClusterer(train_df)
    clusterer.preprocess(temporal_agg='business_day', normalisation_type ='minmax',index= 'Station',city='Lyon') # 'morning','evening','morning_peak','evening_peak','off_peak','bank_holiday','business_day'

    # -- Run agglomerative clustering with 4 clusters:
    clusterer.run_agglomerative(n_clusters=3, linkage_method='complete', metric='precomputed',min_samples=2)
    clusterer.plot_clusters(heatmap= True, daily_profile=True, dendrogram=True)
    # --

    #  -- Run agglomerative clustering with distance threshold:
    clusterer.run_agglomerative(n_clusters=None, linkage_method='complete', metric='precomputed',min_samples=2, distance_threshold=0.1)
    clusterer.plot_clusters(heatmap= True, daily_profile=True, dendrogram=True)

    # -- Run GMM clustering with 3 clusters:
    clusterer.run_gmm(n_clusters=3, covariance_type='full')
    clusterer.plot_clusters(heatmap= True, daily_profile=True)
    # --

    # -- Access to cluster detail: 
    for k,v in clusterer.clusters.items():
        print(k,': ',v)
    # --
