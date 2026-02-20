import pandas as pd
import numpy as np
import torch

class EventGainDatasetBuilder:
    """
    Transforme les tenseurs de prédiction [B, N, H, R] en un dataset tabulaire 
    au niveau de l'événement (pas de temps, station).
    """
    
    def __init__(self, full_predict1, full_predict2, Y_true, ds, dates, spatial_features_df, weather_df, calendar_df=None):
        self.full_predict1 = full_predict1
        self.full_predict2 = full_predict2
        self.Y_true = Y_true
        self.ds = ds
        self.dates = pd.to_datetime(dates)
        self.spatial_features_df = spatial_features_df
        self.weather_df = weather_df
        self.calendar_df = calendar_df
        self.events_df = None
        
        # Extraction de la liste des stations
        self.stations = list(self.ds.spatial_unit)

    def _extract_gains_and_demand(self, metric='mae', target_horizon_idx=0, min_flow=20, min_err = 1e-5):

        y = self.Y_true.float().unsqueeze(-1).repeat(1, 1, 1, self.full_predict1.shape[-1]) # [B, N, H, R]

        if metric == 'mae':
            err1 = torch.abs(y - self.full_predict1)
            err2 = torch.abs(y - self.full_predict2)
        elif metric == 'mse':
            err1 = (y - self.full_predict1)**2
            err2 = (y - self.full_predict2)**2
        else:
            raise NotImplementedError("Seules les métriques 'mae' et 'mse' sont implémentées.")
        

        err1 = err1.mean(dim=-1)[..., target_horizon_idx] # [B, N]
        err2 = err2.mean(dim=-1)[..., target_horizon_idx] # [B, N]
        y = y.mean(dim=-1)[..., target_horizon_idx] # [B, N]

        # Aplatissement des dimensions [B, N] en vecteurs 1D
        B, N = y.size(0),y.size(1)
        dates_repeated = self.dates.to_numpy()[:,np.newaxis].repeat(N,axis=1)
        repeated_stations = np.array(self.stations)[:,np.newaxis].repeat(B,axis=1).transpose()

        df = pd.DataFrame({
            'datetime': dates_repeated.flatten(),
            'station_id': repeated_stations.flatten(),
            'demand': y.flatten().numpy() if torch.is_tensor(y) else y.flatten(),
            'error_mod1': err1.flatten().numpy() if torch.is_tensor(err1) else err1.flatten(),
            'error_mod2': err2.flatten().numpy() if torch.is_tensor(err2) else err2.flatten(),
        })
        
        # Calcul du gain (un gain positif indique que le mod2 est meilleur)
        # Utilisation d'un seuil minimal pour éviter la division par zéro
        cloned_err1 = np.where(df['error_mod1'] < min_err, min_err, df['error_mod1'])
        df['gain'] = 100 * (df['error_mod1'] - df['error_mod2']) / cloned_err1
        
        # Filtrage sur la demande réelle
        df = df[df['demand'] > min_flow].copy()
        
        return df

    def _build_temporal_features(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['weekday'] = df['datetime'].dt.weekday
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        
        # Jointure avec les données météorologiques
        if self.weather_df is not None:
            df = pd.merge(df, self.weather_df.reset_index().copy(), on='datetime', how='left')

        # Jointure avec le calendrier (vacances scolaires, jours fériés)
        if self.calendar_df is not None:
            df = pd.merge(df, self.calendar_df.reset_index().copy(), on='datetime', how='left')
        return df

    def _build_spatial_features(self, df):
        if self.spatial_features_df is not None:
            spatial_reset = self.spatial_features_df.copy()
            if 'station_id' not in spatial_reset.columns:
                spatial_reset.index.name = 'station_id'
                spatial_reset = spatial_reset.reset_index()

            df = pd.merge(df, spatial_reset, on='station_id', how='left')
            
        return df

    def get_dataset(self, metric='mae', target_horizon_idx=0, min_flow=20,min_err = 1e-5):
        df = self._extract_gains_and_demand(metric=metric, target_horizon_idx=target_horizon_idx, min_flow=min_flow, min_err = min_err)
        df = self._build_temporal_features(df)
        df = self._build_spatial_features(df)
        
        # Nettoyage final
        self.events_df = df.dropna().reset_index(drop=True)
        return self.events_df