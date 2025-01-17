import torch
import sys 
import os 
import pandas as pd
import numpy as np
current_file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,current_file_path)

from constants.paths import USELESS_DATES,DATA_TO_PREDICT
from DL_class import FeatureVectorBuilder
from utils.seasonal_decomposition import fill_and_decompose_df


class DataAugmenter(object):
    def __init__(self,ds,DA_method,DA_moment_to_focus):
        super(DataAugmenter,self).__init__()
        self.dates_train = ds.tensor_limits_keeper.df_verif_train.iloc[:,-1].reset_index(drop=True)
        self.start = ds.tensor_limits_keeper.df_verif_train.min().min()
        self.end = ds.tensor_limits_keeper.df_verif_train.max().max()
        self.time_step_per_hour = ds.time_step_per_hour

        self.step_ahead = ds.step_ahead
        self.H = ds.H
        self.D = ds.D
        self.W = ds.W
        self.Day_nb_steps = ds.Day_nb_steps
        self.Week_nb_steps = ds.Week_nb_steps
        self.shift_from_first_elmt = ds.shift_from_first_elmt
        self.normalizer = ds.normalizer


        # Data Augmentation Parameters: 
        self.DA_method = DA_method
        self.get_time_slots_to_augment(DA_moment_to_focus)

    def get_time_slots_to_augment(self,DA_moment_to_focus):
        if DA_moment_to_focus is not None:
            series_hour = self.dates_train.dt.hour
            series_weekday = self.dates_train.dt.weekday

            index_to_augment = []
            for hours_weekdays in DA_moment_to_focus:
                hours = hours_weekdays['hours']
                weekdays = hours_weekdays['weekdays']
                
                mask_hour = series_hour.isin(hours)
                mask_weekday = series_weekday.isin(weekdays)

                associated_feature_vect_index = (set(series_weekday[mask_weekday].index)&set(series_hour[mask_hour].index))

                # Union 
                self.index_to_augment = list(set(index_to_augment) | set(associated_feature_vect_index))
        else: 
            self.index_to_augment = list(set(self.dates_train.index))
        print(f'{len(self.index_to_augment)} train samples had been added thank to Data Augmentation')



    def DA_augmentation(self,U_train,Utarget_train,contextual_train,ds = None,period = None,min_count = None,alpha = None, p = None):

        if self.DA_method == 'interpolation':
            U_train_copy,Utarget_train_copy,contextual_train_copy = self.interpolation(U_train,Utarget_train,contextual_train)

        if self.DA_method == 'noise':
            if period is None:
                weekly_period =  (24-len(USELESS_DATES['hour']))*(7-len(USELESS_DATES['weekday']))*ds.time_step_per_hour
                daily_period =  (24-len(USELESS_DATES['hour']))*ds.time_step_per_hour
                period = [weekly_period,daily_period]
            U_train_copy,Utarget_train_copy,contextual_train_copy = self.noise_injection(U_train,Utarget_train,contextual_train,ds,period,min_count,alpha,p)

        U_train_augmented,Utarget_train_augmented,contextual_train_augmented = self.focus_on_specific_index(U_train,U_train_copy,Utarget_train,Utarget_train_copy,contextual_train,contextual_train_copy)    
        return U_train_augmented,Utarget_train_augmented,contextual_train_augmented
                    

    def noise_injection(self,U_train,Utarget_train,contextual_train,ds,period,min_count,alpha=1,p=1):
        ''' Data Augmentation by noise injection

        First apply a seasonal decomposition on the Time-Series.
        If contextual_train is for NetMob POIs, the computation time is way too long. We don't do it.

        '''
        # Clone 
        U_train_copy = U_train.clone()
        Utarget_train_copy = Utarget_train.clone()

        # Select Randomly p% sequence to augment: 
        n, N, L = U_train_copy.shape
        out_dim = Utarget_train_copy.shape[-1]
        mask_inject = torch.rand(n, N) < p

        # Noise Injection for the Dataset To predict, and its historical data: 
        U_train_copy,Utarget_train_copy = self.compute_noise_injection(U_train_copy,Utarget_train_copy,ds,DATA_TO_PREDICT,mask_inject,out_dim,alpha)

        # Noise Injection for the contextual data (subway-out ok, but not for NetMob)
        contextual_train_copy = {}
        subway_out_already_tackled = False
        for name, tensor in contextual_train.items():
            tensor_copy_i = tensor.clone()
            # Same Noise injection for every-single station, and then go break the first loop 
            if ('subway_out' in name) and not(subway_out_already_tackled):
                noisy_tensor_copy_i,_ = self.compute_noise_injection(tensor_copy_i,None,ds,'subway_out',mask_inject,out_dim,alpha)
                for name_i, _ in contextual_train.keys():
                    if ('subway_out' in name_i) :
                        contextual_train_copy[name_i] = noisy_tensor_copy_i
                subway_out_already_tackled = True
            elif ('netmob' in name):
                print(name,'data augmented by dupplication but not modified')
                contextual_train_copy[name] = tensor_copy_i
            elif ('calendar' in name):
                print(name,'data augmented by dupplication but not modified')
                contextual_train_copy[name] = tensor_copy_i
            else:
                raise NotImplementedError(f'Name {name} has not been implemented for Data Augmentation')
            

        return U_train_copy,Utarget_train_copy,contextual_train_copy

    def compute_noise_injection(self,U_train_copy,Utarget_train_copy,ds,dataset_name,mask_inject,out_dim,alpha):
        n, N, L = U_train_copy.shape
        self.mask_seq_3d = mask_inject.unsqueeze(-1).expand(-1, -1, L+out_dim)  # Repeat on the dimension L + out_dim

        # Get DataFrame of Noises :
        df_noises = ds.noises[dataset_name]
        featurevectorbuilder = FeatureVectorBuilder(self.step_ahead,self.H,self.D,self.W,self.Day_nb_steps,self.Week_nb_steps,self.shift_from_first_elmt)

        # Reindex df of time-series  to tensor: 
        df_noises_with_init_dates = df_noises.reindex(pd.date_range(self.start, self.end, freq=f'{60//self.time_step_per_hour}min'))
        df_noises_with_init_dates[df_noises_with_init_dates.index.hour.isin(USELESS_DATES['hour'])] = 0  # Set all the noise associated to useless hour (when subway is closed) to 0 
        tensor_noises = torch.from_numpy(df_noises_with_init_dates.values).float()

        # Built Feature vector to get adapted noise for evey elemt of the sequence:
        featurevectorbuilder.build_feature_vect(tensor_noises)
        featurevectorbuilder.build_target_vect(tensor_noises)
        U_noise = featurevectorbuilder.U
        Utarget_noise = featurevectorbuilder.Utarget

        # Mask feature vector: 
        mask_U =  [e for e in np.arange(U_noise.shape[0]) if e not in ds.forbidden_indice_U]
        amp_values = U_noise[mask_U]
        amp_values_target = Utarget_noise[mask_U]

        # Gaussian Noise
        noise = torch.randn(n, N, L+amp_values_target.size(-1))  # Gaussian Noise

        # Scaled with computed amplitude and an 'alpha' factor: 
        raw_noise = noise * alpha * torch.cat([amp_values,amp_values_target],-1)  # shape [n, N, L+out_dim]
        scaled_noise = self.normalizer.normalize_tensor(raw_noise,feature_vect=True)

        # Noise injection on some masked values 
        U_train_copy += scaled_noise[...,:-out_dim] * self.mask_seq_3d[...,:-out_dim]

        if Utarget_train_copy is not None:
            Utarget_train_copy += scaled_noise[...,-out_dim:] * self.mask_seq_3d[...,-out_dim:]

        return U_train_copy,Utarget_train_copy
    
    '''
    def build_df_noises(self,raw_values,df_verif_train,time_step_per_hour,columns,period,min_count = 10):
        decomposition = fill_and_decompose_df(raw_values,df_verif_train,time_step_per_hour,columns,min_count = min_count, period = period)
        df_noises = pd.DataFrame({col : decomposition[col]['resid'] for col in decomposition.keys()})
        df_noises = df_noises[columns]
        self.df_noises = df_noises
    '''

    def interpolation(self,U_train,Utarget_train,contextual_train):
        # Interpolation t-5 = (t-6 + t-4)/2
        U_train_copy = U_train.clone()
        Utarget_train_copy = Utarget_train.clone()

        U_train_copy[:, :, self.W + self.D + 1] = 0.5 * (U_train[:, :, self.W + self.D] + U_train[:, :, self.W + self.D + 2])

        # Tackle contextual data:
        contextual_train_copy = {}
        for name, tensor in contextual_train.items():
            tensor_copy_i = tensor.clone()
            if ('subway_out' in name) or ('netmob' in name):
                tensor_copy_i[:, :, self.W + self.D  + 1] = 0.5 * (
                    tensor[:, :, self.W + self.D ] + tensor[:, :, self.W + self.D + 2])
            elif ('calendar' in name):
                print(name,'data augmented by dupplication but not modified')
            else:
                raise NotImplementedError(f'Name {name} has not been implemented for Data Augmentation')
            
            contextual_train_copy[name] = tensor_copy_i

        return U_train_copy,Utarget_train_copy,contextual_train_copy

    def focus_on_specific_index(self,U_train,U_train_copy,Utarget_train,Utarget_train_copy,contextual_train,contextual_train_copy):
        # Concat with Data-Augmented Values:
        U_train_augmented = torch.cat([U_train,U_train_copy[self.index_to_augment]],dim=0)
        Utarget_train_augmented = torch.cat([Utarget_train,Utarget_train_copy[self.index_to_augment]],dim=0)

        contextual_train_augmented = {}
        for name in contextual_train.keys():
            contextual_train_augmented[name] = torch.cat([contextual_train[name],contextual_train_copy[name][self.index_to_augment]],dim=0)

        return U_train_augmented,Utarget_train_augmented,contextual_train_augmented
