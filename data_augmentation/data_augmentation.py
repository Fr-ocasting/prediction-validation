import torch
import sys 
import os 
import pandas as pd
import numpy as np
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from constants.paths import USELESS_DATES,DATA_TO_PREDICT
from DL_class import FeatureVectorBuilder
from data_augmentation.Jittering import JitteringObject

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
        self.normalizers = ds.normalizers 


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



    def DA_augmentation(self,U_train,Utarget_train,contextual_train,ds = None,alpha = None, p = None):

        if self.DA_method == 'interpolation':
            U_train_copy,Utarget_train_copy,contextual_train_copy = self.interpolation(U_train,Utarget_train,contextual_train)

        if self.DA_method == 'noise':
            U_train_copy,Utarget_train_copy,contextual_train_copy = self.noise_injection(U_train,Utarget_train,contextual_train,ds,alpha,p)

        U_train_augmented,Utarget_train_augmented,contextual_train_augmented = self.focus_on_specific_index(U_train,U_train_copy,Utarget_train,Utarget_train_copy,contextual_train,contextual_train_copy)    
        return U_train_augmented,Utarget_train_augmented,contextual_train_augmented
                    

    def noise_injection(self,U_train,Utarget_train,contextual_train,ds,alpha,p):
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
        jiterringobject = JitteringObject(self.normalizers, self.step_ahead, self.H, self.D, self.W, self.Day_nb_steps, self.Week_nb_steps, self.shift_from_first_elmt, self.time_step_per_hour)
        U_train_copy, Utarget_train_copy = jiterringobject.compute_noise_injection(U_train_copy, Utarget_train_copy, ds, DATA_TO_PREDICT, mask_inject, out_dim, alpha)
        #U_train_copy,Utarget_train_copy = self.compute_noise_injection(U_train_copy,Utarget_train_copy,ds,DATA_TO_PREDICT,mask_inject,out_dim,alpha)

        # Noise Injection for the contextual data (subway-out ok, but not for NetMob)
        contextual_train_copy = {}
        subway_out_already_tackled = False
        for name, tensor in contextual_train.items():
            tensor_copy_i = tensor.clone()
            # Same Noise injection for every-single station, and then go break the first loop 
            if ('subway_out' in name):
                if not(subway_out_already_tackled):
                    #noisy_tensor_copy_i,_ = self.compute_noise_injection(tensor_copy_i,None,ds,'subway_out',mask_inject,out_dim,alpha)
                    noisy_tensor_copy_i,_ = jiterringobject.compute_noise_injection(tensor_copy_i,None,ds,'subway_out',mask_inject,out_dim,alpha)

                    for name_i in contextual_train.keys():
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