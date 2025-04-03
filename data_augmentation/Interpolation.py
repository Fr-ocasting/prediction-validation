import torch
import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

import itertools 

class InterpolatorObject(object):
    """
    A class used for interpolation into data for augmentation.
    """

    def __init__(self, H, D, W):
        self.H = H
        self.D = D
        self.W = W

        self.start_min = self.W + self.D 
        self.end_max   = self.W + self.D + self.H -1 


        self.calendar_in_contextual = False

        self.get_all_possible_window_size_and_start()

    def get_all_possible_window_size_and_start(self):

        start_list = [self.start_min+k for k in range(self.end_max-self.start_min)]
        size_list = [k for k in range(1,(self.end_max-self.start_min))]

        possible_start_window = list(itertools.product(start_list,size_list))
        self.possible_start_window = [c for c in possible_start_window if c[0]+c[1] <= self.end_max]



    def compute_interpolation(self, U_train,  Utarget_train, contextual_train):
            '''
            This method retrieves a noise DataFrame for each spatial unit, reindexes it to include missing dates,
            and constructs a noise feature vector consistent with the way U_train/Utarget_train are built.
            It then applies a user-defined noise distribution (scaled and normalized) and injects the noise into
            U_train/Utarget_train to produce data-augmented tensors.

            Returns:
            --------
            U_train_copy (torch.Tensor): Augmented version of the training feature tensor
            Utarget_train_copy (torch.Tensor): Augmented version of the target feature tensor
            '''
            # Copy
            U_train_copy = U_train.clone()
            Utarget_train_copy = Utarget_train.clone()
            contextual_train_copy = {name: tensor.clone() for name,tensor in contextual_train.items()}

            # Init :
            n = U_train_copy.size(0)
            shuffled_index = torch.randperm(n)
            size_sub_index = n//len(self.possible_start_window)

            # For each possible couple (start-index, window size) : 
            for k in range(len(self.possible_start_window)):
                if k < len(self.possible_start_window)-1:
                    sub_index = shuffled_index[k*size_sub_index:(k+1)*size_sub_index]
                else:
                    sub_index = shuffled_index[k*size_sub_index:] 

                U_train_copy = self.apply_interplation(U_train_copy,sub_index,self.possible_start_window[k])    
                contextual_train_copy = self.apply_interpolation_on_contextual_dict(contextual_train_copy,sub_index,self.possible_start_window[k])
            
            if self.calendar_in_contextual:
                 print('calendar data augmented by dupplication but not modified')
            
            return U_train_copy,Utarget_train_copy,contextual_train_copy

    def apply_interplation(self,U,sub_index,start_window):
            start_idx,window_size = start_window
            end_idx = start_idx+window_size

            start_val = U[sub_index, :, start_idx]
            end_val   = U[sub_index, :, end_idx]
            for t in range(start_idx + 1, end_idx):
                alpha = (t - start_idx) / (end_idx - start_idx)
                U[sub_index, :, t] = (1 - alpha) * start_val + alpha * end_val
            
            return U

    def apply_interpolation_on_contextual_dict(self,contextual_train_copy,sub_index,start_window):
        # Interpolate in contextual data
        for name, tensor in contextual_train_copy.items():
            if ('subway_out' in name) or ('netmob' in name) or  ('subway_in' in name):
                # Use the same global window for the entire batch
                tensor = self.apply_interplation(tensor,sub_index,start_window)
            elif 'calendar' in name:
                self.calendar_in_contextual = True
            else:
                raise NotImplementedError(f'Name {name} has not been implemented for Data Augmentation')
            
            contextual_train_copy[name] = tensor
        return contextual_train_copy