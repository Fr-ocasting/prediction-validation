import torch
import numpy as np 
import pandas as pd 
import os 
from datetime import timedelta


# Personnal Import 
from DL_class import FeatureVectorBuilder,DatesVerifFeatureVect,TensorLimitsKeeper
from utils import train_valid_test_split
from loader import DictDataLoader
from utils.save_results import save_object,read_object
# ...

class TrainValidTest_Split_Normalize(object):
    def __init__(self,data,
                 train_indices = None , valid_indices = None, test_indices = None,
                 first_train = None, last_train = None, first_valid = None, last_valid = None, first_test = None, last_test = None, 
                 minmaxnorm = False,standardize = False):
        super(TrainValidTest_Split_Normalize,self).__init__()
        self.data = data
        self.minmaxnorm = minmaxnorm
        self.standardize = standardize

        if train_indices is not None : self.train_indices = train_indices
        if valid_indices is not None : self.valid_indices = valid_indices
        if test_indices is not None : self.test_indices = test_indices

        if first_train is not None : self.first_train = first_train
        if last_train is not None : self.last_train = last_train

        if first_valid is not None : self.first_valid = first_valid
        if last_valid is not None : self.last_valid = last_valid

        if first_test is not None : self.first_test = first_test
        if last_test is not None : self.last_test = last_test

        self.split_data()

    def split_data(self):
        # Split Data within 3 groups:
        if hasattr(self,'train_indices'):
            self.data_train = self.data[self.train_indices] 
            self.data_valid = self.data[self.valid_indices] if self.valid_indices is not None else None
            self.data_test = self.data[self.test_indices] if self.test_indices is not None else None
        elif hasattr(self,'first_train'):
            self.data_train = self.data[self.first_train:self.last_train]
            self.data_valid = self.data[self.first_valid:self.last_valid] if hasattr(self,'first_valid') else None
            self.data_test = self.data[self.first_test:self.last_test]   if hasattr(self,'first_test') else None
        else: 
            raise ValueError("Neither 'train_indices' nor 'first_train' attribute has been designed ")
        

    def split_normalize_tensor_datasets(self,normalizer = None):
        '''Load TensorDataset (train_dataset) object from data_train.
        Define TensorDataset object from valid (valid_dataset) and test (test_dataset). 
        Associate statistics from train dataset to valid and test dataset
        Normalize them according to their statistics 
        '''
        train_dataset = TensorDataset(self.data_train, normalized = False, normalizer=normalizer)


        if hasattr(self,'first_valid'): 
            valid_dataset = TensorDataset(self.data_valid, normalized = False, normalizer=normalizer)
        else:
            valid_dataset = None

        if hasattr(self,'first_test'):  
            test_dataset = TensorDataset(self.data_test,normalized = False, normalizer=normalizer)
        else : 
            test_dataset = None

        if normalizer is not None:
            train_dataset.normalize(feature_vect = True)
            if hasattr(self,'first_valid'): 
                valid_dataset.normalize(feature_vect = True)

            if hasattr(self,'first_test'): 
                test_dataset.normalize(feature_vect = True)

        
        return(train_dataset,valid_dataset,test_dataset)



class Normalizer(object):
    def __init__(self,reference = None,minmaxnorm = False, standardize = False, dims = None):
        self.minmaxnorm = minmaxnorm
        self.standardize = standardize
        self.dims = dims
        reshaped_inputs = self.reshape_input(reference,dims)
        self.get_stats(reshaped_inputs)  # Get Min, Max, Mean, Std 

    def reshape_input(self,inputs,dims):
        # Design Permutation tuple: 
        int_dims = [dim if dim>=0 else inputs.dim()+dim for dim in dims ]   
        int_dims = sorted(int_dims)
        remaining_dims = [dim for dim in np.arange(inputs.dim()) if not(dim in int_dims)] 
        permutations = remaining_dims+int_dims
        self.permutations = permutations
        
        #Permute 
        permuted_inputs = inputs.permute(tuple(permutations))
        self.permuted_size = permuted_inputs.size()
        
        # Reshape (flattening 'input' through dimension 'dims')
        reshape = tuple([permuted_inputs.size(k) for k,_ in enumerate(remaining_dims)]+[-1]) 
        reshaped_inputs = permuted_inputs.reshape(reshape)

        self.reshaped_inputs_dim =  reshaped_inputs.size()
        return(reshaped_inputs)

    def get_stats(self,inputs: torch.Tensor): #,dims: tuple
        ''' Return Min, Max, Mean and Std of inputs through the choosen dimension 'dims' (which have been flattened)'''
        if (not(hasattr(self,'mini'))):
            self.mini = inputs.min(-1).values  
            #self.mini = inputs.min(dims).values  
        if (not(hasattr(self,'maxi'))): 
            self.maxi = inputs.max(-1).values
            #self.maxi = inputs.max(dims).values
        if (not(hasattr(self,'mean'))):
            self.mean= inputs.mean(-1)
            #self.mean = inputs.mean(dims)
        if (not(hasattr(self,'std'))): 
            self.std = inputs.std(-1)
            #self.std = inputs.std(dims)  

    def repeat_stats_tensor(self,X,S, feature_vect = False):
        '''
        According to argument 'dims', reshape and repeat tensor S to match dimension with X.

        args
        ----
        X : feature vector              >>>> torch.randn(T',N,C,H,W,L)
        I : Train input                 >>>> torch.randn(T,N,C,H,W)
        S : statistics (mini,mean...)   >>>> torch.randn(N,H)
        dims : dimension for which we have aggregated >>>> [0,2,4]  # cause we removed T,C,W from 'Train input'
        '''
        reshaped_vector, repeat_vector = [1]*X.dim(),[1]*X.dim()

        # Dépend de si c'est un Feature Vector (pour lequel on a ajouté une dimension L), ou un Input (comme train_input)
        conj_dims = [x for x in np.arange(X.dim()-1) if not x in self.dims] if feature_vect else [x for x in np.arange(X.dim()) if not x in self.dims]

        #Design re-shaping:
        for k,c in enumerate(conj_dims):
            reshaped_vector[c] = S.size(k)

        # Design repeating:
        for k,c in enumerate(X.size()):
            if reshaped_vector[k] == 1:
                repeat_vector[k] = c

        # Apply reshaped and repeat:
        reshaped_S = S.reshape(tuple(reshaped_vector))
        reshaped_S = reshaped_S.repeat(tuple(repeat_vector))
        return(reshaped_S)

    def transform(self,inputs: torch.Tensor, reverse: bool = False,feature_vect: bool = False):

        # MinMax Normalization
        if self.minmaxnorm:
            stacked_mini = self.repeat_stats_tensor(inputs,self.mini,feature_vect)
            stacked_maxi = self.repeat_stats_tensor(inputs,self.maxi,feature_vect)

            #stacked_mini = torch.stack([self.mini]*self.reshaped_inputs_dim[-1],-1)
            #stacked_maxi = torch.stack([self.maxi]*self.reshaped_inputs_dim[-1],-1)

            if reverse:
                return((inputs*(stacked_maxi-stacked_mini) + stacked_mini))
            else: 
                output_with_nan_and_inf = (inputs - stacked_mini)/(stacked_maxi-stacked_mini)  # Sometimes issues when divided by 0
                return(self.tackle_nan_inf_values(output_with_nan_and_inf))
        # ...
            
        # Z-Standardization 
        elif self.standardize:
            stacked_mean = self.repeat_stats_tensor(inputs,self.mean,feature_vect)
            stacked_std = self.repeat_stats_tensor(inputs,self.std,feature_vect)

            #stacked_mean = torch.stack([self.mean]*self.reshaped_inputs_dim[-1],-1)
            #stacked_std = torch.stack([self.std]*self.reshaped_inputs_dim[-1],-1)

            if reverse:
                return(inputs*stacked_std + stacked_mean)
            else: 
                output_with_nan_and_inf = (inputs - stacked_mean)/(stacked_std)  # Sometimes issues when divided by 0
                return(self.tackle_nan_inf_values(output_with_nan_and_inf)) 
        # ...

        else:
            raise ValueError('Standardization method has not been precised. Set minamxnorm = True or standardize = True')


    def tackle_nan_inf_values(self,output_with_nan_and_inf):
        '''For each channel and each station, we can have some issues when the minimum from Training Set is equal to its Maximum. We then can't normalize the dataset and set the values to 0. '''
        regular_values_set_to_0 =  torch.isinf(output_with_nan_and_inf).sum()
        Values_with_normalization_issues = (torch.isnan(output_with_nan_and_inf) + torch.isinf(output_with_nan_and_inf)).sum()
        if (regular_values_set_to_0 > 0) or (Values_with_normalization_issues>0):
            print('Values with issues: ','{:.3%}'.format(Values_with_normalization_issues.item()/output_with_nan_and_inf.numel() ))
            print('Regular Values that we have to set to 0: ','{:.3%}'.format(regular_values_set_to_0.item()/output_with_nan_and_inf.numel() ))
        output = torch.nan_to_num(output_with_nan_and_inf,0,0,0)  # Set 0 when devided by maxi - mini = 0 (0 when Nan, 0 when +inf, 0 when -inf
        return(output)
    

    def unormalize_tensor(self,inputs: torch.Tensor, feature_vect: bool = False):
        unormalized_tensor = self.normalize_tensor(inputs, reverse=True,feature_vect = feature_vect)
        return(unormalized_tensor)

    def normalize_tensor(self,tensor: torch.Tensor,reverse: bool =False,feature_vect: bool = False):
        '''
        args 
        -----
        inputs : n-dimension torch Tensor
        dims :  dimension through which we want to retrieve min/max or mean/std
        minmaxnorm : MinMax-Normalization if True
        standardize: Z-standardization if True 

        Examples:
            inputs = torch.randn(8,4,2,3,6)
            dims = [0,-1,-2]
            minmaxnorm  = True

            output is a Tensor object whose 'tensor' attribute is normalized (or unormalized)
            it returns the minmax-normalization of 'inputs' through dimensions 0,4,3. 
        '''
        normalized_tensor = self.transform(tensor,reverse,feature_vect)
        return(normalized_tensor)


class TensorDataset(object):
    def __init__(self,tensor,normalized,normalizer):
        super(TensorDataset,self).__init__()
        self.tensor = tensor 
        self.normalized = normalized
        self.normalizer = normalizer
    
    def normalize(self,feature_vect):
        assert not(self.normalized), 'TensorDataset already normalized'
        self.tensor = self.normalizer.normalize_tensor(self.tensor,reverse=False,feature_vect=feature_vect)
        self.normalized = True

    def unormalize(self,feature_vect):
        assert (self.normalized), 'TensorDataset already Un-normalized'
        self.tensor = self.normalizer.unormalize_tensor(self.tensor,feature_vect=feature_vect)    

    

class DataSet(object):
    '''
    attributes
    -------------
    df : contain the current df you are working on. It's the full df, normalized or not
    init_df : contain the initial df, no normalized. It's the full initial dataset.
    '''
    def __init__(self,df=None,tensor = None, dates = None, init_df = None, 
                 normalized = False,time_step_per_hour = None,
                 train_df = None,cleaned_df = None,Weeks = None, Days = None, 
                 historical_len = None,step_ahead = None,
                 standardize = None, minmaxnorm = None,dims = None,
                 spatial_unit = None,indices_spatial_unit = None):
        
        if df is not None:
            self.length = len(df)
            self.df = df
            self.spatial_unit = df.columns
            self.df_dates = pd.DataFrame(self.df.index,index = np.arange(len(self.df)),columns = ['date'])
            self.raw_values = torch.tensor(self.df.values)
        if tensor is not None:
            # tensor should follow this shape: [T,N,...]
            # >>>> [T,N,C,H,W] for an succession of T time-steps,  N spatial units, C channel by image (mobile app_1, .., mobile app_C), and H*W the image dimension
            # >>>> [T,N,C] for a succession of T time-steps, N spatial units, and C channel  (speed,flow, density)
            self.length = tensor.size(0)
            self.raw_values = tensor.to(torch.float32)
            self.df_dates = pd.DataFrame(dates,index = np.arange(self.length),columns = ['date'])
            self.spatial_unit = spatial_unit

        self.dims = dims
        self.minmaxnorm = minmaxnorm
        self.standardize = standardize
        self.normalized = normalized
        self.time_step_per_hour = time_step_per_hour
        self.train_df = train_df
        self.indices_spatial_unit = indices_spatial_unit
        if time_step_per_hour is not None :
            self.Week_nb_steps = int(7*24*self.time_step_per_hour)
            self.Day_nb_steps = int(24*self.time_step_per_hour)
        else : 
            self.Week_nb_steps = None
            self.Day_nb_steps = None
        

        if init_df is not None:
            self.init_df = init_df
        else:
            self.init_df = df

        self.shift_from_first_elmt = None
        self.U = None
        self.Utarget = None
        self.df_verif = None
        self.df_shifted = None
        self.invalid_indx_df = None
        self.remaining_dates = None
        self.time_slots_labels = None
        self.step_ahead = step_ahead
        self.Weeks = Weeks
        self.Days = Days
        self.historical_len = historical_len
        self.cleaned_df = cleaned_df

        
    def bijection_name_indx(self):
        colname2indx = {c:k for k,c in enumerate(self.spatial_unit)}
        indx2colname = {k:c for k,c in enumerate(self.spatial_unit)}
        return(colname2indx,indx2colname)
    
    def get_shift_from_first_elmt(self):
        shift_week = self.Weeks if self.Weeks is not None else 0
        shift_day = self.Days if self.Days is not None else 0
        self.shift_from_first_elmt = int(max(shift_week*24*7*self.time_step_per_hour,
                                shift_day*24*self.time_step_per_hour,
                                self.historical_len+self.step_ahead-1
                                ))
        self.shift_between_set = self.shift_from_first_elmt*timedelta(hours = 1/self.time_step_per_hour)


    def clean_dataset_get_tensor_and_train_valid_test_split(self,df,invalid_dates,train_prop,valid_prop,test_prop,normalize):
        '''
        Create a DataSet object from pandas dataframe. Retrieve associated Feature Vector and Target Vector. Remove forbidden indices (dates). Then split it into Train/Valid/Test inputs.
        '''
        dataset = DataSet(df, Weeks = self.Weeks, Days = self.Days, historical_len= self.historical_len,
                                   step_ahead=self.step_ahead,time_step_per_hour=self.time_step_per_hour)
        
        dataset.get_shift_from_first_elmt()   # récupère le 'shift from first elmt' pour la construction du feature vect 
        dataset.get_feature_vect(invalid_dates)  # Construction du feature vect  self.U et self.Utarget 

        dataset.train_valid_test_split_indices(train_prop,valid_prop,test_prop)  # Create df_train,df_valid,df_test, df_verif_train, df_verif_valid, df_verif_test, and dates limits for each df and each tensor U
        dataset.split_tensors(normalize) # Récupère U_test, Utarget_test, NetMob_test, Weather_test etc....  dans 'dataset_init.contextual_tensors.items()' 
        return(dataset)
    
    def warning(self):
        '''Warning in case we don't use trafic data: '''
        if self.Weeks+self.historical_len+self.Days == 0:
            print(f"! H+D+W = {self.Weeks+self.historical_len+self.Days}, which mean the Tensor U will be set to a Null vector")

    def mask_tensor(self):
        # Mask for Tensor U, Utarget
        mask_U =  [e for e in np.arange(self.U.shape[0]) if e not in self.forbidden_indice_U]
        # Apply mask 
        self.U = self.U[mask_U]
        self.Utarget = self.Utarget[mask_U]



    def get_feature_vect(self,invalid_dates): 
        # Get shifted Feature Vector and shifted Target
        featurevectorbuilder = FeatureVectorBuilder(self.step_ahead,self.historical_len,self.Days,self.Weeks,self.Day_nb_steps,self.Week_nb_steps,self.shift_from_first_elmt)
        featurevectorbuilder.build_feature_vect(self.raw_values)
        featurevectorbuilder.build_target_vect(self.raw_values)

        # Def Tensor Input  U and  target Tensor Utarget
        self.U = featurevectorbuilder.U
        self.Utarget = featurevectorbuilder.Utarget
        #  ...

        # Get forbidden indices, and df_verif to check just in case 
        dates_verif_object = DatesVerifFeatureVect(self.df_dates, Weeks = self.Weeks, Days = self.Days, historical_len = self.historical_len, step_ahead = self.step_ahead, time_step_per_hour = self.time_step_per_hour)
        dates_verif_object.get_df_verif(invalid_dates)

        self.forbidden_indice_U = dates_verif_object.forbidden_indice_U
        self.df_verif  = dates_verif_object.df_verif
        # ...
        self.mask_tensor()


    def get_dic_split_limits(self,train_prop,valid_prop,test_prop,
                             train_valid_test_split_method):
        # Split with iterative method 
        if hasattr(self,'Dataset_save_folder'):
            split_path = f"{self.Dataset_save_folder}split_limits.pkl" 
        else:
            split_path = ''

        if train_valid_test_split_method == 'iterative_method':
            if split_path and (os.path.exists(split_path)):   #not empty & path exist
                try:
                    split_limits = read_object(split_path)
                except:
                    split_limits= train_valid_test_split.iterative_method(self,self.df_verif,train_prop,valid_prop,test_prop)
                    save_object(split_limits, split_path)
                    print(f"split_limits.pkl has never been saved or issue with last .pkl save")
            else : 
                split_limits= train_valid_test_split.iterative_method(self,self.df_verif,train_prop,valid_prop,test_prop)
                if split_path: save_object(split_limits, split_path)  #if not empty, save it 

        elif train_valid_test_split_method == 'similar_length_method':
            split_limits= train_valid_test_split.similar_length_method(self,self.df_verif,train_prop,valid_prop,test_prop)
            print(f'>>>> Train/Valid/Test split method : {train_valid_test_split_method}')

        else:
            raise NotImplementedError(f'Train/Valid/Test split method {train_valid_test_split_method} has not been implemented')

        return(split_limits)


    def train_valid_test_split_indices(self,train_prop,valid_prop,test_prop,train_valid_test_split_method):

        split_limits = self.get_dic_split_limits(train_prop,valid_prop,test_prop,train_valid_test_split_method)
        tensor_limits_keeper = TensorLimitsKeeper(split_limits,self.df_dates,self.df_verif,train_prop,valid_prop, test_prop,self.step_ahead)
        for training_mode in ['train','valid','test']:
            tensor_limits_keeper.get_local_df_verif(training_mode)   # Build DataFrame Verif associated to each training mode

            tensor_limits_keeper.keep_track_on_df_limits(training_mode)   # Keep track on DataFrame Limits (dates)
            tensor_limits_keeper.get_raw_values_indices(training_mode)
            tensor_limits_keeper.get_raw_tensor_input_by_training_mode(self,training_mode)
            tensor_limits_keeper.keep_track_on_feature_vect_limits(training_mode)
        
        self.tensor_limits_keeper = tensor_limits_keeper


    def get_dataloader(self):
        ''' Build DataLoader '''
        # Train, Valid, Test split : 
        contextual_train  = {name: self.contextual_tensors[name]['train'] for name in self.contextual_tensors.keys()} #[self.contextual_tensors[name]['train'] for name in self.contextual_tensors.keys()]
        contextual_valid  = {name: self.contextual_tensors[name]['valid'] if 'valid' in self.contextual_tensors[name].keys() else None for name in self.contextual_tensors.keys()}   # [self.contextual_tensors[name]['valid'] for name in self.contextual_tensors.keys()]
        contextual_test  =  {name: self.contextual_tensors[name]['test'] if 'test' in self.contextual_tensors[name].keys() else None for name in self.contextual_tensors.keys()} # [self.contextual_tensors[name]['test'] for name in self.contextual_tensors.keys()] 

        train_tuple =  self.U_train,self.Utarget_train, contextual_train # *contextual_train
        valid_tuple =  (self.U_valid,self.Utarget_valid, contextual_valid)  if hasattr(self,'U_valid') else None # *contextual_valid
        test_tuple =  (self.U_test,self.Utarget_test, contextual_test) if hasattr(self,'U_test') else None# *contextual_test

        # Load DictDataLoader: 
        DictDataLoader_object = DictDataLoader(train_tuple, valid_tuple, test_tuple,self.args)
        dict_dataloader = DictDataLoader_object.get_dictdataloader()
        self.dataloader = dict_dataloader


    def split_normalize_load_feature_vect(self,invalid_dates,train_prop,valid_prop,test_prop,train_valid_test_split_method,normalize = True
                                          #,calib_prop,batch_size,calendar_class
                                          ):
        self.get_shift_from_first_elmt()   # get shift indice and shift date from the first element / between each dataset 
        self.get_feature_vect(invalid_dates)  # Removed the forbidden dates, Build 'df_verif' and the Feature Vector Tensor masked by the forbidden indices.

        # Get Index to Split df, U, Utarget, time_slots_labels
        self.train_valid_test_split_indices(train_prop,valid_prop,test_prop,train_valid_test_split_method)  # Create df_train,df_valid,df_test, df_verif_train, df_verif_valid, df_verif_test, and dates limits for each df and each tensor U

        # Get all the splitted train/valid/test input tensors. Normalize Them 
        self.split_tensors(normalize = normalize)


    def set_train_valid_test_tensor_attribute(self,name,tensor):
        ''' 
        args
        ----
        ref_for_normalization: represents the reference to be used for normalization.
        tensor.size(0) <= ref_for_normalization.size(0)  and can must have an additional axis in the last position. 
        All other dimensions  are identical to Tensor => tensor.size(k) = ref_for_normalization.size(k) for k < tensor.dim()
        >>>> Example: if I've used T time-slots of N stations to produce a feature vector (of length T-p)
        >>>> I'll use these T time-slots to retrieve the associated statistics (min, max, mean, std) 
        >>>> And then apply my standardization.


        dims_agg: represent the dimensions on which aggregtion is to be applied (to compute statistics).
        >>>> dims_agg = (0,2), tensor = torch.randn(4,5,6,7)
        >>>> output : mini.size() = [5,7],  mean.size() = [5,7] ....
        '''
        


        splitter = TrainValidTest_Split_Normalize(tensor,
                                    first_train = self.tensor_limits_keeper.first_train_U, last_train= self.tensor_limits_keeper.last_train_U,
                                    first_valid= self.tensor_limits_keeper.first_valid_U, last_valid = self.tensor_limits_keeper.last_valid_U,
                                    first_test = self.tensor_limits_keeper.first_test_U, last_test = self.tensor_limits_keeper.last_test_U,
                                    minmaxnorm = self.minmaxnorm,standardize = self.standardize)
        
        
        train_tensor_ds,valid_tensor_ds,test_tensor_ds = splitter.split_normalize_tensor_datasets(normalizer = self.normalizer)

        # Tackle Train Tensor:
        setattr(self,f"{name}_train", train_tensor_ds.tensor)

        # Tackle Valid Tensor:
        if hasattr(splitter,'first_valid'):
            setattr(self,f"{name}_valid", valid_tensor_ds.tensor) 

        # Tackle Test Tensor: 
        if hasattr(splitter,'first_test'):
            setattr(self,f"{name}_test", test_tensor_ds.tensor)

    def display_info_on_inputs(self):
        print('\nU size: ',self.U.size(),'Utarget size: ',self.Utarget.size())
        print('U_train size: ',self.U_train.size(),'Utarget_train size: ',self.Utarget_train.size())
        if hasattr(self,'U_valid'): print('U_valid size: ',self.U_valid.size(),'Utarget_valid size: ',self.Utarget_valid.size()) 
        if hasattr(self,'U_test'): print('U_test size: ', self.U_test.size(),'Utarget_test size: ',self.Utarget_test.size())

        print('U_train min: ',self.U_train.min(), 'U_train max: ',self.Utarget_train.max())
        if hasattr(self,'U_valid'): print('U_valid min: ',self.U_valid.min(),'U_valid max: ',self.Utarget_valid.max())
        if hasattr(self,'U_test'): print('U_test min: ',self.U_test.min(),'U_test max: ',self.Utarget_test.max())

    def split_tensors(self,normalize):
        ''' Split input tensors  in Train/Valid/Test part '''
        self.normalizer = Normalizer(reference = self.train_input,minmaxnorm = self.minmaxnorm, standardize = self.standardize, dims = self.dims) if normalize else None
        # Get U_train, U_valid, U_test
        self.set_train_valid_test_tensor_attribute('U',self.U)

        # Get Utarget_train, Utarget_valid, Utarget_test 
        self.set_train_valid_test_tensor_attribute('Utarget',self.Utarget)

        self.display_info_on_inputs()

        if False : 
            # Get NetMob_train, NetMob_valid, NetMob_test, Weather_train etc etc ...
            if hasattr(self,'contextual_tensors'):
                for name, tensor_dict in self.contextual_tensors.items():
                    feature_vect = tensor_dict['feature_vect']
                    dims = tensor_dict['dims']
                    raw_data = tensor_dict['raw_data']
                    normalize = tensor_dict['normalize']
                    self.set_train_valid_test_tensor_attribute(name,feature_vect,dims,raw_data, normalize = normalize)
            else:
                print('\nNo Contextual Data has been considered')

        #if self.time_slots_labels is not None : 
        #    self.time_slots_train = {calendar_class: self.time_slots_labels[calendar_class][self.first_train_U:self.last_train_U] for calendar_class in range(len(self.nb_class)) }
        #    self.time_slots_valid = {calendar_class: self.time_slots_labels[calendar_class][self.first_valid_U:self.last_valid_U] if self.first_valid_U is not None else None for calendar_class in range(len(self.nb_class))}
        #    self.time_slots_test = {calendar_class: self.time_slots_labels[calendar_class][self.first_test_U:self.last_test_U] if self.first_test_U is not None else None for calendar_class in range(len(self.nb_class)) }


class PersonnalInput(DataSet):
    def __init__(self,invalid_dates,arg_parser,*args, **kwargs):
        super(PersonnalInput,self).__init__(*args, **kwargs)
        self.invalid_dates = invalid_dates
        self.args = arg_parser
        
    def preprocess(self,train_prop,valid_prop,test_prop,train_valid_test_split_method,normalize = True):
        self.split_normalize_load_feature_vect(self.invalid_dates,train_prop,valid_prop,test_prop,train_valid_test_split_method,normalize)
