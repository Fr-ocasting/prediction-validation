import torch
import numpy as np 
import pandas as pd 
import os 
from datetime import timedelta
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

# Personnal Import 
from pipeline.DL_class import FeatureVectorBuilder,DatesVerifFeatureVect,TensorLimitsKeeper
from pipeline.DataSet.train_valid_test_split import iterative_method, similar_length_method
from pipeline.trainer.loader import DictDataLoader
from pipeline.utils.save_results import save_object,read_object
from pipeline.preprocessing.data_augmentation.data_augmentation import DataAugmenter
from pipeline.utils.utilities import load_inputs_from_dataloader
from pipeline.DataSet.Normalizer import Normalizer
from pipeline.DataSet.splitter import SplitterTrainValidTest
# ...

    

class DataSet(object):
    '''
    attributes
    -------------
    df : contain the current df you are working on. It's the full df, normalized or not
    init_df : contain the initial df, no normalized. It's the full initial dataset.
    '''
    def __init__(self,df=None,tensor = None, dates = None, init_df = None, 
                 normalized = False,time_step_per_hour = None,
                 train_df = None,cleaned_df = None,W = None, D = None, 
                 H = None,step_ahead = None, horizon_step = None,
                 standardize = None, minmaxnorm = None,dims = None,
                 spatial_unit = None,indices_spatial_unit = None,city = None,
                 data_augmentation=False,
                 periods = None,
                 DA_moment_to_focus=None,
                 DA_method=None,
                 DA_min_count= None,
                 DA_alpha = None,
                 DA_prop = None,
                 DA_noise_from = None,
                 DA_magnitude_max_scale = None,
                 target_data = None,
                 out_dim_factor = None,
                 expanding_train = None,

                 ):
        
        if df is not None:
            self.length = len(df)
            self.df = df
            self.spatial_unit = df.columns
            self.num_nodes = len(self.spatial_unit)
            self.C = 1
            self.df_dates = pd.DataFrame(self.df.index,index = np.arange(len(self.df)))
            self.df_dates.columns = ['date']

            self.raw_values = torch.tensor(self.df.values)
        if tensor is not None:
            # tensor should follow this shape: [T,N,...]
            # >>>> [T,N,C,H,W] for an succession of T time-steps,  N spatial units, C channel by image (mobile app_1, .., mobile app_C), and H*W the image dimension
            # >>>> [T,N,C] for a succession of T time-steps, N spatial units, and C channel  (speed,flow, density)
            self.length = tensor.size(0)
            self.num_nodes = tensor.size(1)
            # if tensor.dim()>2:
            #     self.C = tensor.size(2)
            # else:
            #     self.C = 1
            self.raw_values = tensor.to(torch.float32)
            self.df_dates = pd.DataFrame(dates,index = np.arange(self.length),columns = ['date'])
            self.spatial_unit = spatial_unit
        self.city  = city
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
        self.horizon_step = horizon_step
        self.out_dim_factor = out_dim_factor
        self.W = W
        self.D = D
        self.H = H
        self.cleaned_df = cleaned_df
        self.target_data = target_data
        self.expanding_train = expanding_train

        # Data Augmentation: 
        self.data_augmentation = data_augmentation
        self.DA_moment_to_focus = DA_moment_to_focus
        self.DA_method = DA_method
        self.DA_min_count=DA_min_count
        self.DA_alpha=DA_alpha
        self.DA_prop = DA_prop
        self.DA_magnitude_max_scale = DA_magnitude_max_scale
        self.periods = periods

        


        
    # def bijection_name_indx(self):
    #     colname2indx = {c:k for k,c in enumerate(self.spatial_unit)}
    #     indx2colname = {k:c for k,c in enumerate(self.spatial_unit)}
    #     return(colname2indx,indx2colname)
    
    def get_shift_from_first_elmt(self):
        shift_week = self.W if self.W is not None else 0
        shift_day = self.D if self.D is not None else 0
        self.shift_from_first_elmt = int(max(shift_week*24*7*self.time_step_per_hour,
                                shift_day*24*self.time_step_per_hour,
                                self.H+self.step_ahead-1
                                ))
        self.shift_between_set = self.shift_from_first_elmt*timedelta(hours = 1/self.time_step_per_hour)


    def warning(self):
        '''Warning in case we don't use trafic data: '''
        if self.warning+self.H+self.D == 0:
            print(f"! H+D+W = {self.W+self.H+self.D}, which mean the Tensor U will be set to a Null vector")

    def mask_tensor(self):
        # Mask for Tensor U, Utarget
        mask_U =  [e for e in np.arange(self.U.shape[0]) if e not in self.forbidden_indice_U]
        # Apply mask 
        self.U = self.U[mask_U]
        self.Utarget = self.Utarget[mask_U]

        if self.horizon_step > 1:
            self.predicted_indices = torch.arange(self.horizon_step - 1, self.step_ahead, self.horizon_step)
            self.Utarget = torch.index_select(self.Utarget, dim=-1, index=self.predicted_indices)
            self.df_verif = self.df_verif.drop(columns=[f"t+{sa}" for sa in range(self.step_ahead) if not sa in  (self.predicted_indices)])

            #print('\nself.step_ahead: ',self.step_ahead)
            #print('df_verif: ',self.df_verif.head(2))




    def get_feature_vect(self,invalid_dates): 
        # Get shifted Feature Vector and shifted Target
        featurevectorbuilder = FeatureVectorBuilder(self.step_ahead,self.H,self.D,self.W,self.Day_nb_steps,self.Week_nb_steps,self.shift_from_first_elmt)
        featurevectorbuilder.build_feature_vect(self.raw_values)
        featurevectorbuilder.build_target_vect(self.raw_values)

        # Def Tensor Input  U and  target Tensor Utarget
        self.U = featurevectorbuilder.U
        self.Utarget = featurevectorbuilder.Utarget
        #  ...

        # Get forbidden indices, and df_verif to check just in case 
        dates_verif_object = DatesVerifFeatureVect(self.df_dates, Weeks = self.W, Days = self.D, historical_len = self.H, step_ahead = self.step_ahead, time_step_per_hour = self.time_step_per_hour,target_data = self.target_data)
        dates_verif_object.get_df_verif(invalid_dates)
        self.forbidden_indice_U = dates_verif_object.forbidden_indice_U
        self.df_verif  = dates_verif_object.df_verif
        # ...
        self.mask_tensor()

    def get_dic_split_limits(self,train_prop,valid_prop,test_prop,
                             train_valid_test_split_method,tensor_limits_keeper =None):
        
        if tensor_limits_keeper is not None:
            split_limits = tensor_limits_keeper.split_limits
        else:
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
                        split_limits= iterative_method(self,self.df_verif,train_prop,valid_prop,test_prop)
                        save_object(split_limits, split_path)
                        print(f"split_limits.pkl has never been saved or issue with last .pkl save")
                else : 
                    split_limits= iterative_method(self,self.df_verif,train_prop,valid_prop,test_prop)
                    if split_path: save_object(split_limits, split_path)  #if not empty, save it 

            elif train_valid_test_split_method == 'similar_length_method':
                split_limits= similar_length_method(self,self.df_verif,train_prop,valid_prop,test_prop)

            else:
                raise NotImplementedError(f'Train/Valid/Test split method {train_valid_test_split_method} has not been implemented')

        return(split_limits)


    def train_valid_test_split_indices(self,train_prop,valid_prop,test_prop,train_valid_test_split_method,tensor_limits_keeper):

        split_limits = self.get_dic_split_limits(train_prop,valid_prop,test_prop,train_valid_test_split_method,tensor_limits_keeper)
        tensor_limits_keeper = TensorLimitsKeeper(split_limits,self.df_dates,self.df_verif,train_prop,valid_prop, test_prop,self.step_ahead)
        for training_mode in ['train','valid','test']:
            tensor_limits_keeper.get_local_df_verif(training_mode)   # Build DataFrame Verif associated to each training mode

            # # --- EXPANDING TRAIN: supposed to work here, but really weird results: 
            # if (self.expanding_train is not None and self.expanding_train != 1) and (training_mode == 'train'):
            #     size_t = len(tensor_limits_keeper.df_verif_train)
            #     tensor_limits_keeper.df_verif_train =  tensor_limits_keeper.df_verif_train.iloc[int(size_t*(1-self.expanding_train)):,:]  # Keep only the lasts 'train_pourcent' of the training set
            #print(f"df_verif_{training_mode}: {getattr(tensor_limits_keeper,f'df_verif_{training_mode}')}")
            tensor_limits_keeper.keep_track_on_df_limits(training_mode)   # Keep track on DataFrame Limits (dates)
            tensor_limits_keeper.get_raw_values_indices(training_mode)
            tensor_limits_keeper.get_raw_tensor_input_by_training_mode(self,training_mode)
            tensor_limits_keeper.keep_track_on_feature_vect_limits(training_mode)
    
        self.tensor_limits_keeper = tensor_limits_keeper

    def get_data_augmentation(self,contextual_train):
        ''' Implement a Data-augmentation on the train dataset.

        args:
        -----
        DA_moment_to_focus: focus on some specific moments based on calendar information.
        
        Examples: 
        >>> DA_moment_to_focus = [{'hours':[0,23],'weekdays':[1,3]}] Will focus on Tuesday, Thursday at 23h-00h, i.e last subway

        '''

        if self.data_augmentation:
            data_augmenter = DataAugmenter(self,self.DA_method,self.DA_moment_to_focus,self.DA_magnitude_max_scale)
            U_train_augmented,Utarget_train_augmented,contextual_train_augmented = data_augmenter.DA_augmentation(self.U_train,self.Utarget_train,contextual_train,ds = self,alpha = self.DA_alpha, p = self.DA_prop)
            self.U_train = U_train_augmented
            self.Utarget_train = Utarget_train_augmented

            print('Train/Target size: ',self.U_train.size(),self.Utarget_train.size())
            return contextual_train_augmented

        else:
            return contextual_train


    def get_dataloader(self):
        ''' Build DataLoader '''
        # Train, Valid, Test split : 
        contextual_train = {name_i: contextual_i['train'] for name_i,contextual_i in self.contextual_tensors.items()}  
        contextual_valid ={name_i: contextual_i['valid'] for name_i,contextual_i in self.contextual_tensors.items() if 'valid' in contextual_i.keys()}     # {name: self.contextual_tensors[name]['valid'] if 'valid' in self.contextual_tensors[name].keys() else None for name in self.contextual_tensors.keys()}   # [self.contextual_tensors[name]['valid'] for name in self.contextual_tensors.keys()]
        contextual_test  = {name_i: contextual_i['test'] for name_i,contextual_i in self.contextual_tensors.items() if 'test' in contextual_i.keys()}    #  {name: self.contextual_tensors[name]['test'] if 'test' in self.contextual_tensors[name].keys() else None for name in self.contextual_tensors.keys()} # [self.contextual_tensors[name]['test'] for name in self.contextual_tensors.keys()] 

        # Data Augmentation if needed : 
        contextual_train = self.get_data_augmentation(contextual_train)


        # To keep track on contextual train and see the impact of 'noise': 
        if True:
            self.contextual_train = contextual_train

        # Display usefull information: 
        self.display_info_on_inputs()
        # Data Loader: 
        if self.expanding_train is not None: 
            split = int(self.U_train.size(0)*self.expanding_train)
            self.U_train = self.U_train[-split:]
            self.Utarget_train = self.Utarget_train[-split:]
            contextual_train = {name_i: contextual_i[-split:] for name_i,contextual_i in contextual_train.items()}
            print(f'   Expanding Train activated: keeping only the last {self.expanding_train*100}% of the train set => New Train size: {self.U_train.size(0)}')

        train_tuple =  self.U_train,self.Utarget_train, contextual_train # *contextual_train
        valid_tuple =  (self.U_valid,self.Utarget_valid, contextual_valid)  if hasattr(self,'U_valid') else None # *contextual_valid
        test_tuple =  (self.U_test,self.Utarget_test, contextual_test) if hasattr(self,'U_test') else None# *contextual_test

        # Load DictDataLoader: 
        DictDataLoader_object = DictDataLoader(train_tuple, valid_tuple, test_tuple,self.args)
        dict_dataloader = DictDataLoader_object.get_dictdataloader()
        self.dataloader = dict_dataloader

    def load_all_inputs_from_training_mode(self,training_mode):
        X,Y,X_c,nb_contextual = load_inputs_from_dataloader(self.dataloader[training_mode],self.args.device)
        return X,Y,X_c,nb_contextual


    def split_normalize_load_feature_vect(self,invalid_dates,train_prop,valid_prop,test_prop,train_valid_test_split_method,normalize = True,tensor_limits_keeper = None,
                                          ):
        self.get_shift_from_first_elmt()   # get shift indice and shift date from the first element / between each dataset 
        self.get_feature_vect(invalid_dates)  # Removed the forbidden dates, Build 'df_verif' and the Feature Vector Tensor masked by the forbidden indices.

        # Get Index to Split df, U, Utarget, time_slots_labels
        self.train_valid_test_split_indices(train_prop,valid_prop,test_prop,train_valid_test_split_method,tensor_limits_keeper)  # Create df_train,df_valid,df_test, df_verif_train, df_verif_valid, df_verif_test, and dates limits for each df and each tensor U

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
        


        splitter = SplitterTrainValidTest(tensor,
                                    first_train = self.tensor_limits_keeper.first_train_U, last_train= self.tensor_limits_keeper.last_train_U,
                                    first_valid= self.tensor_limits_keeper.first_valid_U, last_valid = self.tensor_limits_keeper.last_valid_U,
                                    first_test = self.tensor_limits_keeper.first_test_U, last_test = self.tensor_limits_keeper.last_test_U,
                                    minmaxnorm = self.minmaxnorm,standardize = self.standardize)
        
        
        train_tensor_ds,valid_tensor_ds,test_tensor_ds = splitter.split_normalize_tensor_datasets(normalizer = self.normalizer)


        # # # --- EXPANDING TRAIN: supposed to work here, but really weird results: 
        # if self.expanding_train != 100:
        #     size_t = train_tensor_ds.tensor.size(0)
        #     train_tensor_ds.tensor = train_tensor_ds.tensor[int(size_t*(1-self.expanding_train)):] 


        # Tackle Train Tensor:
        setattr(self,f"{name}_train", train_tensor_ds.tensor)

        # Tackle Valid Tensor:
        if hasattr(splitter,'first_valid'):
            setattr(self,f"{name}_valid", valid_tensor_ds.tensor) 

        # Tackle Test Tensor: 
        if hasattr(splitter,'first_test'):
            setattr(self,f"{name}_test", test_tensor_ds.tensor)

    def display_info_on_inputs(self):
        str_to_display = ''
        str_to_display = f"Init {str_to_display}U/Utarget size: {self.U.size()}/{self.Utarget.size()}"
        str_to_display = f"{str_to_display} Train"
        if hasattr(self,'U_valid'):
            str_to_display = f"{str_to_display}/Valid"
        if hasattr(self,'U_test'):
            str_to_display = f"{str_to_display}/Test"

        str_to_display = f"{str_to_display} {self.U_train.size(0)}"

        if hasattr(self,'U_valid'):
            str_to_display = f"{str_to_display} {self.U_valid.size(0)}"
        if hasattr(self,'U_test'):
            str_to_display = f"{str_to_display} {self.U_test.size(0)}"
        print(str_to_display)
        
    def split_tensors(self,normalize):
        ''' Split input tensors  in Train/Valid/Test part '''
        self.normalizer = Normalizer(reference = self.train_input,minmaxnorm = self.minmaxnorm, standardize = self.standardize, dims = self.dims) if normalize else None
        # Get U_train, U_valid, U_test
        self.set_train_valid_test_tensor_attribute('U',self.U)

        # Get Utarget_train, Utarget_valid, Utarget_test 
        self.set_train_valid_test_tensor_attribute('Utarget',self.Utarget)


        #print('self.df_verif_train: ',self.tensor_limits_keeper.df_verif_train)
        #print('self.U_train: ',self.U_train.size())
        #print('self.Utarget_train: ',self.Utarget_train.size())


        #self.display_info_on_inputs()


class PersonnalInput(DataSet):
    def __init__(self,invalid_dates,arg_parser,name,*args, **kwargs):
        super(PersonnalInput,self).__init__(*args, **kwargs)
        self.invalid_dates = invalid_dates
        self.args = arg_parser
        self.target_data = arg_parser.target_data
        self.out_dim_factor = arg_parser.out_dim_factor
        self.name = name
        
    def preprocess(self,train_prop,valid_prop,test_prop,train_valid_test_split_method,normalize = True,tensor_limits_keeper= None):
        self.split_normalize_load_feature_vect(self.invalid_dates,train_prop,valid_prop,test_prop,train_valid_test_split_method,normalize,tensor_limits_keeper)
