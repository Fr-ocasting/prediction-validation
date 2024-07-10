import torch
import numpy as np 
import pandas as pd 
import os 
from datetime import timedelta


# Personnal Import 
from DL_class import FeatureVectorBuilder,DatesVerifFeatureVect
from train_valid_test_split import train_valid_test_split_iterative_method
from loader import DictDataLoader
from save_results import save_object,read_object,Dataset_get_save_folder
from calendar_class import get_time_slots_labels
# ...

class TrainValidTest_Split_Normalize(object):
    def __init__(self,data,dims,
                 train_indices = None , valid_indices = None, test_indices = None,
                 first_train = None, last_train = None, first_valid = None, last_valid = None, first_test = None, last_test = None, 
                 minmaxnorm = False,standardize = False):
        super(TrainValidTest_Split_Normalize,self).__init__()
        self.data = data
        self.minmaxnorm = minmaxnorm
        self.standardize = standardize
        self.dims = dims 

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
            self.data_valid = self.data[self.first_valid:self.last_valid] if self.first_valid is not None else None
            self.data_test = self.data[self.first_test:self.last_test]   if self.first_test is not None else None
        else: 
            raise ValueError("Neither 'train_indices' nor 'first_train' attribute has been designed ")
        

    def load_normalize_tensor_datasets(self,mini = None, maxi = None, mean = None, std = None, normalize = True,feature_vect = True):
        '''Load TensorDataset (train_dataset) object from data_train.
        Define TensorDataset object from valid (valid_dataset) and test (test_dataset). 
        Associate statistics from train dataset to valid and test dataset
        Normalize them according to their statistics 
        '''
        # Define train_dataset and normalize it
        print('Tackling Training Set')
        train_dataset = TensorDataset(self.data_train, mini = mini, maxi = maxi, mean=mean, std=std)
        if normalize:
            train_dataset = train_dataset.normalize_tensor(train_dataset.tensor, self.dims, self.minmaxnorm, self.standardize, reverse = False,feature_vect=feature_vect)

        mini = train_dataset.mini if hasattr(train_dataset,'mini') else None
        maxi = train_dataset.maxi if hasattr(train_dataset,'maxi') else None
        mean = train_dataset.mean if hasattr(train_dataset,'mean') else None
        std =    train_dataset.std if hasattr(train_dataset,'std') else None 

        # Define valid_dataset
        print('Tackling Validation Set')
        valid_dataset = TensorDataset(self.data_valid,mini = mini , maxi = maxi, mean = mean , std = std )

        # Define test_dataset
        print('Tackling Testing Set')
        test_dataset = TensorDataset(self.data_test,mini = mini , maxi = maxi, mean = mean , std = std )
        
        
        # Normalize thank to stats from Training Set 
        if normalize:
            valid_dataset = valid_dataset.normalize_tensor(valid_dataset.tensor,self.dims, self.minmaxnorm, self.standardize, reverse = False,feature_vect=feature_vect)
            test_dataset = test_dataset.normalize_tensor(test_dataset.tensor, self.dims, self.minmaxnorm, self.standardize, reverse = False,feature_vect=feature_vect)

        return(train_dataset,valid_dataset,test_dataset)



class TensorDataset(object):
    def __init__(self,tensor,mini=None,maxi=None,mean=None,std=None, normalized = False):
        super(TensorDataset,self).__init__()
        if mini is not None: self.mini = mini 
        if maxi is not None: self.maxi = maxi 
        if mean is not None: self.mean = mean 
        if std is not None: self.std = std 
        self.normalized = normalized
        self.tensor = tensor 

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

    def repeat_stats_tensor(self,X,S, dims, feature_vect = False):
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
        conj_dims = [x for x in np.arange(X.dim()-1) if not x in dims] if feature_vect else [x for x in np.arange(X.dim()) if not x in dims]

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

    def transform(self,inputs: torch.Tensor, minmaxnorm: bool = False, standardize: bool = False, reverse: bool = False, dims: list = [],feature_vect: bool = False):

        # MinMax Normalization
        if minmaxnorm:
            stacked_mini = self.repeat_stats_tensor(inputs,self.mini,dims,feature_vect )
            stacked_maxi = self.repeat_stats_tensor(inputs,self.maxi,dims,feature_vect)

            #stacked_mini = torch.stack([self.mini]*self.reshaped_inputs_dim[-1],-1)
            #stacked_maxi = torch.stack([self.maxi]*self.reshaped_inputs_dim[-1],-1)

            if reverse:
                return((inputs*(stacked_maxi-stacked_mini) + stacked_mini))
            else: 
                output_with_nan_and_inf = (inputs - stacked_mini)/(stacked_maxi-stacked_mini)  # Sometimes issues when divided by 0
                return(self.tackle_nan_inf_values(output_with_nan_and_inf))
        # ...
            
        # Z-Standardization 
        elif standardize:
            stacked_mean = self.repeat_stats_tensor(inputs,self.mean,dims,feature_vect)
            stacked_std = self.repeat_stats_tensor(inputs,self.std,dims,feature_vect)

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
        print('Values with issues: ','{:.3%}'.format(Values_with_normalization_issues.item()/output_with_nan_and_inf.numel() ))
        print('Regular Values that we have to set to 0: ','{:.3%}'.format(regular_values_set_to_0.item()/output_with_nan_and_inf.numel() ))
        output = torch.nan_to_num(output_with_nan_and_inf,0,0,0)  # Set 0 when devided by maxi - mini = 0 (0 when Nan, 0 when +inf, 0 when -inf
        return(output)
    

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
        
        # Reshape
        reshape = tuple([permuted_inputs.size(k) for k,_ in enumerate(remaining_dims)]+[-1]) 
        reshaped_inputs = permuted_inputs.reshape(reshape)

        self.reshaped_inputs_dim =  reshaped_inputs.size()
        return(reshaped_inputs)

    def unormalize_tensor(self,inputs: torch.Tensor, dims: list, minmaxnorm: bool = False, standardize: bool = False):
        assert self.normalized, 'TensorDataset is already un-normalized'

        self.normalize_tensor(inputs, dims, minmaxnorm, standardize,reverse=True)

    def normalize_tensor(self,tensor: torch.Tensor, dims: list, minmaxnorm: bool = False, standardize: bool = False,reverse: bool =False,feature_vect: bool = False):
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
        assert not(self.normalized), 'TensorDataset already normalized'
        # Get Min, Max, Mean, Std if not already available 
        if ((not(hasattr(self,'mini'))) or (not(hasattr(self,'maxi')))) or ((not(hasattr(self,'mean'))) or (not(hasattr(self,'std')))):
            reshaped_inputs = self.reshape_input(tensor,dims)
            self.get_stats(reshaped_inputs)

        if False:
            # Normalize
            normalized_tensor = self.transform(reshaped_inputs,minmaxnorm,standardize,reverse)

            # reshape-back, inverse-permute
            normalized_tensor = self.inverse_reshape_permute(normalized_tensor)
        
        else:
            normalized_tensor = self.transform(tensor,minmaxnorm,standardize,reverse,dims,feature_vect)
            self.normalized = True

        return TensorDataset(normalized_tensor,mini=self.mini,maxi=self.maxi,mean=self.mean,std=self.std, normalized = not(reverse))
    
    # ======== Following method should be useless:  ========

    def inverse_reshape_permute(self,normalized_tensor):
        # Reshape and inverse-permute:
        normalized_tensor = normalized_tensor.reshape(self.permuted_size) #Un-flatten

        inverse_permute = torch.argsort(torch.LongTensor(self.permutations)).tolist()
        normalized_tensor = normalized_tensor.permute(inverse_permute) # inverse permutation 
        return(normalized_tensor)
    # ======== ...................................  ========


class DataSet(object):
    '''
    attributes
    -------------
    df : contain the current df you are working on. It's the full df, normalized or not
    init_df : contain the initial df, no normalized. It's the full initial dataset.
    '''
    def __init__(self,df=None,tensor = None, dates = None, init_df = None,mini= None, maxi = None, mean = None, normalized = False,time_step_per_hour = None,
                 train_df = None,cleaned_df = None,Weeks = None, Days = None, historical_len = None,step_ahead = None):
        
        if df is not None:
            self.length = len(df)
            self.df = df
            self.columns = df.columns
            self.df_dates = pd.DataFrame(self.df.index,index = np.arange(len(self.df)),columns = ['date'])
            self.raw_values = torch.tensor(self.df.values)
        if tensor is not None:
            # tensor should follow this shape: [T,N,...]
            # >>>> [T,N,C,H,W] for an succession of T time-steps,  N spatial units, C channel by image (mobile app_1, .., mobile app_C), and H*W the image dimension
            # >>>> [T,N,C] for a succession of T time-steps, N spatial units, and C channel  (speed,flow, density)
            self.length = tensor.size(0)
            self.raw_values = tensor
            self.df_dates = pd.DataFrame(dates,index = np.arange(self.length),columns = ['date'])


        self.normalized = normalized
        self.time_step_per_hour = time_step_per_hour
        self.train_df = train_df
        if time_step_per_hour is not None :
            self.Week_nb_steps = int(7*24*self.time_step_per_hour)
            self.Day_nb_steps = int(24*self.time_step_per_hour)
        else : 
            self.Week_nb_steps = None
            self.Day_nb_steps = None

        if mini is not None: 
            self.mini = mini
        else : 
            self.mini = df.min()

        if maxi is not None: 
            self.maxi = maxi
        else : 
            self.maxi = df.max()

        if mean is not None:
            self.mean = mean
        else:
            self.mean = df.mean()

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
        colname2indx = {c:k for k,c in enumerate(self.columns)}
        indx2colname = {k:c for k,c in enumerate(self.columns)}
        return(colname2indx,indx2colname)
    
    def get_shift_from_first_elmt(self):
        shift_week = self.Weeks if self.Weeks is not None else 0
        shift_day = self.Days if self.Days is not None else 0
        self.shift_from_first_elmt = int(max(shift_week*24*7*self.time_step_per_hour,
                                shift_day*24*self.time_step_per_hour,
                                self.historical_len+self.step_ahead-1
                                ))
        self.shift_between_set = self.shift_from_first_elmt*timedelta(hours = 1/self.time_step_per_hour)


    def standardize(self,x,reverse = False):
        ''' Standardization : z <- (x-mu) / sigma'''
        if reverse:
            x = x*(self.std) + self.mean
        else:
            x = (x-self.mean)/ self.std

    def minmaxnorm(self,x,reverse = False):
        ''' MinMax Normalization : z <- (x-min) / (max-min)'''
        if reverse:
            x = x*(self.maxi - self.mini) +self.mini
        else :
            x = (x-self.mini)/(self.maxi-self.mini)
        return x   

    def clean_dataset_get_tensor_and_train_valid_test_split(self,df,invalid_dates,train_prop,valid_prop,test_prop,normalize,dims_agg):
        '''
        Create a DataSet object from pandas dataframe. Retrieve associated Feature Vector and Target Vector. Remove forbidden indices (dates). Then split it into Train/Valid/Test inputs.
        '''
        dataset = DataSet(df, Weeks = self.Weeks, Days = self.Days, historical_len= self.historical_len,
                                   step_ahead=self.step_ahead,time_step_per_hour=self.time_step_per_hour)
        
        dataset.get_shift_from_first_elmt()   # récupère le 'shift from first elmt' pour la construction du feature vect 
        dataset.get_feature_vect(invalid_dates)  # Construction du feature vect  self.U et self.Utarget 

        dataset.train_valid_test_split_indices(train_prop,valid_prop,test_prop)  # Create df_train,df_valid,df_test, df_verif_train, df_verif_valid, df_verif_test, and dates limits for each df and each tensor U
        dataset.split_tensors(normalize,dims_agg) # Récupère U_test, Utarget_test, NetMob_test, Weather_test etc....  dans 'dataset_init.contextual_tensors.items()' 
        return(dataset)
    
    def warning(self):
        '''Warning in case we don't use trafic data: '''
        if self.Weeks+self.historical_len+self.Days == 0:
            print(f"! H+D+W = {self.Weeks+self.historical_len+self.Days}, which mean the Tensor U will be set to a Null vector")


    def split_K_fold(self,args,invalid_dates,netmob = False,dims = [0]):
        '''
        Split la DataSet Initiale en K-fold
        args 
        -------


        outputs
        -------
        fold_dataset_limits: dict -> Contains TimeStamps Limits and Indices Limits of each Fold and each training modes. 
        examples : 
            - fold_dataset_limits[fold_k]['valid']['timestamp']  = (Timestamp('2019-02-26 00:00:00', freq='15T'),Timestamp('2019-04-27 11:45:00', freq='15T'))
            - fold_dataset_limits[fold_k]['valid']['tensor_indices']  = (50,146)


        '''
        print(f'Compute statistics through dimensions: {dims} ')
        self.warning()
        self.fold_dataset_limits = {k : {name : {} for name in ['fold_limits','train','valid','test']} for k in range(args.K_fold)}
        Datasets,DataLoader_list = [],[]


        # Crée une DataSet copie et y récupère la DataSet de Test Commune à tous les K-fold : 
        dataset_init = self.clean_dataset_get_tensor_and_train_valid_test_split(self.df,invalid_dates,args.train_prop,args.valid_prop,args.test_prop, normalize = False,dims_agg = dims)
        # On peut maintenant appeler dataset_init.U_test pour récupérer le test_set dans 'init', qu'il faut maintenant Normaliser avec les min/max des Train DataSet de chaque fold. 
        # ................................................................................


        # ANCIENNE VERSION A RETIRER : 
        # dataset_init.Dataset_save_folder = Dataset_get_save_folder(args,K_fold = 1,fold=0,netmob=netmob)
        # _,_,_,_ = dataset_init.split_normalize_load_feature_vect(invalid_dates,args.train_prop, args.valid_prop,args.test_prop)
        # dict_dataloader = dataset_init.get_dataloader()
        # ==========================================================================================
        # ==========================================================================================


        # Fait la 'Hold-Out' séparation, pour enlever les dernier mois de TesT
        df_hold_out = self.df[: dataset_init.first_test_date]  

        # ================================================================================================================================================================
        for k in range(args.K_fold): 
            self.fold_dataset_limits[k]['test']['timestamp'] = (dataset_init.first_test_date,dataset_init.last_test_date)
        # ================================================================================================================================================================


        # Récupère la Taille de cette DataFrame
        n = len(df_hold_out)

        # Adapt Valid and Train Prop (cause we want Test_prop = 0)
        valid_prop_tmps = args.valid_prop/(args.train_prop+args.valid_prop)
        train_prop_tmps = args.train_prop/(args.train_prop+args.valid_prop)
        
        # Découpe la dataframe en K_fold 
        for k in range(args.K_fold):
            # Slicing 
            if args.validation == 'wierd_blocked':
                l_lim_fold = int((k/args.K_fold)*n)
                u_lim_fold = int(((k+1)/args.K_fold)*n)

                df_tmps = df_hold_out[l_lim_fold:u_lim_fold]


            if args.validation == 'sliding_window':
                width_dataset = int(n/(1+(args.K_fold-1)*valid_prop_tmps))   # Stay constant. W = N/(1 + (K-1)*Pv/(Pv+Pt))
                l_lim_pos = int(k*valid_prop_tmps*width_dataset)    # Shifting of (valid_prop/train_prop)% of the width of the window, at each iteration 
                

                if k == args.K_fold - 1:
                    u_lim_pos = n
                else:
                    u_lim_pos = l_lim_pos + width_dataset

                df_tmps = df_hold_out[l_lim_pos:u_lim_pos]          

            # ================================================================================================================================================================
            self.fold_dataset_limits[k]['fold_limits']['df_indices'] = (l_lim_pos,u_lim_pos)
            self.fold_dataset_limits[k]['fold_limits']['timestamp'] = (df_hold_out.index[l_lim_pos],df_hold_out.index[u_lim_pos])
            # ================================================================================================================================================================         

            # On crée une DataSet à partir de df_tmps, qui a toujours la même taille, et toute les df_temps concaténée recouvre Valid Prop + Train Prop, mais pas Test Prop 
            dataset_tmps = DataSet(df_tmps, Weeks = self.Weeks, Days = self.Days, historical_len= self.historical_len,
                                   step_ahead=self.step_ahead,time_step_per_hour=self.time_step_per_hour)
            dataset_tmps.Dataset_save_folder = Dataset_get_save_folder(args,fold=k,netmob=netmob)


            time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding = dataset_tmps.split_normalize_load_feature_vect(invalid_dates,
                                                                                                                              train_prop = train_prop_tmps,
                                                                                                                              valid_prop = valid_prop_tmps,
                                                                                                                              test_prop =  0,
                                                                                                                              dims_agg= dims)
            dict_dataloader = dataset_tmps.get_dataloader()

            dict_dataloader['test'] = data_loader_with_test['test']


            # ================ Set Every Test-related information thank to dataset_init ================
            dataset_tmps.U_test, dataset_tmps.Utarget_test, dataset_tmps.time_slots_test, = dataset_init.U_test, dataset_init.Utarget_test, dataset_init.time_slots_test
            dataset_tmps.first_predicted_test_date,dataset_tmps.last_predicted_test_date = dataset_init.first_predicted_test_date,dataset_init.last_predicted_test_date
            dataset_tmps.first_test_date,dataset_tmps.last_test_date = dataset_init.first_test_date,dataset_init.last_test_date
            dataset_tmps.df_verif_test = dataset_init.df_verif_test
            #dataset_tmps.df_test = dataset_init.df_test
             # ================ ........................................................ ================


            Datasets.append(dataset_tmps)
            DataLoader_list.append(dict_dataloader)
            


        return(Datasets,DataLoader_list,time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding)
    
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

    def train_valid_test_split_indices(self,train_prop,valid_prop,test_prop):
        # Split with iterative method 
        if hasattr(self,'Dataset_save_folder'):
            split_path = f"{self.Dataset_save_folder}split_limits.pkl" 
        else:
            split_path = ''
        if split_path and (os.path.exists(split_path)):   #not empty & path exist
            try:
                split_limits = read_object(split_path)
            except:
                split_limits= train_valid_test_split_iterative_method(self,self.df_verif,train_prop,valid_prop,test_prop)
                save_object(split_limits, split_path)
                print(f"split_limits.pkl has never been saved or issue with last .pkl save")
        else : 
            split_limits= train_valid_test_split_iterative_method(self,self.df_verif,train_prop,valid_prop,test_prop)
            if split_path: save_object(split_limits, split_path)  #if not empty, save it 

        first_predicted_train_date= split_limits['first_predicted_train_date']
        last_predicted_train_date = split_limits['last_predicted_train_date']
        first_predicted_valid_date = split_limits['first_predicted_valid_date']
        last_predicted_valid_date = split_limits['last_predicted_valid_date']
        first_predicted_test_date = split_limits['first_predicted_test_date']
        last_predicted_test_date = split_limits['last_predicted_test_date']

        # Keep track on predicted limits (dates): 
        self.first_predicted_train_date,self.last_predicted_train_date = first_predicted_train_date,last_predicted_train_date
        self.first_predicted_valid_date,self.last_predicted_valid_date = first_predicted_valid_date,last_predicted_valid_date
        self.first_predicted_test_date,self.last_predicted_test_date = first_predicted_test_date,last_predicted_test_date

        # Keep track on DataFrame Verif:
        predicted_dates = self.df_verif.iloc[:,-1]
        self.df_verif_train = self.df_verif[( predicted_dates >= self.first_predicted_train_date) & (predicted_dates < self.last_predicted_train_date)]
        self.df_verif_valid = self.df_verif[(predicted_dates >= self.first_predicted_valid_date) & (predicted_dates < self.last_predicted_valid_date)] if self.last_predicted_valid_date is not None else None
        self.df_verif_test = self.df_verif[(predicted_dates >= self.first_predicted_test_date) & (predicted_dates < self.last_predicted_test_date)]  if self.last_predicted_test_date is not None else None

        # Keep track on DataFrame Limits (dates): 
        self.first_train_date,self.last_train_date = self.df_verif_train.iat[0,0] ,self.df_verif_train.iat[-1,-1]
        if valid_prop > 1e-3 :
            self.first_valid_date,self.last_valid_date = self.df_verif_valid.iat[0,0] ,self.df_verif_valid.iat[-1,-1]
        else:
            self.first_valid_date,self.last_valid_date = None, None

        if test_prop > 1e-3 :
            self.first_test_date,self.last_test_date = self.df_verif_test.iat[0,0] ,self.df_verif_test.iat[-1,-1]
        else:
            self.first_test_date,self.last_test_date = None, None
        
        # Get All the involved dates and keep track on splitted DataFrame 
        #self.df_train = self.df.reindex(self.df_verif_train.stack().unique())
        #self.df_valid = self.df.reindex(self.df_verif_valid.stack().unique()) if valid_prop > 1e-3 else None
        #self.df_test = self.df.reindex(self.df_verif_test.stack().unique()) if test_prop > 1e-3 else None

        # ====== Keep track on data (unique) used for train,valid and test, so that we can compute MinMax-Normalization
        reindex_train = self.df_verif_train.stack().unique()
        indices = self.df_dates[self.df_dates['date'].isin(reindex_train)].index.tolist()
        self.train_input = self.raw_values[indices]

        if valid_prop > 1e-3 :
            reindex_valid = self.df_verif_valid.stack().unique()
            indices = self.df_dates[self.df_dates['date'].isin(reindex_valid)].index.tolist()
            self.valid_input = self.raw_values[indices]
        else :
            self.valid_input = None
        
        if test_prop > 1e-3 :
            reindex_test = self.df_verif_test.stack().unique()
            indices = self.df_dates[self.df_dates['date'].isin(reindex_test)].index.tolist()
            self.test_input = self.raw_values[indices]
        else :
            self.test_input = None
        # =======


        # Get all the limits for U / Utarget split : 
        self.first_train_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.first_predicted_train_date].index[0])
        self.last_train_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.last_predicted_train_date].index[0])

        self.first_valid_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.first_predicted_valid_date].index[0]) if valid_prop > 1e-3 else None
        self.last_valid_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.last_predicted_valid_date].index[0]) if valid_prop > 1e-3 else None

        self.first_test_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.first_predicted_test_date].index[0]) if test_prop > 1e-3 else None
        self.last_test_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.last_predicted_test_date].index[0]) if test_prop > 1e-3 else None

    def get_dataloader(self):
        #   DataLoader 
        #DictDataLoader_object = DictDataLoader(self,args)
        #dict_dataloader = DictDataLoader_object.get_dictdataloader(args.batch_size)

        # Train, Valid, Test split : 
        train_tuple =  self.U_train,self.Utarget_train,{getattr(self,f"{name}_train") for name in self.contextual_tensors.keys()} # subway_X[train_subset],subway_Y[train_subset], dict(netmob = netmob[train_subset], calendar = calendar[train_subset])
        valid_tuple =  self.U_valid,self.Utarget_valid,{getattr(self,f"{name}_valid") for name in self.contextual_tensors.keys()}  # subway_X[valid_subset],subway_Y[valid_subset], dict(netmob = netmob[valid_subset], calendar = calendar[valid_subset])
        test_tuple =  self.U_test,self.Utarget_test,{getattr(self,f"{name}_test") for name in self.contextual_tensors.keys()}   # subway_X[test_subset],subway_Y[test_subset], dict(netmob = netmob[test_subset], calendar = calendar[test_subset])

        # Load DictDataLoader: 
        DictDataLoader_object = DictDataLoader(train_tuple, valid_tuple, test_tuple,self.args)
        dict_dataloader = DictDataLoader_object.get_dictdataloader()
        
        # =============== Ajout ===============
        if 'train' in dict_dataloader.keys(): self.train_loader = dict_dataloader['train']
        if 'validate' in dict_dataloader.keys(): self.valid_loader = dict_dataloader['validate']
        if 'test' in dict_dataloader.keys(): self.test_loader = dict_dataloader['test']
        if 'cal' in dict_dataloader.keys(): self.cal_loader = dict_dataloader['cal']
        # =============== Ajout ===============
        return(dict_dataloader)


    def split_normalize_load_feature_vect(self,invalid_dates,train_prop,valid_prop,test_prop,dims_agg
                                          #,calib_prop,batch_size,calendar_class
                                          ):
        self.get_shift_from_first_elmt()   # get shift indice and shift date from the first element / between each dataset 
        self.get_feature_vect(invalid_dates)  # Removed the forbidden dates, Build 'df_verif' and the Feature Vector Tensor masked by the forbidden indices.

        # Get Index to Split df, U, Utarget, time_slots_labels
        self.train_valid_test_split_indices(train_prop,valid_prop,test_prop)  # Create df_train,df_valid,df_test, df_verif_train, df_verif_valid, df_verif_test, and dates limits for each df and each tensor U

        # Get all the splitted train/valid/test input tensors. Normalize Them 
        self.split_tensors(normalize = True, dims_agg =dims_agg)

        # ================ FAIRE QULEQUE CHOSE POUR LE TIME-SLOTS LABELS. ESSAYER DE LES INTEGRER DANS LE CONTEXTUAL TENSORS  ================
        #
        # get Associated time_slots_labels (from df_verif)
        time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding = get_time_slots_labels(self)
        #
        #
        # ================ ................................................................................................ ================

        return(time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding)


    def set_train_valid_test_tensor_attribute(self,name,tensor,dims_agg,ref_for_normalization = None, normalize = False):
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
        
        if normalize : 
            # Get stats from the reference dataset, which mean dimension - 1 with respect to the neural network input 
            ref_tensor_df = TensorDataset(ref_for_normalization)
            ref_tensor_df = ref_tensor_df.normalize_tensor(ref_tensor_df.tensor,dims = dims_agg, minmaxnorm = True, standardize = False,reverse=False,feature_vect = False)
            mini, maxi, mean, std = ref_tensor_df.mini,ref_tensor_df.maxi,ref_tensor_df.mean,ref_tensor_df.std  # 

        else : 
            mini, maxi, mean, std = None, None, None, None

        splitter = TrainValidTest_Split_Normalize(tensor,dims_agg,
                                    first_train = self.first_train_U, last_train= self.last_train_U,
                                    first_valid= self.first_valid_U, last_valid = self.last_valid_U,
                                    first_test = self.first_test_U, last_test = self.last_test_U,
                                    minmaxnorm = True,standardize = False)


        train_tensor_ds,valid_tensor_ds,test_tensor_ds = splitter.load_normalize_tensor_datasets(mini = mini, maxi = maxi, mean = mean, std = std, normalize=normalize,feature_vect=True)
        setattr(self,f"{name}_train", train_tensor_ds.tensor)
        setattr(self,f"{name}_valid", valid_tensor_ds.tensor)
        setattr(self,f"{name}_test", test_tensor_ds.tensor)
        # ....

    def display_info_on_inputs(self):
        print('U size: ',self.U.size(),'Utarget size: ',self.Utarget.size())
        print('U_train size: ',self.U_train.size(),'Utarget_train size: ',self.Utarget_train.size())
        if hasattr(self,'U_valid'): print('U_valid size: ',self.U_valid.size(),'Utarget_valid size: ',self.Utarget_valid.size()) 
        if hasattr(self,'U_test'): print('U_test size: ', self.U_test.size(),'Utarget_test size: ',self.Utarget_test.size())

        print('U_train min: ',self.U_train.min(), 'U_train max: ',self.Utarget_train.max())
        if hasattr(self,'U_valid'): print('U_valid min: ',self.U_valid.min(),'U_valid max: ',self.Utarget_valid.max())
        if hasattr(self,'U_test'): print('U_test min: ',self.U_test.min(),'U_test max: ',self.Utarget_test.max())

    def split_tensors(self,normalize, dims_agg):
        ''' Split input tensors  in Train/Valid/Test part '''
        # Get U_train, U_valid, U_test
        self.set_train_valid_test_tensor_attribute('U',self.U,dims_agg=dims_agg,ref_for_normalization = self.train_input, normalize = normalize)

        # Get Utarget_train, Utarget_valid, Utarget_test 
        self.set_train_valid_test_tensor_attribute('Utarget',self.Utarget,dims_agg=dims_agg,ref_for_normalization = self.train_input, normalize = normalize)

        self.display_info_on_inputs()

        # Get NetMob_train, NetMob_valid, NetMob_test, Weather_train etc etc ...
        if hasattr(self,'contextual_tensors'):
            for name, tensor_dict in self.contextual_tensors.items():
                feature_vect = tensor_dict['feature_vect']
                dims = tensor_dict['dims']
                raw_data = tensor_dict['raw_data']
                self.set_train_valid_test_tensor_attribute(name,feature_vect,dims,raw_data, normalize = normalize)
        else:
            print('No Contextual Data has been considered')

        #if self.time_slots_labels is not None : 
        #    self.time_slots_train = {calendar_class: self.time_slots_labels[calendar_class][self.first_train_U:self.last_train_U] for calendar_class in range(len(self.nb_class)) }
        #    self.time_slots_valid = {calendar_class: self.time_slots_labels[calendar_class][self.first_valid_U:self.last_valid_U] if self.first_valid_U is not None else None for calendar_class in range(len(self.nb_class))}
        #    self.time_slots_test = {calendar_class: self.time_slots_labels[calendar_class][self.first_test_U:self.last_test_U] if self.first_test_U is not None else None for calendar_class in range(len(self.nb_class)) }




class TimeSerie(object):
    def __init__(self,ts,init_ts = None,mini = None, maxi = None, mean = None, normalized = False):
        self.length = len(ts)
        self.ts = ts
        self.normalized = normalized
        if mini is not None: 
            self.mini = mini
        else : 
            self.mini = ts.min()
        if maxi is not None: 
            self.maxi = maxi
        else : 
            self.maxi = ts.max()
        if mean is not None:
            self.mean = mean
        else:
            self.mean = ts.mean()
        if init_ts is not None:
            self.init_ts = init_ts
        else:
            self.init_ts = ts
        
    def normalize(self):
        if self.normalized:
            print('The TimeSerie might be already normalized')
        minmaxnorm = lambda x : (x-self.mini)/(self.maxi-self.mini)
        return(minmaxnorm(self.init_ts))
    
    def unormalize(self):
        if not(self.normalized):
            print('The TimeSerie might be already unnormalized')
        return(self.ts*(self.maxi - self.mini)+self.mini)