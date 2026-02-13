# Relative path:
import sys 
import os 
import numpy as np 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal imports: 

from pipeline.build_inputs.load_preprocessed_dataset import load_complete_ds
from pipeline.DataSet.splitter import subclassTensorDataset
from constants.paths import FOLDER_PATH
from argparse import Namespace

class KFoldSplitter(object):
    def __init__(self,args,folds):
        super(KFoldSplitter,self).__init__()
        self.args = args
        self.FOLDER_PATH = FOLDER_PATH
        self.folds = folds
        self.validation_split_method = args.validation_split_method

    def add_df_verif_test(self,subway_ds_tmps,subway_ds):
        subway_ds_tmps.tensor_limits_keeper.df_verif_test = subway_ds.tensor_limits_keeper.df_verif_test 
        subway_ds_tmps.tensor_limits_keeper.first_test_date = subway_ds.tensor_limits_keeper.first_test_date
        subway_ds_tmps.tensor_limits_keeper.last_test_date =  subway_ds.tensor_limits_keeper.last_test_date
        subway_ds_tmps.tensor_limits_keeper.first_predicted_test_date = subway_ds.tensor_limits_keeper.first_predicted_test_date
        subway_ds_tmps.tensor_limits_keeper.last_predicted_test_date = subway_ds.tensor_limits_keeper.last_predicted_test_date
        subway_ds_tmps.init_invalid_dates = subway_ds.invalid_dates
        return subway_ds_tmps

    def add_U_test_and_Utarget_test(self,subway_ds_tmps,subway_ds):
        ''' Tackle U_test and Utarget_test'''
        U_test_tmps = subclassTensorDataset(subway_ds.U_test, normalized = False, normalizer=subway_ds_tmps.normalizer)
        U_test_tmps.normalize(feature_vect = True)

        Utarget_test_tmps = subclassTensorDataset(subway_ds.Utarget_test, normalized = False, normalizer=subway_ds_tmps.normalizer)
        Utarget_test_tmps.normalize(feature_vect = True)

        subway_ds_tmps.U_test = U_test_tmps.tensor
        subway_ds_tmps.Utarget_test = Utarget_test_tmps.tensor
        # ...
        return (subway_ds_tmps)

    def add_contextual_U_test(self,subway_ds_tmps,subway_ds,NetMob_ds_tmps):
        ''' Tackle contextual Test vector:''' 
        for name in subway_ds.contextual_tensors.keys():
            if ('netmob' in name) or ('subway_out' in name) :  # name == 'netmob'
                if type(NetMob_ds_tmps) == list:
                    #k = [k for k,cod_trg in enumerate(LIST_COD_TRG) if cod_trg == name.split('_')[-1]][0]
                    k = [k for k,cod_trg in enumerate(subway_ds_tmps.spatial_unit) if cod_trg == name.split('_')[-1]][0]
                    normalizer = NetMob_ds_tmps[k].normalizer
                else:
                    normalizer = NetMob_ds_tmps.normalizer

                U_context_tmps = subclassTensorDataset(subway_ds.contextual_tensors[name]['test'], normalized = False, normalizer=normalizer)
                U_context_tmps.normalize(feature_vect = True)
                subway_ds_tmps.contextual_tensors[name]['test'] = U_context_tmps.tensor
            elif 'calendar' in name:
                subway_ds_tmps.contextual_tensors[name]['test'] = subway_ds.contextual_tensors[name]['test'] 
            else:
                raise NotImplementedError(f'contextual data {name} has not been implemented')

        return(subway_ds_tmps)
    
    def load_init_ds(self,normalize = False):
        if type(self.folds) == np.ndarray:
            self.folds = list(self.folds)
        if self.folds == [0]:
            print('---- No K-fold')
            target_ds,contextual_ds,args = load_complete_ds(self.args,normalize = normalize,k=0)
        else:
            target_ds,contextual_ds,args = load_complete_ds(self.args,normalize = normalize)  #,dic_class2rpz
        
        return(target_ds,contextual_ds,args) #,dic_class2rpz)

    def split_k_fold(self):
        if self.validation_split_method == 'custom_blocked_cv':
            return self.custom_blocked_cv()
        elif self.validation_split_method == 'forward_chaining_cv':
            return self.forward_chaining_cv()
        else:
            raise NotImplementedError


    def forward_chaining_cv(self):
        '''
        Implement validation following 'forward chaining cross validation' method. 
        Split ds with respected proportion train_prop/valid_prop and keep initial test dataset:       

        >>> Each fold contains a training dataset and a validation dataset. 
        >>> We use the validation dataset to determine the best training epoch (the point at which the model performs best). 
        >>> At this point, we save the weights and then perform a forward pass on the test dataset to collect the model’s metrics.

        >>> We consider the hyperparameter tuning already done, meaning the hyperparameters are fixed. 
        >>> The validation phase is used solely to evaluate the model’s performance through various metrics.


        # forward_chaining_cv: 
                         Train          Valid    Test 
        Fold 1     ==================== ======= =======
                               Train           Valid    Test 
        Fold 2     =========================== ======= =======
                                     Train          Valid    Test 
        Fold 3     ================================ ======= =======
                                           Train           Valid   Test 
        Fold 4     ====================================== ======= =======
                                                 Train          Valid    Test 
        Fold 5     ============================================ ======= =======
        # -------------------------------------------------------------------------------

        Limitations:
        >>> The first few folds are particularly small, leading to worse metrics and large variance that can significantly impact the overall average MSE.
        '''
        K_ds = []
        print(f'----------------------------------------')
        print(f'Loading the Complete Dataset for K-fold splitting')
        target_ds_init,_,args = self.load_init_ds(normalize = True)  # Load 'U' and 'U_target'. # Define already feature vect for the K-th fold with proportion train/valid/test.
        # Get Init Coverage Period
        df_verif_init = target_ds_init.tensor_limits_keeper.df_verif 
        nb_samples = len(df_verif_init)
        t_step_ahead = [c for c in df_verif_init.columns if 't+' in c][0]
        coverage_period_init = df_verif_init[t_step_ahead]

        # Get number of added samples for each next fold :
        if self.args.K_fold ==1:
            N1 = len(target_ds_init.tensor_limits_keeper.df_verif_train)
        else:
            N1 = len(target_ds_init.tensor_limits_keeper.df_verif_train)//(self.args.K_fold-1)
        N_valid = len(target_ds_init.tensor_limits_keeper.df_verif_valid)
        N_test = len(target_ds_init.tensor_limits_keeper.df_verif_test)

        
        for k in self.folds:
            print(f'\n----------------------------------------')
            print(f'Loading the dataset for fold n°{k}')
            if k == self.args.K_fold-1:
                print('  add last fold : the initil one')
                target_ds_init.init_invalid_dates = target_ds_init.invalid_dates
                K_ds.append(target_ds_init)   
            else:
                if k == 0:
                    new_nb_samples = nb_samples-int((self.args.K_fold-1)*N1*args.min_fold_size_proportion)
                else:
                    new_nb_samples = nb_samples-int((self.args.K_fold-k)*N1*args.min_fold_size_proportion)
                args_copy = Namespace(**vars(args))
                args_copy.set_spatial_units = target_ds_init.spatial_unit
                coverage_period_tmps =pd.date_range(df_verif_init.min().min(), df_verif_init.iloc[:new_nb_samples].max().max(), freq=f'{60 // target_ds_init.time_step_per_hour}min')# coverage_period_init.iloc[:new_nb_samples]
                # Modify local 'args':
                args_copy.train_prop = 1 - (N_valid+N_test)/new_nb_samples
                args_copy.valid_prop = N_valid/new_nb_samples
                args_copy.test_prop = N_test/new_nb_samples

                target_ds_tmps,_,_= load_complete_ds(args_copy,coverage_period=coverage_period_tmps,normalize = True,k=k)  # Normalize
                target_ds_tmps.init_invalid_dates = target_ds_init.invalid_dates

                K_ds.append(target_ds_tmps)
        return K_ds,args



    def custom_blocked_cv(self):
        '''
        Implement validation following 'blocked cross validation' method. 
        Split ds in K-fold with respected proportion train_prop/valid_prop and keep initial test dataset:
        >>> Each fold contains a training dataset and a validation dataset. 
        >>> We use the validation dataset to determine the best training epoch (the point at which the model performs best). 
        >>> At this point, we save the weights and then perform a forward pass on the test dataset to collect the model’s metrics.

        >>> We consider the hyperparameter tuning already done, meaning the hyperparameters are fixed. 
        >>> The validation phase is used solely to evaluate the model’s performance through various metrics.
        

        # Custom Sliding K-fold validation: 
                         Train Set          Valid Set                           Test Set
        Fold 1     ====================      =======                            =======
                               Train Set          Valid Set                     Test Set
        Fold 2           ====================      =======                      =======
                                     Train Set          Valid Set               Test Set
        Fold 3                 ====================      =======                =======
                                           Train Set          Valid Set         Test Set
        Fold 4                       ====================      =======          =======
                                                 Train Set          Valid Set   Test Set
        Fold 5                             ====================      =======    =======
        # -------------------------------------------------------------------------------


        Limitations:
        >>> The metrics often exhibit high variance because the training dataset is very small.
        >>> The performance metrics are generally poor. On a larger dataset (e.g., the complete dataset), one can easily achieve an MSE that is half as large.
        '''

        # Init
        subway_ds,_,args = self.load_init_ds(normalize = False)  # build a copy of 'args'
        K_subway_ds = []

        # Remove test-dates:
        first_test_date = subway_ds.tensor_limits_keeper.first_test_date

        
        # Adapt Valid and Train Prop (cause we want Test_prop = 0)
        train_prop_tmps = args.train_prop/(args.train_prop+args.valid_prop)   
        valid_prop_tmps = args.valid_prop/(args.train_prop+args.valid_prop)



        # Modify local 'args':
        args_copy = Namespace(**vars(args))
        args_copy.train_prop = train_prop_tmps
        args_copy.valid_prop = valid_prop_tmps
        args_copy.test_prop = 0
        # ...


        #if (args.train_valid_test_split_method == 'similar_length_method') #or (args.train_valid_test_split_method == 'iterative_method') :
        df_verif = subway_ds.tensor_limits_keeper.df_verif 
        t_step_ahead = [c for c in df_verif.columns if 't+' in c][0]
        coverage_without_test = df_verif[t_step_ahead]
        coverage_without_test = coverage_without_test[coverage_without_test<first_test_date]  # Remove samples associated to test dataset
        fold_length = int(len(coverage_without_test)/(valid_prop_tmps*args.K_fold+(1-valid_prop_tmps)))
        # ... 


        # Découpe la dataframe en K_fold 
        for k in self.folds:
            # Slicing 
            print(f'----------------------------------------')
            print(f'Fold n°{k}')

            #if (args.train_valid_test_split_method == 'similar_length_method'):
            start_coverage = int(k*fold_length*valid_prop_tmps)

            if k == args.K_fold -1:
                coverage_tmps = coverage_without_test[coverage_without_test>coverage_without_test.iloc[start_coverage]]           
            else:
                coverage_tmps = coverage_without_test[(coverage_without_test>coverage_without_test.iloc[start_coverage]) &
                                                    (coverage_without_test<coverage_without_test.iloc[start_coverage+fold_length])
                                                    ]   

            subway_ds_tmps,NetMob_ds_tmps,_= load_complete_ds(args_copy,
                                                                 coverage_period=coverage_tmps,
                                                                 normalize = True)  # Normalize

            # Tackle U_test and Utarget_test (normalize U_test with normalizer from subway_ds_TMPS):
            subway_ds_tmps = self.add_U_test_and_Utarget_test(subway_ds_tmps,subway_ds)

            # Tackle contextual Test vector (normalize U_test_contextual with normalizer from NetMob_ds_TMPS):
            subway_ds_tmps = self.add_contextual_U_test(subway_ds_tmps,subway_ds,NetMob_ds_tmps)

            # Add df_verif_test : 
            subway_ds_tmps = self.add_df_verif_test(subway_ds_tmps,subway_ds)

            # Tackle dataloader: load dataloader again (call it 2 times: within 'load_complete_ds', and then here. It's longer but easier to implement)
            subway_ds_tmps.get_dataloader()
            # ...

            K_subway_ds.append(subway_ds_tmps)
        return K_subway_ds,args