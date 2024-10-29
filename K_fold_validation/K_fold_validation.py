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

from build_inputs.load_preprocessed_dataset import load_complete_ds
from dataset import TensorDataset
from constants.paths import FOLDER_PATH, FILE_NAME


class KFoldSplitter(object):
    def __init__(self,dataset_names,args,coverage,vision_model_name,folds):
        super(KFoldSplitter,self).__init__()
        self.dataset_names = dataset_names
        self.args = args
        self.coverage = coverage
        self.FOLDER_PATH = FOLDER_PATH
        self.FILE_NAME = FILE_NAME
        self.vision_model_name = vision_model_name
        self.folds = folds

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
        U_test_tmps = TensorDataset(subway_ds.U_test, normalized = False, normalizer=subway_ds_tmps.normalizer)
        U_test_tmps.normalize(feature_vect = True)

        Utarget_test_tmps = TensorDataset(subway_ds.Utarget_test, normalized = False, normalizer=subway_ds_tmps.normalizer)
        Utarget_test_tmps.normalize(feature_vect = True)

        subway_ds_tmps.U_test = U_test_tmps.tensor
        subway_ds_tmps.Utarget_test = Utarget_test_tmps.tensor
        # ...
        return (subway_ds_tmps)

    def add_contextual_U_test(self,subway_ds_tmps,subway_ds,NetMob_ds_tmps):
        ''' Tackle contextual Test vector:''' 
        for name in subway_ds.contextual_tensors.keys():

            if name == 'netmob':
                U_context_tmps = TensorDataset(subway_ds.contextual_tensors[name]['test'], normalized = False, normalizer=NetMob_ds_tmps.normalizer)
                U_context_tmps.normalize(feature_vect = True)
                subway_ds_tmps.contextual_tensors[name]['test'] = U_context_tmps.tensor
            elif 'calendar' in name:
                subway_ds_tmps.contextual_tensors[name]['test'] = subway_ds.contextual_tensors[name]['test'] 
            else:
                raise NotImplementedError(f'contextual data {name} has not been implemented')

        return(subway_ds_tmps)
    
    def load_init_ds(self,normalize = False):
        subway_ds,NetMob_ds,args,dic_class2rpz = load_complete_ds(self.dataset_names,self.args,self.coverage,self.vision_model_name,normalize = normalize)
        return(subway_ds,NetMob_ds,args,dic_class2rpz)

    def split_k_fold(self):
        '''Split ds in K-fold with respected proportion train_prop/valid_prop and keep initial test dataset:
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
        '''

        # Init
        subway_ds,_,args,dic_class2rpz = self.load_init_ds(normalize = False)  # build a copy of 'args'
        K_subway_ds = []


        # Remove test-dates:
        first_test_date = subway_ds.tensor_limits_keeper.first_test_date
        
        
        # Adapt Valid and Train Prop (cause we want Test_prop = 0)
        train_prop_tmps = args.train_prop/(args.train_prop+args.valid_prop)   
        valid_prop_tmps = args.valid_prop/(args.train_prop+args.valid_prop)

        # Modify local 'args':
        args.train_prop = train_prop_tmps
        args.valid_prop = valid_prop_tmps
        args.test_prop = 0
        # ...


        #if (args.train_valid_test_split_method == 'similar_length_method') #or (args.train_valid_test_split_method == 'iterative_method') :
        df_verif = subway_ds.tensor_limits_keeper.df_verif 
        coverage_without_test = df_verif['t+0']
        coverage_without_test = coverage_without_test[coverage_without_test<first_test_date]  # Remove samples associated to test dataset
        fold_length = int(len(coverage_without_test)/(valid_prop_tmps*args.K_fold+(1-valid_prop_tmps)))

        '''
        elif: 
            coverage_without_test =  self.coverage[self.coverage<first_test_date]
            n = len(coverage_without_test)
        '''
        # ... 


        # Découpe la dataframe en K_fold 
        for k in self.folds:
            # Slicing 
            print(f'\nFold n°{k}')
            if args.validation == 'sliding_window':
                #if (args.train_valid_test_split_method == 'similar_length_method'):
                start_coverage = int(k*fold_length*valid_prop_tmps)

                if k == args.K_fold -1:
                    coverage_tmps = coverage_without_test[coverage_without_test>coverage_without_test.iloc[start_coverage]]           
                else:
                    coverage_tmps = coverage_without_test[(coverage_without_test>coverage_without_test.iloc[start_coverage]) &
                                                        (coverage_without_test<coverage_without_test.iloc[start_coverage+fold_length])
                                                        ]   
                '''
                else:
                    width_dataset = int(n/(1+(args.K_fold-1)*valid_prop_tmps))   # Stay constant. W = N/(1 + (K-1)*Pv/(Pv+Pt))
                    l_lim_pos = int(k*valid_prop_tmps*width_dataset)    # Shifting of (valid_prop/train_prop)% of the width of the window, at each iteration 
                    if k == args.K_fold - 1:
                        u_lim_pos = n
                    else:
                        u_lim_pos = l_lim_pos + width_dataset
                    
                    # Get df_tmps and coverage_tmps:
                    coverage_tmps = coverage_without_test[l_lim_pos:u_lim_pos] 
                    #time_slot_limits = np.arange(l_lim_pos,u_lim_pos)
                '''


            subway_ds_tmps,NetMob_ds_tmps,_,_ = load_complete_ds(self.dataset_names,args,coverage_tmps,self.vision_model_name, normalize = True)  # Normalize

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
        return K_subway_ds,dic_class2rpz,args