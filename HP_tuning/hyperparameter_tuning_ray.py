import torch
import ray 
from ray import tune 

# Relative path:
import sys 
import os 
import json
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal imports: 
from trainer import Trainer, report
from HP_tuning.ray_search_space import get_search_space_ray 
from HP_tuning.ray_config import get_ray_config
from utils.utilities_DL import get_loss,load_model_and_optimizer
from utils.save_results import get_date_id,load_json_file,update_json


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def HP_modification(config,args):
    '''Update the hyperparameters'''
    forbidden_keys = ['batch_size','train_prop','valid_prop','test_prop']
    for key, value in config.items():
        if key in forbidden_keys:
            raise ValueError(f"Key {key} cant' be modified while loading trainer for HP-tuning cause it has also impact on dataloader which is already defined")
        else:
            if key == 'scheduler':
                if config['scheduler']['scheduler']:
                    for args_scheduler in ['torch_scheduler_milestone','torch_scheduler_gamma','torch_scheduler_lr_start_factor']:
                         setattr(args, args_scheduler, config['scheduler'][args_scheduler])

            elif hasattr(args, key):
                setattr(args, key, value)
            elif 'vision_' in key:
                key = key.replace('vision_', '')
                setattr(args.args_vision,key,value)
            else: 
                raise ValueError(f"Key {key} issue")
    return(args)

def load_trainer(config, dataset, args, dic_class2rpz):
    '''Change the hyperparameters and load the model accordingly. 
    The hyperparameters to be modified must not concern the dataloader, so don't change: 
    - train/valid/test/calib proportion
    - batch-size
    '''
    args = HP_modification(config,args)

    loss_function = get_loss(args)
    model,optimizer,scheduler = load_model_and_optimizer(args,dic_class2rpz)
    #model_ref = ray.put(model)
    
    trainer = Trainer(dataset,model,
                    args,optimizer,loss_function,scheduler = scheduler,
                    dic_class2rpz=dic_class2rpz)
    return(trainer)

def HP_tuning(dataset,args,num_samples,dic_class2rpz,working_dir = '/home/rrochas/prediction_validation/',save_dir = 'save/HyperparameterTuning/'): 
    # Load ray parameters:
    config = get_search_space_ray(args)
    ray_scheduler, ray_search_alg, resources_per_trial, num_gpus, max_concurrent_trials, num_cpus = get_ray_config(args)
    
    # Init Ray
    if ray.is_initialized:
        ray.shutdown()
    ray.init(runtime_env={'working_dir': working_dir,
                         'excludes': [f'{working_dir}/.git/',  # Exclude .git folder
                                    f'{working_dir}/__pycache__/',  # Exclude python cache
                                    f'{working_dir}/save/',  #Exclude save folder 
                                    f'{working_dir}/data/',  #Exclude data folder 
                                    '/home/rrochas/prediction_validation/.git/objects/6a/2f986b4cfd0d5c1b5370539c60cbc60376ee7c',
                                    '/home/rrochas/prediction_validation/.git/objects/62/340d41856c322da363f152c93b2cc7ca7e1b52',
                                    '/home/rrochas/prediction_validation/.git/objects/3f/2332cfa867063c2cfbf14628f0745c1d23aa85',
                                    '/home/rrochas/prediction_validation/.git/objects/b8/b89fb56b6e4740dc2d54abe53a4e1c9d85b47d'
                                    ]
                            },
             num_gpus=num_gpus,
             num_cpus=num_cpus
            )
    
    # Put large objects into the Ray object store
    dataset_ref = ray.put(dataset) # Put dataset (large object) in a 'Ray Object Store'. Which mean a worker won't serealize it but access to a shared memory where dataset_ref is located for everyone.
    #dataset_ref = dataset


    def train_with_tuner(config):
        # Dereference the large objects within the worker
        dataset = ray.get(dataset_ref)
        trainer = load_trainer(config, dataset, args, dic_class2rpz)
        trainer.train_and_valid()  # No plotting, No testing

        # Clean Memory: 
        torch.cuda.empty_cache()
        del trainer 
        del dataset
        # gc.collect()  # Clean CPU memory
        # ...

    
    analysis = tune.run(
            lambda config: train_with_tuner(config),
            config=config,
            num_samples=num_samples,  # Increase num_samples for more random combinations
            resources_per_trial = resources_per_trial,
            max_concurrent_trials = max_concurrent_trials,
            scheduler = ray_scheduler,
            search_alg = ray_search_alg,
            fail_fast = False, # Continue even with errors 
            raise_on_failed_trial=False,
        )
    
    # Get Trial ID (Name of the entire HP-tuning)
    date_id = get_date_id()
    datasets_names = '_'.join(args.dataset_names)
    model_names = '_'.join([args.model_name,args.args_vision.model_name]) if hasattr(args.args_vision,'model_name')  else args.model_name
    trial_id =  f"{datasets_names}_{model_names}_{args.loss_function_type}Loss_{date_id}"

    # Keep track only on successfull trials:
    analysis.results_df.to_csv(f'{working_dir}/{save_dir}/{trial_id}.csv')
    

    # Keep track on other args:
    json_file = load_json_file(f'{working_dir}/{save_dir}')
    update_json(args,json_file,trial_id,performance={},save_dir=f'{working_dir}/{save_dir}')

    return(analysis,trial_id)

if __name__ == '__main__': 
    from constants.config import get_args,update_modif
    from constants.paths import FOLDER_PATH,FILE_NAME
    from utils.utilities_DL import match_period_coverage_with_netmob,get_small_ds
    from K_fold_validation.K_fold_validation import KFoldSplitter

    # Set working directory (the one where you will find folder 'save/' or 'HP_tuning'):
    current_path = notebook_dir = os.getcwd()
    working_dir = os.path.abspath(os.path.join(current_path, '..'))
    save_dir = 'save/HyperparameterTuning'

    # Load config
    model_name = 'STGCN' #'CNN'
    args = get_args(model_name)

    # Modification : 
    args.K_fold = 5
    args.ray = True
    args.W = 0  # IMPORTANT AVEC NETMOB
    args.epochs = 10
    args.loss_function_type = 'MSE' # 'quantile'

    args = update_modif(args)

    # Coverage Period : 
    small_ds = False
    coverage = match_period_coverage_with_netmob(FILE_NAME,dataset_names=['subway_in','netmob'])
    (coverage,args) = get_small_ds(small_ds,coverage,args)

    # Choose DataSet and VisionModel if needed: 
    dataset_names = ['subway_in'] # ['calendar','netmob'] #['subway_in','netmob','calendar']
    vision_model_name = 'ImageAvgPooling'  # 'ImageAvgPooling'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',


    # Load K-fold subway-ds 
    folds = [0] # Here we use the first fold for HP-tuning. 

    # In case we need to compute the Sliding K-fold validation:
    # folds = np.arange(1,args.K_fold)

    K_fold_splitter = KFoldSplitter(dataset_names,args,coverage,vision_model_name,folds)
    K_subway_ds,dic_class2rpz = K_fold_splitter.split_k_fold()

    num_samples = 8
    subway_ds = K_subway_ds[0]
    analysis,trial_id = HP_tuning(subway_ds,args,num_samples,dic_class2rpz,working_dir,save_dir)