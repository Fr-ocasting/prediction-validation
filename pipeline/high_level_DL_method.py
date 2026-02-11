# GET PARAMETERS
import os 
import sys
import torch 
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.Flex_MDI.Flex_MDI import full_model
from pipeline.plotting.plotting_bokeh import plot_bokeh
from pipeline.trainer import Trainer
from pipeline.utils.utilities_DL import choose_optimizer, load_scheduler,get_loss
from pipeline.K_fold_validation.K_fold_validation import KFoldSplitter
from torchinfo import summary
from pipeline.profiler.profiler import model_memory_cost
from examples.train_and_visu_non_recurrent import get_ds
from pipeline.high_level_DL_method import load_optimizer_and_scheduler
from pipeline.Flex_MDI.Flex_MDI import full_model
from pipeline.trainer import Trainer
from pipeline.ML_trainer import ML_trainer


def model_loading_and_training(fold_to_evaluate,
                               save_folder,
                               args_init,
                               modification: dict = {},
                               trial_id: str = None,
                               ):
    """ 
    Load and train the model from a specific configuration defined by args_init and modification
    Args:
        fold_to_evaluate (list): list of fold to evaluate. 
            If only the complete dataset should be evaluated, set fold_to_evaluate = [args_init.K_fold-1]
        save_folder (str): folder where to save the model and the results
        args_init (Namespace): Namespace containing the initial arguments before modification.
        modification (dict): dictionary containing the modifications to apply to args_init.
        trial_id (str): id of the trial, used for  saving the model.

        Returns:
        trainer (Trainer): Trainer object containing the trained model and the training history.
        ds (Dataset): Dataset object containing the data used for training and validation.
        model (nn.Module): trained model.
        args (Namespace): Namespace updated with information from training
    """
    trainer,ds,model,args = load_init_model_trainer_ds(fold_to_evaluate,save_folder,args_init,modification,trial_id)
    trainer.train_and_valid(normalizer = ds.normalizer, mod = 1000,mod_plot = None,unormalize_loss = args.unormalize_loss) 
    return trainer,ds,model,args

def load_init_model_trainer_ds(fold_to_evaluate,save_folder,args_init,modification,trial_id):
    ds,args,_,_,_ = get_ds(modification=modification,args_init=args_init,fold_to_evaluate=fold_to_evaluate)
    print('Loaded dataset with args:')
    for key,value in vars(args).items():
        print(f"{key}: {value}")

    if args.model_name in ['SARIMAX','XGBoost']:
        trainer,model = ML_trainer(ds,args)
    else:
        model = full_model(ds, args).to(args.device)
        optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
        if len(fold_to_evaluate) == 1: 
            fold = fold_to_evaluate[0]
        else:
            raise ValueError("fold_to_evaluate should contain only one fold cause only one training will be done here.")
        

        trainer = Trainer(ds,model,args,optimizer,loss_function,
                          scheduler = scheduler,
                          show_figure = False,
                          trial_id = trial_id, 
                          fold=fold,
                          save_folder = save_folder)
    return trainer,ds,model,args



def load_optimizer_and_scheduler(model,args):
    optimizer = choose_optimizer(model,args)
    scheduler = load_scheduler(optimizer,args)
    loss_function = get_loss(args)
    return(optimizer,scheduler,loss_function)


def load_everything(args,folds=None):
    # Load DataSet, DataLoader, Args :
    if folds is None:
        folds = [0]
    K_fold_splitter = KFoldSplitter(args,folds)
    K_subway_ds,args = K_fold_splitter.split_k_fold()
    subway_ds = K_subway_ds[0]
    # ...

    # Load Model:
    model = full_model(subway_ds, args).to(args.device)
    print('number of total parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print('number of trainable parameters: {}'.format(sum([p.numel() for p in model.parameters() if p.requires_grad])))
    model_memory_cost(model)
    summary(model)
    
    # Load Optimizer, Scheduler, Loss function: 
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    
    return(model,subway_ds,loss_function,optimizer,scheduler,args)


def evaluate_config(args,mod_plot=None,folds_to_evaluate=None):
    # Load Model, Optimizer, Scheduler:

    model,subway_ds,loss_function,optimizer,scheduler,args = load_everything(args,folds_to_evaluate)
    normalizer = subway_ds.normalizer
    df_verif_test = subway_ds.tensor_limits_keeper.df_verif_test
    # Load trainer: 
    trainer = Trainer(subway_ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = True)# Ajoute dans trainer, if calibration_prop is not None .... et on modifie le dataloader en ajoutant un clabration set
    # Train Model 
    trainer.train_and_valid(normalizer = normalizer,mod = 1000,mod_plot = mod_plot)  # Récupère les conformity scores sur I1, avec les estimations faites precedemment 

    pi,pi_cqr = plot_bokeh(trainer,normalizer,df_verif_test,args)
    return(trainer,model,normalizer,df_verif_test,args,pi,pi_cqr)

# =======================================================================================================================
# =======================================================================================================================