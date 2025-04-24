from dl_models.full_model import full_model
from plotting.plotting_bokeh import plot_bokeh
from trainer import Trainer
from utils.utilities_DL import choose_optimizer, load_scheduler,get_loss
from K_fold_validation.K_fold_validation import KFoldSplitter

# =======================================================================================================================
# =======================================================================================================================

def load_optimizer_and_scheduler(model,args):
    optimizer = choose_optimizer(model,args)
    scheduler = load_scheduler(optimizer,args)
    loss_function = get_loss(args)
    return(optimizer,scheduler,loss_function)


def load_everything(args):
    # Load DataSet, DataLoader, Args :
    folds = [0]
    K_fold_splitter = KFoldSplitter(args,folds)
    K_subway_ds,args = K_fold_splitter.split_k_fold()
    subway_ds = K_subway_ds[0]
    # ...

    # Load Model:
    model = full_model(subway_ds, args).to(args.device)
    print('number of total parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print('number of trainable parameters: {}'.format(sum([p.numel() for p in model.parameters() if p.requires_grad])))
    
    # Load Optimizer, Scheduler, Loss function: 
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    
    return(model,subway_ds,loss_function,optimizer,scheduler,args)


def evaluate_config(dataset_names,FOLDER_PATH,FILE_NAME,args,coverage,vision_model_name,mod_plot):
    # Load Model, Optimizer, Scheduler:

    model,subway_ds,loss_function,optimizer,scheduler,args = load_everything(args)
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