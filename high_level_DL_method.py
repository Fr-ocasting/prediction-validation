from build_inputs.load_preprocessed_dataset import load_complete_ds
from dl_models.full_model import full_model
from plotting.plotting_bokeh import plot_bokeh
from trainer import Trainer
from utils.utilities_DL import choose_optimizer, load_scheduler,get_loss


# =======================================================================================================================
# =======================================================================================================================
def load_model(args,dic_class2rpz):
    model = full_model(args,dic_class2rpz).to(args.device)
    print('number of total parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print('number of trainable parameters: {}'.format(sum([p.numel() for p in model.parameters() if p.requires_grad])))
    return(model)


def load_optimizer_and_scheduler(model,args):
    optimizer = choose_optimizer(model,args)
    scheduler = load_scheduler(optimizer,args)
    loss_function = get_loss(args.loss_function_type,args)
    return(optimizer,scheduler,loss_function)


def load_everything(dataset_names,folder_path,file_name,args,coverage,vision_model_name):

    # Load DataSet, DataLoader, Args :
    (subway_ds,NetMob_ds,args,dic_class2rpz) = load_complete_ds(dataset_names,args,coverage,folder_path,file_name,vision_model_name)

    # Load Model:
    model = load_model(args,dic_class2rpz)

    # Load Optimizer, Scheduler, Loss function: 
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    
    return(model,subway_ds,loss_function,optimizer,scheduler,args,dic_class2rpz)


def evaluate_config(dataset_names,folder_path,file_name,args,coverage,vision_model_name,mod_plot):
    # Load Model, Optimizer, Scheduler:
    model,subway_ds,loss_function,optimizer,scheduler,args,dic_class2rpz = load_everything(dataset_names,folder_path,file_name,args,coverage,vision_model_name)
    # Load trainer: 
    trainer = Trainer(subway_ds,model,args,optimizer,loss_function,scheduler = scheduler,dic_class2rpz = dic_class2rpz,show_figure = True)# Ajoute dans trainer, if calibration_prop is not None .... et on modifie le dataloader en ajoutant un clabration set
    # Train Model 
    trainer.train_and_valid(mod = 1000,mod_plot = mod_plot)  # Récupère les conformity scores sur I1, avec les estimations faites precedemment 

    pi,pi_cqr = plot_bokeh(trainer,args)
    return(trainer,model,args,pi,pi_cqr)

# =======================================================================================================================
# =======================================================================================================================