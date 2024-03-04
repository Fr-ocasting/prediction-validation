import torch
import torch.nn as nn
from DL_utilities import DictDataLoader,Trainer,PI_object,QuantileLoss,get_loss,choose_optimizer
import numpy as np 
import matplotlib.pyplot as plt

def plot_bands_CQR(trainer,Y_true,preds,pi,window_pred,alpha,conformity_scores,results,bins = 100):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (25,8))
    lower_band,upper_band = pi.lower[...,0,0][window_pred],pi.upper[...,0,0][window_pred]
    restricted_true = Y_true[window_pred][:,0,0]

    xaxis = np.arange(len(restricted_true))

    # plot true
    ax1.plot(xaxis,restricted_true, label = 'True', color = 'black')

    # plot PI
    ax1.plot([0],[0],label = f"{'{:.2f}'.format(1-alpha)}-th empirical quantile: {'{:.0f}'.format(pi.Q[0,0,0].item())}",linestyle = 'dashed')
    grid = xaxis.ravel()
    ax1.fill_between(grid,lower_band,upper_band, label = f"PI {'{:.2f}'.format(1-alpha)}% \n\
                                                        PICP: {'{:.2%}'.format(pi.picp)} \n\
                                                        MPIW: {'{:.2f}'.format(pi.mpiw)} \n\
                                                        last train loss {'{:.4f}'.format(results['last train loss'])} \n\
                                                        last valid loss {'{:.4f}'.format(results['last valid loss'])} ",
               alpha=0.6, color='#86cfac')

    # Lower and upper band 
    ql = preds[window_pred][:,0,0]
    qu = preds[window_pred][:,0,1]
    ax1.plot(np.arange(len(ql)),ql,label = f'Estimated lower {alpha/2}quantile',color = 'red',linestyle = 'dashed')
    ax1.plot(np.arange(len(qu)),qu,label = f'Estimated upper {1-alpha/2}quantile',color = 'red',linestyle = 'dashed')

    ax1.legend()
    
    
    ax2.plot(np.arange(len(trainer.train_loss)),trainer.train_loss, label = 'training')
    ax2.plot(np.arange(len(trainer.valid_loss)),trainer.valid_loss, label = 'validation')
    ax2.legend()

    ax3.hist(conformity_scores,bins = bins, label = 'Conformity Scores distirbution',density = True)
    ax3.plot([pi.Q[0,0,0],pi.Q[0,0,0]],[0,0.2],color = 'red', linestyle = 'dashed',label = f"Q={'{:.0f}'.format(pi.Q[0,0,0].item())} is the {'{:.2f}'.format(1-alpha)}-th quantile")
    #ax3.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Proportion')
    ax3.legend()
    
    plt.show()


class DeepEnsemble(object):
    def __init__(self,Models,dataset,dataloader,args):
        super(DeepEnsemble,self).__init__()
        self.Models = Models

        self.dataloader = dataloader
        self.dataset = dataset
        self.args = args

    def train_valid_model(self,model,mod):
        optimizer = choose_optimizer(model,self.args)
        loss_function = get_loss(self.args.loss_function_type,quantiles = None)
        trainer = Trainer(model,self.dataloader,self.args.epochs,optimizer,loss_function,scheduler = self.args.scheduler, ray = self.args.ray)
        trainer.train_and_valid(mod = mod)
        return(trainer)

    def train_and_test_n_times(self, mod=1000,metrics= ['mse','mae']):
        L_predictions = []
        for model in self.Models: 
            trainer = self.train_valid_model(model,mod)
            test_pred,Y_true,df_metrics = trainer.testing(self.dataset,metrics)
            L_predictions.append(test_pred)

        self.L_predictions = L_predictions
        self.Y_true =Y_true
    
    def plot_spatial_unit_i(self,i,output_ind = 0, window_pred = np.arange(2*96), method = 'std_range', Lambda_coeffs =[1,2,3]):
        # Load prediction for a specific spatial unit, and a specific step_ahead
        predictions_unit_i = np.array([test_pred[:,i,output_ind].detach().numpy() for test_pred in self.L_predictions])
        y_true = self.Y_true[:,i,output_ind].numpy()

        # Keep track on global mean and std :
        y_i_mean = predictions_unit_i.mean(axis = 0)
        y_i_std = predictions_unit_i.std(axis = 0)

        if window_pred is not None: 
            y_true,y_i_mean,y_i_std = y_true[window_pred],y_i_mean[window_pred],y_i_std[window_pred]

        fig, ax = plt.subplots(figsize = (18,6))
        ax.plot(np.arange(len(y_i_mean)),y_i_mean, label = f'mean prediction',alpha=1, color='#86cfac')
        ax.plot(np.arange(len(y_true)),y_true,label = 'True')

        if method == 'std_range':
            for lambda_coeff in Lambda_coeffs:
                lower = y_i_mean-lambda_coeff*y_i_std
                upper = y_i_mean+lambda_coeff*y_i_std
                picp = ((lower < y_true)&(y_true<upper)).sum()/len(lower)
                mpiw = (upper-lower).mean()                             
                grid = np.arange(len(y_i_mean)).ravel()
                mini,maxi = max(Lambda_coeffs),min(Lambda_coeffs)
                alpha = 0.6 if len(Lambda_coeffs) == 1 else 0.7*(1-0.8*(maxi-lambda_coeff)/(maxi-mini))
                ax.fill_between(grid,lower,upper, label = f"{lambda_coeff} std, PICP station {i}:{'{:.2%}'.format(picp)}, MPIW station {i}: {'{:.2f}'.format(mpiw)}",alpha=alpha, color='#86cfac')

        ax.set_xlabel('Time-slots')
        ax.set_ylabel('Flow')
        ax.legend()

        plt.show()

