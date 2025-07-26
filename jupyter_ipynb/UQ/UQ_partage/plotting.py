import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import torch 
from PI import PI_object

def plot_PI_from_historical_UQ(real,lower,upper, station_i=0,window_size = 96*4):
    plt.figure(figsize=(12, 6))
    plt.plot(real.iloc[:window_size].index, real.iloc[:window_size], label='Real values', color='blue')
    plt.fill_between(x=real.iloc[:window_size].index, y1=lower.iloc[:window_size], y2=upper.iloc[:window_size], color='lightgray', alpha=0.7, label='Historical UQ')
    plt.title(f'PI of Station {station_i} from Historical Data')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


def plot_uncertainty_bands(L_predictions,Y_true,station_i, window_pred = np.arange(2*96), method = 'std_range', Lambda_coeffs =[1,2,3]):
    # Load prediction for a specific spatial unit, and a specific step_ahead
    predictions_unit_i = np.array([test_pred[:,station_i].detach().numpy() for test_pred in L_predictions])
    y_true = Y_true[:,station_i].numpy()

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
            ax.fill_between(grid,lower,upper, label = f"{lambda_coeff} std, PICP station {station_i}:{'{:.2%}'.format(picp)}, MPIW station {station_i}: {'{:.2f}'.format(mpiw)}",alpha=alpha, color='#86cfac')

    ax.set_xlabel('Time-slots')
    ax.set_ylabel('Flow')
    ax.legend()

    plt.show()



def plot_conformal_bands(preds,Y_true,station_i, q_order,pi,conformity_scores,window_pred = np.arange(2*96),bins = 20):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (18,6))
    pred = preds[window_pred][:,station_i,0]
    y_true = Y_true[window_pred][:,station_i,0]
    lower = pi.lower[window_pred][:,station_i,0]
    upper = pi.upper[window_pred][:,station_i,0]
    Q = pi.Q[station_i].item()


    ax1.plot(np.arange(len(pred)),pred, label = f'Prediction',alpha=1, color='#86cfac')
    ax1.plot(np.arange(len(y_true)),y_true,label = 'True')


    grid = np.arange(len(pred)).ravel()
    ax1.fill_between(grid,lower,upper, label = f"PI: {'{:.2f}'.format(q_order.item())}-th empirical quantile: {'{:.0f}'.format(Q)} \n PICP: {'{:.2%}'.format(pi.picp)} \n MPIW: {'{:.2f}'.format(pi.mpiw)}",alpha=0.6, color='#86cfac')

    ax1.set_xlabel('Time-slots')
    ax1.set_ylabel('Flow')
    ax1.legend()

    #histogram
    conformity_scores_first_station = conformity_scores[:,0].detach().cpu()
    ax2.hist(conformity_scores_first_station,bins = bins, label = 'Conformity Scores distirbution',density = True)
    ax2.plot([Q,Q],[0,0.2],color = 'red', linestyle = 'dashed',label = f"Q={'{:.0f}'.format(Q)} is the {'{:.2f}'.format(q_order.item())}-th quantile")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Proportion')
    ax2.legend()

    plt.show()



def plot_DQR(test_pred,Y_true,quantiles,i,window_pred = np.arange(2*96)):
    fig, ax = plt.subplots()
    restricted_pred, restricted_true = test_pred[window_pred][:,i,:],Y_true[window_pred][:,i,0]
    xaxis = np.arange(len(restricted_pred))
    restricted_pred = restricted_pred.unbind(-1)  #List of prediction associated to quantiles 

    # plot true
    ax.plot(xaxis,restricted_true, label = 'True', color = 'black')

    # plot mean :
    q = 0.5
    ind = int((len(restricted_pred)/2))
    ax.plot(xaxis,restricted_pred[ind],label = f"q = {'{:.1}'.format(q)}", alpha = 1,color = 'blue')

    # plot prediction
    colors = ['#86cfac','blue','lightblue']
    for pos in range(ind):
        q = quantiles[pos]
        upper = restricted_pred[pos]
        lower = restricted_pred[len(restricted_pred)-1-pos]
        pi = PI_object(torch.cat([test_pred[...,pos].unsqueeze(-1),test_pred[...,len(restricted_pred)-1-pos].unsqueeze(-1)],axis = -1),Y_true,1-q, type_calib = 'classic',Q = None)
        grid = xaxis.ravel()
        ax.fill_between(grid,lower,upper, label = f"PI {'{:.1%}'.format(1-q)}, PICP station {i}: {'{:.2%}'.format(pi.picp)}, MPIW station {i}: {'{:.2f}'.format(pi.mpiw)}",alpha=0.6/(1+2*pos), color=colors[pos])

    fig.legend()
    plt.show()


def plot_bands_CQR(trainer,Y_true,preds,pi_CQR,alpha,conformity_scores,station_i = 0, bins = 100,window_size = 96*4):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (25,8))
    restricted_true = Y_true[:window_size][:,station_i,:].cpu()
    restricted_preds = preds[:window_size][:,station_i,:].cpu()
    xaxis = np.arange(len(restricted_true))

    ### ------------ Figure 1: CQR vs Quantile Regression 
    # plot true
    ax1.plot(xaxis,restricted_true, label = 'True', color = 'black')

    # plot PI without calibration (only from quantile estimator) : 
    init_pi = PI_object(preds,Y_true,alpha,type_calib='classic')
    ql = restricted_preds[:,0].cpu()
    qu = restricted_preds[:,1].cpu()
    grid = xaxis.ravel()
    ax1.fill_between(grid,ql,qu, label = f"Quantile Regression (estimated {alpha/2} -  {1-alpha/2} quantiles): \n\
     PICP:  {'{:.2%}'.format(init_pi.picp)} \n\
     MPIW:  {'{:.2f}'.format(init_pi.mpiw)}",
               alpha=0.6, color='#86cfac')


    # Plot PI with calibration : 
    lower_band,upper_band = pi_CQR.lower[...,station_i,0][:window_size].cpu(),pi_CQR.upper[...,station_i,0][:window_size].cpu()
    ax1.plot(np.arange(len(lower_band)),lower_band,label = f"CQR ({'{:.2f}'.format(1-alpha)}% ): \n\
        PICP: {'{:.2%}'.format(pi_CQR.picp)} \n\
        MPIW: {'{:.2f}'.format(pi_CQR.mpiw)} ",
             color = 'red',linestyle = 'dashed')
    ax1.plot(np.arange(len(upper_band)),upper_band,color = 'red',linestyle = 'dashed')
    ax1.legend()
    # ----------------------------------

    ### ------------ Figure 2: Histogram 
    ax2.hist(conformity_scores[:,station_i].cpu(),bins = bins, label = 'Conformity Scores distirbution',density = True)
    ax2.plot([pi_CQR.Q[station_i],pi_CQR.Q[station_i]],[0,0.2],color = 'red', linestyle = 'dashed',label = f"Q={'{:.1f}'.format(pi_CQR.Q[station_i])} is the {'{:.2f}'.format(1-alpha)}-th quantile")
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Proportion')
    ax2.legend()
    # ----------------------------------

    ### ------------ Figure 3: Losses
    ax3.plot(np.arange(len(trainer.train_loss)),trainer.train_loss, label = 'training')
    ax3.plot(np.arange(len(trainer.valid_loss)),trainer.valid_loss, label = 'validation')
    ax3.legend()
    # ----------------------------------


    plt.show()