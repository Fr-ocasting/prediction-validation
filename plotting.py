import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch 
from PI_object import PI_object
import matplotlib.dates as mdates

from datetime import timedelta
from utilities_DL import get_associated__df_verif_index


def plot_k_fold_split(Datasets,invalid_dates):
    if not(type(Datasets) == list):
        Datasets = [Datasets]
    fig,ax = plt.subplots(figsize=(16,9))

    # Forbidden Dates
    delta_t = timedelta(hours= 1/Datasets[0].time_step_per_hour)
    already_ploted = []
    for i,invalid_date in enumerate(invalid_dates):
        if (not invalid_date in already_ploted):   # Avoid to plot too many vbar
            ax.axvspan(invalid_date, invalid_date+delta_t, alpha=0.3, color='grey',label= 'Invalid dates' if i==0 else None)
            date_for_grey_label = invalid_date
            already_ploted.append(invalid_date)

    for i,invalid_date in enumerate(invalid_dates):
        if Datasets[0].Weeks is not None:
            shift = int(Datasets[0].Weeks*24*7*Datasets[0].time_step_per_hour)
            if (not invalid_date+shift*delta_t in already_ploted):
                ax.axvspan(invalid_date+shift*delta_t, invalid_date+(shift+1)*delta_t, alpha=0.1, color='grey')
                already_ploted.append(invalid_date+shift*delta_t)

        if Datasets[0].Days is not None:
            shift = int(Datasets[0].Days*24*Datasets[0].time_step_per_hour)
            if (not invalid_date+shift*delta_t in already_ploted):
                ax.axvspan(invalid_date+shift*delta_t, invalid_date+(shift+1)*delta_t, alpha=0.1, color='grey')
                already_ploted.append(invalid_date+shift*delta_t)

        if Datasets[0].historical_len is not None:
            shift = int(Datasets[0].historical_len*Datasets[0].time_step_per_hour)
            if (not invalid_date+shift*delta_t in already_ploted):
                ax.axvspan(invalid_date+shift*delta_t, invalid_date+(shift+1)*delta_t, alpha=0.1, color='grey')
                already_ploted.append(invalid_date+shift*delta_t)
    ax.axvspan(date_for_grey_label, date_for_grey_label, alpha=0.1, color='grey', label = "Impacted Time-Slots which couldn't be predicted")
    # ...

    # K-folds : 
    dates_xticks = []
    for i,dset in enumerate(Datasets):

        # Convert Numpy Timestamp into 'mdates num'
        lpt1, lpt2, lpv1,lpv2,lpte1,lpte2  = mdates.date2num(dset.first_predicted_train_date),mdates.date2num(dset.last_predicted_train_date),mdates.date2num(dset.first_predicted_valid_date) ,mdates.date2num(dset.last_predicted_valid_date),mdates.date2num(dset.first_predicted_test_date) ,mdates.date2num(dset.last_predicted_test_date)
        lt1, lt2,lv1,lv2,lte1,lte2  = mdates.date2num(dset.first_train_date),mdates.date2num(dset.last_train_date), mdates.date2num(dset.first_valid_date),mdates.date2num(dset.last_valid_date),mdates.date2num(dset.first_test_date),mdates.date2num(dset.last_test_date)

        # Display specifics dates on the plot
        dates_xticks = dates_xticks + [x for x in [lpt1,lpt2,lpv1,lpv2,lpte1,lpte2,lt1,lt2,lv1,lv2,lte1,lte2 ] if not np.isnan(x)]  # Remove all the useless dates

        # Compute Width of each horizontal bar
        width_predict_train = lpt2 - lpt1
        width_predict_valid = lpv2 - lpv1
        width_predict_test = lpte2 - lpte1

        width_train_set = lt2 - lt1
        width_valid_set = lv2 - lv1   
        width_test_set = lte2 - lte1

        # Plot each horizontal bar (if exists):
        ax.barh(i-0.2, width_predict_train, left=lpt1, color='blue', height = 0.35, alpha = 0.7, label='Predicted Time-Slot within Train' if i == 0 else None)
        ax.barh(i+0.2, width_train_set, left=lt1, color='cornflowerblue', height = 0.35, alpha = 0.7, label='Values in TrainSet' if i == 0 else None)

        if not np.isnan(width_predict_valid):
            ax.barh(i-0.2, width_predict_valid, left=lpv1, color='orangered', height = 0.35, alpha = 0.7, label='Predicted Valid' if i == 0 else None)
            ax.barh(i+0.2, width_valid_set, left=lv1, color='coral', height = 0.35, alpha = 0.7, label='Values ValidSet' if i == 0 else None)

        if not np.isnan(width_predict_test):
            ax.barh(i-0.2, width_predict_test, left=lpte1, color='forestgreen', height = 0.35, alpha = 0.7, label='Predicted Test' if i == 0 else None)
            ax.barh(i+0.2, width_test_set, left=lte1, color='springgreen', height = 0.35, alpha = 0.7, label='Values TestSet' if i == 0 else None)
    # ...
            
    # Date formater
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Add xticks
    ax.set_xticks(dates_xticks)
    ax.tick_params(axis='x',rotation=70)

    # Might be useless : 
    fig.autofmt_xdate()

    #ax.legend()
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.show()




def plot_loss(trainer,test_pred,Y_true,window_pred = None):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (18,6))

    ax1.plot(np.arange(len(trainer.train_loss)),trainer.train_loss, label = 'train_loss')
    ax1.plot(np.arange(len(trainer.valid_loss)),trainer.valid_loss,label = 'Valid_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()

    try:
        pred = test_pred[:,0,0,0]
    except:
        pred = test_pred[:,0,0]
    if window_pred is not None:
        pred = pred[window_pred]
        Y_true = Y_true[window_pred]

    ax2.plot(np.arange(len(pred)),pred, label = 'prediction')
    ax2.plot(np.arange(len(Y_true)),Y_true[:,0,0],label = 'True')
    ax2.set_xlabel('Time-slots')
    ax2.set_ylabel('Flow')
    ax2.legend()

    plt.show()


def visualize_prediction_and_embedding_space(trainer,dataset,Q,args,args_embedding,plot2D = True,plot3D=False):
    trainer.model.eval()   # pas grad, pas de dropout 
    with torch.no_grad():
        #output = trainer.model.Tembedding(T_labels_cal.long())
        output = trainer.model.Tembedding(torch.arange(args_embedding.nb_words_embedding).to(args.device)).long().to(args.device)
    
    print(f"T embedding -> sum: {trainer.model.Tembedding.embedding.weight.grad.sum()}, mean: {trainer.model.Tembedding.embedding.weight.grad.sum()}")
    #print(f"output 0 -> sum: {trainer.model.Dense_outs[0].weight.grad.sum()}, mean: {trainer.model.Dense_outs[0].weight.grad.mean()}")
    #print(f"output 1 -> sum: {trainer.model.Dense_outs[1].weight.grad.sum()}, mean: {trainer.model.Dense_outs[1].weight.grad.mean()}")
    

    # Plot 3D: 
    if plot3D:
        X1,Y1,Z1 = output[:,0].cpu().numpy(),output[:,1].cpu().numpy(),output[:,2].cpu().numpy()
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(X1,Y1,Z1,label = 'embedding')
        ...

    # PLot 2D: 
    if plot2D: 
        # Plotting Temporal Embedding 
        X1,Y1 = output[:,0].cpu().numpy(),output[:,1].cpu().numpy()
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (15,8))
        ax1.scatter(X1,Y1,label = 'embedding')
        ax1.set_xlim([-1,1])
        ax1.set_ylim([-1,1])
        ax1.legend()
        # ...

        # Plotting Loss
        ax2.plot(np.arange(len(trainer.valid_loss)),trainer.valid_loss,label = f"validation loss: {'{:.4f}'.format(trainer.train_loss[-1])}")
        ax2.plot(np.arange(len(trainer.train_loss)),trainer.train_loss,label = f"Training loss:  {'{:.4f}'.format(trainer.valid_loss[-1])}")
        ax2.legend()
        # ...

        # Prediction Test Set
        (preds,Y_true,T_labels,df_metrics) = trainer.testing(dataset,metrics= ['mse','mae'])
        if len(preds.size()) == 2:
            preds = preds.unsqueeze(1)
        # ...

        # PI
        pi = PI_object(preds,Y_true,alpha = args.alpha, type_calib = 'classic')     # PI 'classic' :
        pi_cqr = PI_object(preds,Y_true,alpha = args.alpha, Q = Q, type_calib = 'CQR',T_labels = T_labels)      # PI 'CQR' 
        # str legend
        str_picp,str_mpiw = f"{'{:.2%}'.format(pi.picp)}" , f"{'{:.2f}'.format(pi.mpiw)}"
        str_picp_cqr, str_mpiw_cqr = f"{'{:.2%}'.format(pi_cqr.picp)}" , f"{'{:.2f}'.format(pi_cqr.mpiw)}"
        str_pi_alpha = f"{'{:.2f}'.format(1-args.alpha)}%"
        # ...

        # Plotting Prediction
        ax3.plot(np.arange(100),pi_cqr.upper[:100,0,0].cpu(),color = 'green',linestyle = 'dashed',label = f"PI {str_pi_alpha}, \n PICP: {str_picp_cqr} \n MPIW: {str_mpiw_cqr}")
        ax3.plot(np.arange(100),pi_cqr.lower[:100,0,0].cpu(),color = 'green',linestyle = 'dashed')
        ax3.plot(np.arange(100),pi.upper[:100,0,0].cpu(),color = 'red',linestyle = 'dashed',label = f"quantile estimation \n PICP: {str_picp} \n MPIW: {str_mpiw}")
        ax3.plot(np.arange(100),pi.lower[:100,0,0].cpu(),color = 'red',linestyle = 'dashed')
        ax3.plot(np.arange(100),Y_true[:100,0,0].cpu(),color = 'blue',label = 'True value')
        ax3.legend()
        # ...

        plt.show()





def plot_coverage_matshow(data, x_labels = None, y_labels = None, log = False, cmap ="afmhot", save = None, cbar_label =  "Number of Data"):
    # Def function to plot a df with matshow
    # Use : plot the coverage through week and days 

    if log : 
        data = np.log(data + 1)
    
    data[data == 0] = np.nan

    fig = plt.figure(figsize=(40, 12))
    cax = plt.matshow(data.values, cmap=cmap)  #

    cmap_perso = plt.get_cmap(cmap)
    cmap_perso.set_bad('gray', 1.0)  # Configurez la couleur grise pour les valeurs nulles

    # Configurez la colormap pour gérer les valeurs NaN comme le gris
    cax.set_cmap(cmap_perso)
    cax.set_clim(vmin=0.001, vmax=data.max().max())  # Ajustez les limites pour exclure les NaN


    #x labels
    if x_labels is None:
        x_labels = data.columns.values
    plt.gca().set_xticks(range(len(x_labels)))
    plt.gca().set_xticklabels(x_labels, rotation=85, fontsize=8)
    plt.gca().xaxis.set_ticks_position('bottom')

    #y labels
    if y_labels is None: 
        y_labels = data.index.values
    plt.gca().set_yticks(range(len(y_labels)))
    plt.gca().set_yticklabels(y_labels, fontsize=8)

    # Add a colorbar to the right of the figure
    cbar = plt.colorbar(cax, aspect=10)
    cbar.set_label(cbar_label)  # You can customize the label as needed

    if save is not None: 
            plt.savefig(save, format="pdf")
            
def coverage_day_month(df_metro,freq= '24h',index = 'month_year',columns = 'day_date',save = 'subway_id',folder_save = 'save/'):

    df_agg = df_metro.groupby([pd.Grouper(key = 'datetime',freq = freq)]).agg('sum').reset_index()[['datetime','in','out']]
    df_agg['date']= df_agg.datetime.dt.date
    df_agg['day_date'] = df_agg.datetime.dt.day
    df_agg['month_year']= df_agg.datetime.dt.month.transform(lambda x : str(x)) + ' ' + df_agg.datetime.dt.year.transform(lambda x : str(x))
    df_agg['month_year']= pd.to_datetime(df_agg['month_year'],format = '%m %Y')
    #df_agg['hour']= df_agg.datetime.dt.hour.transform(lambda x : str(x)) + ':' + df_agg.datetime.dt.minute.transform(lambda x : str(x))
    df_agg['hour']= df_agg.datetime.dt.hour + df_agg.datetime.dt.minute*0.01
    df_agg['tot'] = df_agg['in'] + df_agg['out']
    # Pivot

    df_agg_in = df_agg.pivot(index = index,columns = columns,values = 'in').fillna(0)
    df_agg_out = df_agg.pivot(index = index,columns = columns,values = 'out').fillna(0)
    df_agg_tot = df_agg.pivot(index = index,columns = columns,values = 'tot').fillna(0)
    
    if index == 'month_year':
        df_agg_in.index = df_agg_in.index.strftime('%Y-%m')
        df_agg_out.index = df_agg_out.index.strftime('%Y-%m')
        df_agg_tot.index = df_agg_out.index.strftime('%Y-%m')


    # Plot 
    plot_coverage_matshow(df_agg_in, log  = False, cmap = 'YlOrRd',save = f'{folder_save}in_{save}')   
    plot_coverage_matshow(df_agg_out, log  = False, cmap = 'YlOrRd',save = f'{folder_save}out_{save}')  
    plot_coverage_matshow(df_agg_tot, log  = False, cmap = 'YlOrRd',save = f'{folder_save}tot_{save}')  
    return(df_agg_in,df_agg_out)

if __name__ == '__main__':
    # Exemple with 'plot_coverage_matshow':
    range_dates = pd.date_range(start= "2019-9-30",end="2021-5-31",freq = '7D')
    data = pd.DataFrame(np.random.randint(50,size = (88,7)), index = range_dates)
    data.index = data.index.strftime('%Y-%m-%d')
    plot_coverage_matshow(data, log  = False, cmap = 'YlOrRd')

    