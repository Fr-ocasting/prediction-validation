import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch 
import matplotlib.dates as mdates
from datetime import timedelta


# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal imports:
from pipeline.PI.PI_object import PI_object
from pipeline.utils.metrics import error_along_ts
from pipeline.calendar_class import is_morning_peak,is_evening_peak,is_weekday,get_temporal_mask



def plot_k_fold_split(Datasets,invalid_dates,figsize=(14,14),save_path = None,hpo = True):
    if not(type(Datasets) == list):
        Datasets = [Datasets]
    fig,ax = plt.subplots(figsize=figsize)

    # Forbidden Dates
    delta_t = timedelta(hours= 1/Datasets[0].time_step_per_hour)
    already_ploted = []
    for i,invalid_date in enumerate(invalid_dates):
        if (not invalid_date in already_ploted):   # Avoid to plot too many vbar
            ax.axvspan(invalid_date, invalid_date+delta_t, alpha=0.3, color='grey') #,label= 'Invalid dates' if i==0 else None)
            date_for_grey_label = invalid_date
            already_ploted.append(invalid_date)

    for i,invalid_date in enumerate(invalid_dates):
        if Datasets[0].W is not None:
            shift = int(Datasets[0].W*24*7*Datasets[0].time_step_per_hour)
            if (not invalid_date+shift*delta_t in already_ploted):
                ax.axvspan(invalid_date+shift*delta_t, invalid_date+(shift+1)*delta_t, alpha=0.1, color='grey')
                already_ploted.append(invalid_date+shift*delta_t)

        if Datasets[0].D is not None:
            shift = int(Datasets[0].D*24*Datasets[0].time_step_per_hour)
            if (not invalid_date+shift*delta_t in already_ploted):
                ax.axvspan(invalid_date+shift*delta_t, invalid_date+(shift+1)*delta_t, alpha=0.1, color='grey')
                already_ploted.append(invalid_date+shift*delta_t)

        if Datasets[0].H is not None:
            shift = int(Datasets[0].H*Datasets[0].time_step_per_hour)
            if (not invalid_date+shift*delta_t in already_ploted):
                ax.axvspan(invalid_date+shift*delta_t, invalid_date+(shift+1)*delta_t, alpha=0.1, color='grey')
                already_ploted.append(invalid_date+shift*delta_t)
    ax.axvspan(date_for_grey_label, date_for_grey_label, alpha=0.3, color='grey',label = "Invalid dates and Impacted Time-Slots\nwhich couldn't be predicted")  #label = "Impacted Time-Slots which couldn't be predicted")
    # ...

    # K-folds : 
    dates_xticks = []
    fontsize = 12*figsize[1]/14
    K_fold = len(Datasets)
    for i,dset in enumerate(Datasets):
        limits = dset.tensor_limits_keeper

        # Convert Numpy Timestamp into 'mdates num'
        lpt1, lpt2, lpv1,lpv2,lpte1,lpte2  = mdates.date2num(limits.first_predicted_train_date),mdates.date2num(limits.last_predicted_train_date),mdates.date2num(limits.first_predicted_valid_date) ,mdates.date2num(limits.last_predicted_valid_date),mdates.date2num(limits.first_predicted_test_date) ,mdates.date2num(limits.last_predicted_test_date)
        lt1, lt2,lv1,lv2,lte1,lte2  = mdates.date2num(limits.first_train_date),mdates.date2num(limits.last_train_date), mdates.date2num(limits.first_valid_date),mdates.date2num(limits.last_valid_date),mdates.date2num(limits.first_test_date),mdates.date2num(limits.last_test_date)

        # Display specifics dates on the plot
        if i == 0:
            dates_xticks = dates_xticks + [x for x in [lt1] if not np.isnan(x)]
        if i == len(Datasets)-1:
            dates_xticks = dates_xticks + [x for x in [lpte2] if not np.isnan(x)]    
        #dates_xticks = dates_xticks + [x for x in [lpt1,lpt2,lpv1,lpv2,lpte1,lpte2,lt1,lt2,lv1,lv2,lte1,lte2 ] if not np.isnan(x)]  # Remove all the useless dates

        # Compute Width of each horizontal bar
        width_predict_train = lpt2 - lpt1
        width_predict_valid = lpv2 - lpv1
        width_predict_test = lpte2 - lpte1

        width_train_set = lt2 - lt1
        width_valid_set = lv2 - lv1   
        width_test_set = lte2 - lte1

        # Plot each horizontal bar (if exists):
        ax.barh(i-0.2, width_predict_train, left=lpt1, color='blue', height = 0.35, alpha = 0.7, label='Predicted Train' if i == 0 else None)
        ax.barh(i+0.2, width_train_set, left=lt1, color='cornflowerblue', height = 0.35, alpha = 0.7, label='Time-slots within Train Set' if i == 0 else None)

        ax.text(lpt1 + width_predict_train / 2, i - 0.2, f'{mdates.num2date(lpt1).strftime("%m/%d")} - {mdates.num2date(lpt2).strftime("%m/%d")}', ha='center', va='center', fontsize=fontsize, color='black')
        ax.text(lt1 + width_train_set / 2, i + 0.2, f'{mdates.num2date(lt1).strftime("%m/%d")} - {mdates.num2date(lt2).strftime("%m/%d")}', ha='center', va='center', fontsize=fontsize, color='black')

        if not np.isnan(width_predict_valid):
            ax.barh(i-0.2, width_predict_valid, left=lpv1, color='orangered', height = 0.35, alpha = 0.7, label='Predicted Valid' if i == 0 else None)
            ax.barh(i+0.2, width_valid_set, left=lv1, color='coral', height = 0.35, alpha = 0.7, label='Time-slots within Valid Set' if i == 0 else None)

            ax.text(lpv1 + width_predict_valid / 2, i - 0.2, f'{mdates.num2date(lpv1).strftime("%m/%d")} - {mdates.num2date(lpv2).strftime("%m/%d")}', ha='center', va='center', fontsize=fontsize, color='black')
            ax.text(lv1 + width_valid_set / 2, i + 0.2, f'{mdates.num2date(lv1).strftime("%m/%d")} - {mdates.num2date(lv2).strftime("%m/%d")}', ha='center', va='center', fontsize=fontsize, color='black')

        if i != 0 or not hpo:
            if not np.isnan(width_predict_test):
                first_test_iter = (hpo and i == 1) or (not hpo and i == 0)
                
                label_test = 'Predicted Test' if first_test_iter else None
                label_test_set = 'Time-slots within Test Set' if first_test_iter else None

                # Plot the bars
                ax.barh(i-0.2, width_predict_test, left=lpte1, color='forestgreen', height = 0.35, alpha = 0.7, label=label_test)
                ax.barh(i+0.2, width_test_set, left=lte1, color='springgreen', height = 0.35, alpha = 0.7, label=label_test_set)
                
                # Plot the text
                ax.text(lpte1 + width_predict_test / 2, i - 0.2, f'{mdates.num2date(lpte1).strftime("%m/%d")} - {mdates.num2date(lpte2).strftime("%m/%d")}', ha='center', va='center', fontsize=fontsize, color='black')
                ax.text(lte1 + width_test_set / 2, i + 0.2, f'{mdates.num2date(lte1).strftime("%m/%d")} - {mdates.num2date(lte2).strftime("%m/%d")}', ha='center', va='center', fontsize=fontsize, color='black')


            
    # Date formater
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Add xticks
    ax.set_xticks(dates_xticks)
    ax.tick_params(axis='x',rotation=30,labelsize = fontsize)
    # Add yticks
    ax.set_yticks(np.arange(K_fold))
    if hpo : 
        ax.set_yticklabels(['Fold HPO']+[f'Fold {x}' for x in list(map(str,list(np.arange(K_fold))))[1:]])
    else:
        ax.set_yticklabels([f'Fold {x}' for x in list(map(str,list(np.arange(K_fold))))])
    # Might be useless : 
    fig.autofmt_xdate()
    ax.set_xlabel('Date',fontsize=fontsize)

    #ax.legend()
    #ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center',fontsize=fontsize) #loc='upper left'

    # ----- BEGIN BLOCK LEGENDS -----
    handles, labels = ax.get_legend_handles_labels()
    
    # Create a dictionary for easier access
    legend_data = dict(zip(labels, handles))

    # --- HELPER FUNCTION to find label regardless of space ---
    def get_handle_and_label(key_with_space, key_no_space):
        if key_with_space in legend_data:
            return legend_data[key_with_space], key_with_space
        if key_no_space in legend_data:
            return legend_data[key_no_space], key_no_space
        return None, None
    # --------------------------------------------------------

    # Define the labels for each block
    h, l_invalid = get_handle_and_label("Invalid dates and Impacted Time-Slots\nwhich couldn't be predicted", "Invalid dates and Impacted Time-Slots\nwhich couldn't be predicted")
    
    h_pt, l_pt = get_handle_and_label("Predicted Train", "Predicted Train")
    h_ts, l_ts = get_handle_and_label("Time-slots within Train Set", "Time-slots within Train Set")

    h_pv, l_pv = get_handle_and_label("Predicted Valid", "Predicted Valid")
    h_vS, l_vS = get_handle_and_label("Time-slots within Valid Set", "Time-slots within Valid Set")

    h_pte, l_pte = get_handle_and_label("Predicted Test", "Predicted Test")
    h_teS, l_teS = get_handle_and_label("Time-slots within Test Set", "Time-slots within Test Set")

    # Create handles lists
    block1_handles = [h]
    block1_labels = [l_invalid]
    
    block2_handles = [h_pt, h_ts]
    block2_labels = [l_pt, l_ts]

    block3_handles = [h_pv, h_vS]
    block3_labels = [l_pv, l_vS]

    block4_handles = [h_pte, h_teS]
    block4_labels = [l_pte, l_teS]

    # Filter out any None values (if a label wasn't found)
    block1_handles, block1_labels = zip(*[(h, l) for h, l in zip(block1_handles, block1_labels) if h])
    block2_handles, block2_labels = zip(*[(h, l) for h, l in zip(block2_handles, block2_labels) if h])
    block3_handles, block3_labels = zip(*[(h, l) for h, l in zip(block3_handles, block3_labels) if h])
    block4_handles, block4_labels = zip(*[(h, l) for h, l in zip(block4_handles, block4_labels) if h])

    # Create the four legends
    if block1_handles:
        leg1 = ax.legend(block1_handles, block1_labels, bbox_to_anchor=(0.2, 0.0), loc='upper center', fontsize=fontsize, frameon=False)
        ax.add_artist(leg1)
    
    if block2_handles:
        leg2 = ax.legend(block2_handles, block2_labels, bbox_to_anchor=(0.4, 0.0), loc='upper center', fontsize=fontsize, frameon=False)
        ax.add_artist(leg2)
    
    if block3_handles:
        leg3 = ax.legend(block3_handles, block3_labels, bbox_to_anchor=(0.6, 0.0), loc='upper center', fontsize=fontsize, frameon=False)
        ax.add_artist(leg3)
    
    if block4_handles:
        leg4 = ax.legend(block4_handles, block4_labels, bbox_to_anchor=(0.8, 0.0), loc='upper center', fontsize=fontsize, frameon=False)
        ax.add_artist(leg4)
    # ----- END BLOCK LEGENDS -----

    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
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




def plot_coverage_matshow(data, x_labels = None, y_labels = None, log = False, 
                          cmap ="afmhot", save = None, cbar_label =  "Number of Data",bool_reversed=False
                          ,v_min=None,v_max=None,
                          display_values = False,
                          bool_plot = None,cbar_magic_args = False,
                          figsize = None
                          ):
    # Def function to plot a df with matshow
    # Use : plot the coverage through week and days 

    if log : 
        data = np.log(data + 1)
    
    data[data == 0] = np.nan
    if figsize is not None:
        matfig = plt.figure(figsize=figsize)
        cax = plt.matshow(data.values, cmap=cmap,aspect='auto',fignum=matfig.number)  #
    else:
        cax = plt.matshow(data.values, cmap=cmap,fignum=False)  #

    #cmap_perso = plt.get_cmap(cmap)
    if bool_reversed: 
        cmap_perso =  plt.get_cmap(cmap).reversed()
    else: 
        cmap_perso =  plt.get_cmap(cmap)
    cmap_perso.set_bad('gray', 1.0)  # Configurez la couleur grise pour les valeurs nulles

    # Configurez la colormap pour gérer les valeurs NaN comme le gris
    cax.set_cmap(cmap_perso)
    if v_min is None:
        v_min=0.001
    if v_max is None:
        v_max=data.max().max()
    cax.set_clim(vmin=v_min, vmax=v_max)  # Ajustez les limites pour exclure les NaN


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

    if cbar_magic_args :
        cbar = plt.colorbar(cax,fraction=0.046, pad=0.04)
    else:
        cbar = plt.colorbar(cax, aspect=10)
    cbar.set_label(cbar_label)  # You can customize the label as needed

    ## Plot values if needed: 
    if display_values:
        for (i, j), val in np.ndenumerate(data.values):
            if not np.isnan(val):
                plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=6)

    if save is not None: 
            plt.savefig(save, format="pdf")

def coverage_day_month(df_metro,freq= '24h',index = 'month_year',columns = 'day_date',save = 'subway_id',folder_save = 'save/',key_columns = ['datetime','in','out']):
    
    df_agg = add_calendar_columns(df_metro,freq,key_columns)
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


def add_calendar_columns(df_agg,freq=None,key_columns=None,agg_func = 'sum'):
    if freq is not None:
        df_agg = df_agg.groupby([pd.Grouper(key = 'datetime',freq = freq)]).agg(agg_func).reset_index()
    if key_columns is not None:
        df_agg=df_agg[key_columns]
    df_agg['date']= df_agg.datetime.dt.date
    df_agg['day_date'] = df_agg.datetime.dt.day
    df_agg['month_year']= df_agg.datetime.dt.month.transform(lambda x : str(x)) + ' ' + df_agg.datetime.dt.year.transform(lambda x : str(x))
    df_agg['month_year']= pd.to_datetime(df_agg['month_year'],format = '%m %Y')
    #df_agg['hour']= df_agg.datetime.dt.hour.transform(lambda x : str(x)) + ':' + df_agg.datetime.dt.minute.transform(lambda x : str(x))
    df_agg['hour']= df_agg.datetime.dt.hour + df_agg.datetime.dt.minute*0.01
    df_agg['weekday']= df_agg.datetime.dt.weekday
    return df_agg


def build_matrix_for_matshow(ds,column,training_mode,error,freq,index_matshow,columns_matshow):
    '''
    From pipeline.a time-series of error (error) and associated dates, 
    return de pd.DataFrame with :
    >>> 'index_matshow' (day) as index
    >>> 'columns_matshow' (hour) as column
    and containing the associated flow by hour and by day 
    '''
    df_verif = getattr(ds.tensor_limits_keeper,f"df_verif_{training_mode}")
    dates = df_verif.iloc[:,-1]

    # Get predition error :
    df_error_station_i = pd.DataFrame({column: error, 'datetime':dates})

    # Plotting error by day/hour to display daily pattern : 
    df_agg = add_calendar_columns(df_error_station_i,freq=freq,key_columns=df_error_station_i.columns,agg_func = 'mean')
    if columns_matshow is None:
        df_agg['dummy'] = index_matshow
        columns_matshow = 'dummy'
    df_agg = df_agg.pivot_table(index = index_matshow,columns = columns_matshow,values = column,aggfunc='mean').fillna(0)

    return df_agg


def plot_matshow(df_agg,column,metric,v_min,v_max,cmap,cbar_label,bool_reversed,axes,station_c,ind_metric):
    ''' Plotting function for 'error_per_station_calendar_pattern' '''
    if len(axes.shape)==1:
        plt.sca(axes[ind_metric])
    else:
        plt.sca(axes[station_c,ind_metric])       
    plot_coverage_matshow(df_agg, log=False, cmap=cmap, save=None, cbar_label=cbar_label,bool_reversed=bool_reversed,v_min=v_min,v_max=v_max)

    if metric == 'previous_value':
        title = f"Gain (%) of MSE error compared to using previous value as prediction"
    else : 
        title = f"{metric} error on station {column}"  
    if len(axes.shape)==1:     
        axes[ind_metric].set_title(title)
    else:     

        axes[station_c,ind_metric].set_title(title)




def get_gain_from_mod1(real,predict1,predict2,previous,min_flow,metrics = ['mse'],acceptable_error= 0,mape_acceptable_error=0):
    '''

    args:
    -----
    
    '''
    dic_gain,dic_error = {},{}
    real = real.detach().clone().reshape(-1)
    previous = previous.detach().clone().reshape(-1)  

    predict1 = predict1.detach().clone().reshape(-1)    
    predict2 = predict2.detach().clone().reshape(-1)   

    mask = real>min_flow

    # Tackle metrics: 
    for metric in metrics:
        if metric == 'mse':
            error_pred1 = (real - predict1)**2
            error_pred2 = (real - predict2)**2
            local_acceptable_error = acceptable_error**2
        elif metric == 'mae':
            error_pred1 = abs(real - predict1)
            error_pred2 = abs(real - predict2)
            local_acceptable_error = acceptable_error
        elif metric == 'mape':
            error_pred1 = torch.full(real.shape, -1.0)  # Remplir avec -1 par défaut
            error_pred2 = torch.full(real.shape, -1.0)  # Remplir avec -1 par défaut
            error_pred1[mask] = 100 * (torch.abs(real[mask] - predict1[mask]) / real[mask]) 
            error_pred2[mask] = 100 * (torch.abs(real[mask] - predict2[mask]) / real[mask]) 
            local_acceptable_error = mape_acceptable_error
            
        else:
            raise NotImplementedError
        dic_error[metric] = {'error_pred1':error_pred1,
                             'error_pred2':error_pred2}


        # In case the reference error (model1), is too small, we use an 'acceptable error' 
        cloned_error_pred1 = error_pred1.clone()
        local_mask = error_pred1 <  local_acceptable_error
        cloned_error_pred1[local_mask] = local_acceptable_error  
        gain = 100*(error_pred2-error_pred1)/cloned_error_pred1

        dic_gain[metric] = gain

    # Tackle Naiv :
    if 'mae' in metrics: 
        error_naiv = abs(real-previous)
        gain_naiv1 = dic_error['mae']['error_pred1']-error_naiv
        gain_naiv2 = dic_error['mae']['error_pred2']-error_naiv
        dic_error['mase'] = {'error_pred1':gain_naiv1,
                            'error_pred2':gain_naiv2}
        dic_error['mae_naiv'] = {'error_naiv':error_naiv}
        # ....

    return dic_gain,dic_error

def get_fig_size(nb_station_to_plots,columns_matshow):
    y_size = nb_station_to_plots*5
    if columns_matshow is None:
        x_size = 10
    else:
        x_size = 30
    figsize = (x_size,y_size)
    return figsize


def plot_gains(ds1,column,station_ind,training_mode,dic_error,freq,index_matshow,columns_matshow, metrics,v_min,v_max,cmap,bool_reversed,axes):
    for ind_metric,metric_i in enumerate(metrics) : 
        cbar_label = f"Gain {metric_i}"
        # Build Matshow matrix : 
        df_agg = build_matrix_for_matshow(ds1,column,training_mode,dic_error[metric_i],freq,index_matshow,columns_matshow)

        # Plotting : 
        plot_matshow(df_agg,column,metric_i,v_min,v_max,cmap,cbar_label,bool_reversed,axes,station_ind,ind_metric)      
   



def gain_between_models(trainer1,trainer2,ds1,ds2,training_mode,
                         metrics = ['mse','mae','mape'],
                        freq='1h',
                        index_matshow = 'day_date',
                        columns_matshow = 'hour',
                        min_flow = 20,
                        limit_percentage_error = 300,
                        acceptable_error = 10,
                        stations = None,
                        plot_each_station = False,
                        plot_all_station = True

                         ):
    '''

    '''
    # Init:
    if stations is not None:
        n_station = len(stations) 
    else:
        n_station = len(ds1.spatial_unit)
        stations = list(ds1.spatial_unit)

    nb_station_to_plots = plot_each_station*n_station + plot_all_station
    fig, axes = plt.subplots(nb_station_to_plots, len(metrics), figsize=get_fig_size(nb_station_to_plots,columns_matshow))

    # Get Pred1,Pred2, TrueValues:
    full_predict1,Y_true,_ = trainer1.testing(ds1.normalizer, training_mode =training_mode)
    full_predict2,_,_ = trainer2.testing(ds2.normalizer, training_mode =training_mode)
    inputs = [[x,y,x_c] for  x,y,x_c in ds1.dataloader[training_mode]]
    previous  = torch.cat([x for x,_,_ in inputs],0)
    previous  = ds1.normalizer.unormalize_tensor(inputs = previous,feature_vect = True) # unormalize input cause prediction is unormalized 

    # Set cmap:
    cmap = 'RdYlBu'
    bool_reversed = True
    v_min,v_max = -limit_percentage_error,limit_percentage_error

    # Evaluate Prediciton per stations:
    if plot_each_station:
        for station_ind in range(n_station):
            station_c = list(ds1.spatial_unit).index(stations[station_ind])
            column = stations[station_ind]

            dic_gain,dic_error = get_gain_from_mod1(real = Y_true[:,station_c:station_c+1,:],
                                        predict1 = full_predict1[:,station_c:station_c+1,:],
                                        predict2 = full_predict2[:,station_c:station_c+1,:],
                                        previous = previous[:,station_c:station_c+1,-1],
                                        min_flow=min_flow,metrics = metrics,acceptable_error= acceptable_error) 
            plot_gains(ds1,column,station_ind,training_mode,dic_gain,freq,index_matshow,columns_matshow, metrics,v_min,v_max,cmap,bool_reversed,axes)

    if plot_all_station:
        station_ind = n_station if plot_each_station else 1
        column = 'All'
        T,N,C = Y_true.size()
        dic_gain,dic_error = get_gain_from_mod1(real =Y_true,
                                    predict1 = full_predict1,
                                    predict2 = full_predict2,
                                    previous = previous[:,:,-1],
                                    min_flow=min_flow,metrics = metrics,acceptable_error= acceptable_error)
        dic_gain = {metric_i:error_i.reshape(T,N).mean(axis=1) for metric_i,error_i in dic_gain.items()}
        plot_gains(ds1,column,station_ind,training_mode,dic_gain,freq,index_matshow,columns_matshow, metrics,v_min,v_max,cmap,bool_reversed,axes)



    plt.tight_layout()
    plt.show()
    return fig,axes




def error_per_station_calendar_pattern(trainer,ds,training_mode,
                                       metrics = ['mse','mae','mape','previous_value'],
                                       freq='1h',
                                       index_matshow = 'day_date',
                                       columns_matshow = 'hour',
                                       min_flow = 20,
                                       figsize = (20,20),
                                       limit_percentage_error = 300,
                                       acceptable_error = 10,
                                       stations = None
                                       ):
    '''
    args:
    ------
    #Parameter for matshow: 
    freq : frequence of temporal aggregation. 
    index_matshow : set the type of calendar information to display along the rows. 
    columns_matshow : set the type of calendar information to display along the columns.
    '''
    if stations is not None:
        n_station = len(stations) 
    else:
        n_station = len(ds.spatial_unit)
    fig, axes = plt.subplots(n_station+1, len(metrics), figsize=figsize)

    # Get Prediction
    Preds,Y_true,T_labels = trainer.testing(ds.normalizer, training_mode =training_mode)
    inputs = [[x,y,x_c] for  x,y,x_c in ds.dataloader[training_mode]]
    X = torch.cat([x for x,_,_ in inputs],0)
    X = ds.normalizer.unormalize_tensor(inputs = X,feature_vect = True) # unormalize input cause prediction is unormalized 
    #index_perrache = list(ds.spatial_unit).index('PER')

    # Evaluate Prediciton per stations:
    for station_ind in range(n_station+1):
        if station_ind<n_station:
            if stations is not None:
                station_c = list(ds.spatial_unit).index(stations[station_ind])
                column = stations[station_ind]

            else:
                station_c = station_ind
                column = ds.spatial_unit[station_c] 
        else:
            column = 'All'
        for ind_metric,metric in enumerate(metrics) : 
            min_flow_i = min_flow if metric == 'mape' else 0 
            cbar_label = f"Percentage error" if metric == 'mape' else 'absolute error'
            if metric == 'previous_value':
                if station_ind<n_station:
                    #error = get_gain_from_naiv_model(Y_true[:,station_c:station_c+1,:],Preds[:,station_c:station_c+1,:],X[:,station_c:station_c+1,-1],min_flow,limit_percentage_error)
                    error = get_gain_from_mod1(real = Y_true[:,station_c:station_c+1,:],
                                               predict1 = X[:,station_c:station_c+1,-1],
                                               predict2 = Preds[:,station_c:station_c+1,:],
                                               min_flow=min_flow,metrics = ['mse'],acceptable_error= acceptable_error)
                    error = error['mse']
                else:
                    T,N,C = Y_true.size()
                    #error = get_gain_from_naiv_model(Y_true,Preds,X[:,:,-1],min_flow,limit_percentage_error)
                    error = get_gain_from_mod1(real =Y_true,
                                               predict1 = X[:,:,-1],
                                               predict2 = Preds,
                                               min_flow=min_flow,metrics = ['mse'],acceptable_error= acceptable_error)
                    error = error['mse']
                    error = error.reshape(T,N)     
                    error = error.mean(axis=1)
                cmap = 'RdYlBu'
                bool_reversed = True
                v_min,v_max = -limit_percentage_error,limit_percentage_error
            else:
                if station_ind<n_station:
                    error = error_along_ts(Preds[:,station_c:station_c+1,:],Y_true[:,station_c:station_c+1,:],metric,min_flow_i,normalize = False)
                else:
                    T,N,C = Y_true.size()
                    error = error_along_ts(Preds,Y_true,metric,min_flow_i,normalize = False)
                    error = error.reshape(T,N)     
                    error = error.mean(axis=1)
                cmap = 'YlOrRd'
                bool_reversed = False
                if metric == 'mape':
                    v_min,v_max = 0,50
                else:
                    #v_min,v_max = None,None
                    v_min,v_max = None,torch.quantile(error,0.95).item()

            # Build Matshow matrix : 
            df_agg = build_matrix_for_matshow(ds,column,training_mode,error,freq,index_matshow,columns_matshow)

            # Plotting : 
            plot_matshow(df_agg,column,metric,v_min,v_max,cmap,cbar_label,bool_reversed,axes,station_ind,ind_metric)      

    plt.tight_layout()
    plt.show()
    return fig,axes


def temporal_aggregation_of_attn_weight(attn_weights_reshaped,ds,training_mode,temporal_agg):
    ''' 
    Return the temporal aggregation of attn weights to visualise them 

    args:
    ------
    temporal_agg : choices ['hour','weekday','weekday_hour','weekday_hour_minutes']
    '''
    index_df = getattr(ds.tensor_limits_keeper,f"df_verif_{training_mode}").iloc[:,-1]
    df = pd.DataFrame(attn_weights_reshaped,index = index_df,columns = ds.spatial_unit)
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    if temporal_agg is not None:
        if temporal_agg == 'hour':
            df_agg = df.groupby([df.index.hour]).agg('mean')
            str_dates = list(df_agg.index.map(lambda x: f"{x:02d}"))

        elif temporal_agg == 'weekday':
            df_agg = df.groupby([df.index.weekday]).agg('mean')
            str_dates = list(df_agg.index.map(lambda x: weekdays[x]))

        elif temporal_agg == 'weekday_hour':
            df_agg = df.groupby([df.index.weekday,df.index.hour]).agg('mean')
            str_dates = list(df_agg.index.map(lambda x: f"{weekdays[x[0]]} {x[1]:02d}"))

        elif temporal_agg == 'weekday_hour_minute':
            df_agg = df.groupby([df.index.weekday,df.index.hour,df.index.minute]).agg('mean')
            str_dates = list(df_agg.index.map(lambda x: f"{weekdays[x[0]]} {x[1]:02d}:{x[2]:02d}"))
        else:
            raise NotImplementedError(f'Temporal aggregation {temporal_agg} has not been implemented')
        attn_weights_reshaped = df_agg.values  
    else:
        str_dates = list(df.index.strftime('%Y-%m-%d %H:%M'))
        attn_weights_reshaped = df.values

    return attn_weights_reshaped,str_dates

def get_y_size_from_temporal_agg(temporal_agg):
    if temporal_agg is not None:
        if temporal_agg == 'hour':
            y_size = 12
        elif temporal_agg == 'weekday':
            y_size = 7    
        elif temporal_agg == 'weekday_hour':
            y_size = 7*6
        elif temporal_agg == 'weekday_hour_minute':
            y_size = 7*10*2
        elif temporal_agg == 'date':
            y_size = 12
        else:
            raise NotImplementedError(f'Temporal aggregation {temporal_agg} has not been implemented')
    else:
        y_size = 7*6
    return(y_size)

def plot_attn_weight(trainer,nb_calendar_data,ds= None,training_mode = None,temporal_agg = None,save=None,stations= None, teporal_mha = False,layer = None):

    if teporal_mha:
        attn_model = trainer.model.core_model.output.temporal_agg.layers[layer].attention.sublayer.heads
        num_heads = len(attn_model)
    else:
        attn_model = trainer.model.netmob_vision.model
        num_heads = attn_model[0].attention.num_heads

    
    # Load Inputs : 
    X,Y,X_c,nb_contextual = trainer.load_all_inputs_from_training_mode(training_mode)

    # Init:
    nb_contextuals = len(X_c) - nb_calendar_data
    spatial_units = list(ds.spatial_unit)
    if stations is not None :
        nb_stations_to_plot = len(stations) 
    else :
        stations = list(ds.spatial_unit)
        nb_stations_to_plot = Y.size(1)
    num_cols = 4

    nb_rows = (nb_stations_to_plot*num_heads + num_cols - 1) // num_cols  
    y_size = get_y_size_from_temporal_agg(temporal_agg)
    #plt.figure(figsize=(5*num_cols,y_size))  
    plt.figure(figsize=(5*num_cols*max(1,nb_stations_to_plot//15),int(y_size*max(1,nb_contextuals//num_cols))))

    vmin,vmax = 0,min(1,1/(nb_contextuals/3)) 
    for station_i in range(nb_stations_to_plot):

        station_ind  = spatial_units.index(stations[station_i])

        enhanced_x,attn_weights = attn_model[station_ind](X[:,station_ind,:],X_c[station_ind+nb_calendar_data],x_known = None)
        if attn_weights.dim()==4:
            for attn_head_i in range(num_heads):
                attn_weights_reshaped = attn_weights[:,attn_head_i,:,:].squeeze(1).detach().cpu().numpy()  # Shape [B, P]
                # Temporal Aggregation of attn weight:
                attn_weights_reshaped,str_dates = temporal_aggregation_of_attn_weight(attn_weights_reshaped,ds,training_mode,temporal_agg)
                ax = plt.subplot(nb_rows, num_cols, station_i*num_heads+attn_head_i + 1)  # Créer un subplot
                im = ax.imshow(attn_weights_reshaped, cmap='hot', aspect='auto',vmin=vmin,vmax=vmax)
                plt.colorbar(im,label='Attention Weight',shrink = 0.25)
                
                if temporal_agg is None:
                    plt.title(f'Attention Weight head {attn_head_i}\nof station {station_ind} ({spatial_units[station_ind]}) \nfor each sample of the batch')
                    plt.ylabel('Samples')
                else:
                    plt.title(f'Mean Attention Weight head {attn_head_i}\nof station {station_i}({spatial_units[station_ind]}) \nby calendar class') 
                    plt.ylabel('Calendar class')
                plt.xlabel('Contextual time-series')


                num_samples, nb_contextual_on_plot_i = attn_weights_reshaped.shape
                plt.xticks(ticks=np.arange(nb_contextual_on_plot_i), labels=[f'Unit {i}' for i in range(nb_contextual_on_plot_i)], rotation=45)
                plt.yticks(ticks=np.arange(num_samples), labels=str_dates)

    plt.tight_layout()

    if save is not None:
        plt.savefig(f'{save}.pdf',format = 'pdf',bbox_inches='tight')
    plt.show()



def get_df_error(ds1,dic_error,metric,error_name,training_mode,dates = None):
    #Init
    if dates is None:
        df_verif = getattr(ds1.tensor_limits_keeper,f"df_verif_{training_mode}")
        dates = df_verif.iloc[:,-1]
    n_units = len(ds1.spatial_unit)
    # Get df of time-serie of a metric: 
    if metric == 'rmse':
        metric = 'mse'
    error_per_stations = dic_error[metric][error_name].reshape(-1,n_units)
    dict_for_df = {column: error_per_stations[:,i] for i,column in enumerate(ds1.spatial_unit)}
    dict_for_df.update({'datetime':dates})
    df_error_station = pd.DataFrame(dict_for_df)

    # Add calendar information : 
    df_error_station = add_calendar_columns(df_error_station)
    return df_error_station


def temporal_agg_for_matshow(df_error_station,column,index_matshow,metric = None):
    '''
    From pipeline.a df_error_station, return the temporal aggregation of the column
    args:
    -----
    df_error_station : pd.DataFrame with columns ['datetime',column]
    column : str, name of the column to aggregate
    index_matshow : str, type of aggregation to apply on the index of the df_error_station
    >>> e.g. 'hour', 'date', 'weekday', 'weekday_hour', 'weekday_hour_minute', 'daily_period', 'working_day_hour'
    '''

    df_error_station['new_hour'] = df_error_station['datetime'].dt.hour
    df_error_station['time'] = df_error_station['datetime'].dt.time
    s = df_error_station['datetime']
    df_error_station['is_weekday'] = is_weekday(s)
    df_error_station['is_weekday'] = df_error_station['is_weekday'].replace({True: 'Weekday', False: 'Weekend'})
    df_error_station['evening_peak'] = is_evening_peak(s)
    df_error_station['morning_peak'] =  is_morning_peak(s)


    if index_matshow == 'hour':
        df_agg = df_error_station[[column,'new_hour']].groupby(['new_hour']).mean()
    elif index_matshow == 'date':
        df_agg = df_error_station[[column,'date']].groupby(['date']).mean()
    elif index_matshow == 'hour_minute':
        df_agg = df_error_station[[column,'hour']].groupby(['hour']).mean()
    elif index_matshow == 'weekday':
        df_agg = df_error_station[[column,'weekday']].groupby(['weekday']).mean()
    elif index_matshow == 'weekday_hour':
        df_agg = df_error_station[[column,'weekday','new_hour']].groupby(['weekday','new_hour']).mean()
    elif index_matshow == 'weekday_hour_minute':
        df_agg = df_error_station[[column,'weekday','hour']].groupby(['weekday','hour']).mean()
    elif index_matshow == 'daily_period':
        df_agg_morning = df_error_station[df_error_station['morning_peak'] & df_error_station['is_weekday']][column].mean()
        df_agg_evening =  df_error_station[df_error_station['evening_peak']& df_error_station['is_weekday']][column].mean()
        df_agg_off_peak =  df_error_station[~df_error_station['evening_peak'] & 
                                    ~df_error_station['evening_peak'] &
                                    df_error_station['is_weekday']][column].mean()
        df_agg_all = df_error_station[column].mean()
        df_agg = pd.DataFrame({column:[df_agg_all,df_agg_morning,df_agg_evening,df_agg_off_peak]},index=['all_day','morning_peak','evening_peak','off_peak'])
    elif index_matshow == 'working_day_hour':
        working_days = df_error_station[df_error_station.is_weekday == 'Weekday'].copy()
        df_agg = working_days[[column,'new_hour']].groupby(['new_hour']).mean()
    elif index_matshow == 'working_weekday_weekend_hour':
        df_agg = df_error_station[[column,'is_weekday','new_hour']].groupby(['is_weekday','new_hour']).mean()
    else:
        raise NotImplementedError
    
    if metric is not None and metric == 'rmse':
        df_agg = np.sqrt(df_agg)
    return df_agg 

def get_df_mase_and_gains(ds1,dic_error,training_mode,temporal_agg,stations,dates = None):
    df_naiv_error = get_df_error(ds1,dic_error,metric = 'mae_naiv',error_name = 'error_naiv',training_mode=training_mode,dates = dates)
    df_mae_error1 = get_df_error(ds1,dic_error,metric = 'mae',error_name = 'error_pred1',training_mode=training_mode,dates = dates)
    df_mae_error2 = get_df_error(ds1,dic_error,metric = 'mae',error_name = 'error_pred2',training_mode=training_mode,dates = dates)

    df_mase1,df_mase2,df_gain21 = {},{},{}

    for column in stations: 
        naiv_error_agg = temporal_agg_for_matshow(df_naiv_error,column,temporal_agg)
        error_pred1_agg = temporal_agg_for_matshow(df_mae_error1,column,temporal_agg)
        error_pred2_agg = temporal_agg_for_matshow(df_mae_error2,column,temporal_agg)   
        mase1 = error_pred1_agg/naiv_error_agg
        mase2 = error_pred2_agg/naiv_error_agg

        gain_mase = 100*(mase2/mase1-1)

        df_mase1.update({column:mase1[column]})
        df_mase2.update({column:mase2[column]})
        df_gain21.update({column:gain_mase[column]})
    return df_gain21,df_mase1,df_mase2

def get_df_gains(ds1,dic_error,metric,training_mode,temporal_agg,stations,dates = None):
    df_error1 = get_df_error(ds1,dic_error,metric =metric,error_name = 'error_pred1',training_mode=training_mode,dates = dates)
    df_error2 = get_df_error(ds1,dic_error,metric =metric,error_name = 'error_pred2',training_mode=training_mode,dates = dates)
    df_gain21 = {}
    df_error_pred1_agg,df_error_pred2_agg = {},{}

    for column in stations: 
        error_pred1_agg = temporal_agg_for_matshow(df_error1,column,temporal_agg,metric)
        error_pred2_agg = temporal_agg_for_matshow(df_error2,column,temporal_agg,metric)   
        gain = 100*(error_pred2_agg/error_pred1_agg-1)

        df_gain21.update({column:gain[column]})
        df_error_pred1_agg.update({column:error_pred1_agg[column]})
        df_error_pred2_agg.update({column:error_pred2_agg[column]})
    return df_gain21,df_error_pred1_agg,df_error_pred2_agg




def set_attention_weights_agregated_per_daily_period(gdf,NetMob_attn_weights, 
                                                     station_i,head, mask, agg_iris_target_n=None,
                                                     dict_label2agg= None,list_correspondence=None,
                                                     kept_zones = None, contextual_dataset =None):
    if agg_iris_target_n is None:
        agg_iris_target_n = len(gdf)
    gdf_copy = gdf.copy()



    # --- Tackle Netmob contextual datasets:
    if list_correspondence is not None: 
        # -- Build dict_agg2label
        dict_agg2label = {}
        for k,v in dict_label2agg.items():
            if v not in dict_agg2label:
                dict_agg2label[v] = []
            dict_agg2label[v].append(k)
        # -- 
        for channel_spatial_unit in range(NetMob_attn_weights.size(-1)):
            init_labels = dict_agg2label[channel_spatial_unit]
            list_range = [range(k*agg_iris_target_n,(k+1)*agg_iris_target_n) for k in range(len(list_correspondence))]
            is_associated_to_an_unique_app = [all([label in range_agg_iris_target_n for label in init_labels]) for range_agg_iris_target_n in list_range]

            # If it's associated to an unique app: 
            assert sum(is_associated_to_an_unique_app) < 2, 'Issue with the discretisation'
            if sum(is_associated_to_an_unique_app) > 0:
                # Attention weight at head 'head' and station 'station_i'.  
                attn_weight_head_i = NetMob_attn_weights[:,head,station_i,:]  # [T,n_head,N,channel_spatial_units] -> [T,channel_spatial_units]

                # Fin the associated app: 
                app_tag_mode = list_correspondence[np.where(np.array(is_associated_to_an_unique_app) == True)[0][0]]

                # Find the associated zones 
                reduced_init_labels = [init_labels[i]%agg_iris_target_n for i in range(len(init_labels))]

                # Specifie daily period (morning peak on working day ...) :
                attn_weight_head_i = torch.index_select(attn_weight_head_i.detach().cpu(),0, torch.tensor(mask).long())

                # Add the attention weights to the gdf:
                gdf_copy.loc[reduced_init_labels,app_tag_mode] = attn_weight_head_i.mean(0)[channel_spatial_unit].detach().cpu()
                gdf_copy.loc[reduced_init_labels,f'{app_tag_mode}_channel_spatial'] = channel_spatial_unit
            else:
                if len(list_correspondence) == 2:
                    app_tag_mode = list_correspondence
                    print('mix of apps: ',init_labels)
                else:
                    raise NotImplementedError
        # ---
                
    # --- Tackle Bike-in or Bike-out contextual datasets: 
    elif kept_zones is not None:
        attn_weight_head_i = NetMob_attn_weights[:,head,station_i,:]  # [T,n_head,N,channel_spatial_units] -> [T,channel_spatial_units]
        attn_weight_head_i = torch.index_select(attn_weight_head_i.detach().cpu(),0, torch.tensor(mask).long())
        attn_weight_head_i = attn_weight_head_i.mean(0)  
        gdf_copy.loc[kept_zones,contextual_dataset] = attn_weight_head_i.numpy()
        # gdf.loc[:,contextual_dataset] = gdf.loc[:,contextual_dataset].fillna(-1)  # Fill NaN values with 0
        gdf_copy.loc[:,f"{contextual_dataset}_channel_spatial"] = gdf_copy.index
    # ---
    
    else:
        raise NotImplementedError(f'Contextual dataset {contextual_dataset} not implemented')


    return gdf_copy








def get_attn_weight_from_model(model,contextual_dataset):
    """
    Get the attention weights from the model, depending on the model architecture.
    """
    try: 
        NetMob_attn_weights = getattr(model.spatial_attn_poi,contextual_dataset).attn_weight
    except:
        try: 
            NetMob_attn_weights = getattr(model.core_model.output.ModuleContextualAttnLate,contextual_dataset).attn_weight
        except:
            NetMob_attn_weights = getattr(model.spatial_attn_poi,contextual_dataset).mha_list[-1].attn_weight
    return NetMob_attn_weights


def plot_avg_attn_weights(args,trainer_list,ds,temporal_aggs,training_mode,vmax_coeff=4):
    """ Plot the average attention weights for each contextual dataset.
    Args:
        trainer_list: List of Trainer to use for loading inputs and forward in model .
        ds: Dataset object
        training_mode: Mode of training (e.g., 'test','train', or 'valid').
    """
    dict_attn_weights = {}
    if hasattr(args,'contextual_kwargs') and len(args.contextual_kwargs) > 0:
        contextual_datasets = list(args.contextual_kwargs.keys())
        for contextual_dataset in contextual_datasets:
            NetMob_attn_weights = []
            for trainer in trainer_list:
                X,Y,Xc,nb_contextual = trainer.load_all_inputs_from_training_mode(training_mode)
                model = trainer.model
                model.eval()
                with torch.no_grad():
                    pred = model(X,Xc)
                    test_NetMob_attn_weights = get_attn_weight_from_model(model,contextual_dataset)

                NetMob_attn_weights.append(test_NetMob_attn_weights)
            NetMob_attn_weights = torch.stack(NetMob_attn_weights,0).mean(0)

            spatial_unit = ds.spatial_unit
            s_dates = ds.tensor_limits_keeper.df_verif_test.iloc[:,-1].reset_index(drop=True)
            plot_attn_weights(NetMob_attn_weights,s_dates,temporal_aggs,spatial_unit,city = ds.city,
                      vmax_coeff = vmax_coeff)
            dict_attn_weights[contextual_dataset] = NetMob_attn_weights
    return dict_attn_weights




def plot_heatmap(M,xlabel=None,ylabel=None, title=None,cmap='hot',figsize=(15, 15),vmin = None,vmax= None):
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = ax.imshow(M, cmap=cmap, interpolation='nearest', aspect='auto',vmin = vmin, vmax=vmax)
    if xlabel is not None:
        ax.set_xticks(range(len(xlabel)), labels=xlabel,rotation=45, ha="right", rotation_mode="anchor")
    if ylabel is not None:
        ax.set_yticks(range(len(ylabel)), labels=ylabel)

    fig.colorbar(heatmap,ax=ax)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()
        
def plot_attn_weights(NetMob_attn_weights,s_dates,
                      temporal_aggs,
                      spatial_unit,city = None,
                      vmax_coeff = 3):
    # ----- Find Indices related to specifics period of the days: 

    # Find the indices of the hours between 7 and 10 on torch tensor
    indices_morning = torch.tensor(get_temporal_mask(s_dates,temporal_agg = 'morning_peak',city=city)).long().detach().cpu()
    indices_evening = torch.tensor(get_temporal_mask(s_dates,temporal_agg = 'evening_peak',city=city)).long().detach().cpu()
    NetMob_attn_weights = NetMob_attn_weights.detach().cpu()  # Ensure the attention weights are on CPU and detached from the computation graph
    # -----

    # head = 0


    uniform_weight = 1/NetMob_attn_weights.size(-1)
    vmin = 0
    vmax = min(1,uniform_weight*vmax_coeff)

    for head in range(NetMob_attn_weights.size(1)):
        for temporal_agg in temporal_aggs:
            if temporal_agg == 'all_day':
                # -- Average Attention Weight : 
                average_attn_weight = NetMob_attn_weights.mean(0)   # [heads, stations, Iris]
                plot_heatmap(average_attn_weight[head],ylabel =spatial_unit,figsize = (15,7) ,title=f'Average Attention Weight throughout the day\n Head {head}',vmin=vmin,vmax=vmax)
            elif temporal_agg == 'morning_peak':
                # -- Morning Average Attention Weight : 
                morning_attn_weight = torch.index_select(NetMob_attn_weights, 0, indices_morning).mean(0)
                plot_heatmap(morning_attn_weight[head], title=f'Attention Weight during Morning (7:00 - 10:45)\n Head {head}',ylabel =spatial_unit,figsize = (15,7),vmin=vmin,vmax=vmax)
            elif temporal_agg == 'evening_peak':
                # -- Evening Attention Weight : 
                evening_attn_weight = torch.index_select(NetMob_attn_weights, 0, indices_evening).mean(0)
                plot_heatmap(evening_attn_weight[head], title=f'Attention Weight during evening (17:00 - 19:45)\n Head {head}',ylabel =spatial_unit,figsize = (15,7),vmin=vmin,vmax=vmax)
            else:
                raise NotImplementedError


if __name__ == '__main__':
    # Exemple with 'plot_coverage_matshow':
    range_dates = pd.date_range(start= "2019-9-30",end="2021-5-31",freq = '7D')
    data = pd.DataFrame(np.random.randint(50,size = (88,7)), index = range_dates)
    data.index = data.index.strftime('%Y-%m-%d')
    plot_coverage_matshow(data, log  = False, cmap = 'YlOrRd')

    

