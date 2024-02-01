import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axes_grid1
import pandas as pd
import numpy as np

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

    