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

    plt.show()


if __name__ == '__main__':
    # Exemple with 'plot_coverage_matshow':
    range_dates = pd.date_range(start= "2019-9-30",end="2021-5-31",freq = '7D')
    data = pd.DataFrame(np.random.randint(50,size = (88,7)), index = range_dates)
    data.index = data.index.strftime('%Y-%m-%d')
    plot_coverage_matshow(data, log  = False, cmap = 'YlOrRd')

    