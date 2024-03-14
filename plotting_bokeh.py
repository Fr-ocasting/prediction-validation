from bokeh.plotting import figure, show, output_file, save,output_notebook
from bokeh.models import HoverTool, ColumnDataSource
import torch

def d2day(d):
    days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    return(days[d])


def hm2hour(hm):
    if hm[0][1] == 0:
        m1 = 'OO'
    else:
        m1 = str(hm[0][1])
    
    if hm[1][1] == 0:
        m2 = 'OO'
    else:
        m2 = str(hm[1][1])
    
    return f"{hm[0][0]}:{m1} - {hm[1][0]}:{m2}h"

def plot_latent_space(trainer,data_loader,args,dic_class2rpz,name_save):
    # Get unique labels : 
    data = [[x_b,y_b,t_b] for  x_b,y_b,t_b in data_loader['test']]
    X_test,Y_test,T_test = torch.cat([x_b for [x_b,_,_] in data]),torch.cat([y_b for [_,y_b,_] in data]),torch.cat([t_b for [_,_,t_b] in data])
    labels = T_test.unique().long().to(args.device)
    # ...

    # Plot Each Point : 
    X,Y,Legend = [],[],[]
    trainer.model.eval()
    with torch.no_grad():
        for label in labels:
            x,y = trainer.model.Tembedding(label).cpu().detach().numpy()
            rpz = dic_class2rpz[label.item()]
            X.append(x)
            Y.append(y)
            Legend.append(f"{d2day(rpz[0][0])} {hm2hour(rpz[1])}")
    # ...
    

    # Assurons-nous que la sortie s'affiche dans le notebook
    output_notebook()


    # Création du ColumnDataSource à partir des données
    source = ColumnDataSource(data=dict(x=X, y=Y, legend=Legend))

    # Création de la figure
    p = figure(title="Visualisation of latent space", x_axis_label='dimension 1', y_axis_label='dimension 2')

    # Ajout des points à la figure
    p.circle('x', 'y', size=15, source=source)

    # Configuration du HoverTool pour afficher la légende lors du survol
    hover = HoverTool()
    hover.tooltips =[("Time Slot: ", "@legend")] #[("Legend", "@legend"), ("(x,y)", "($x, $y)")]
    p.add_tools(hover)

    # Affichage de la figure

    show(p)
    output_file(f"{name_save}.html")
    save(p)