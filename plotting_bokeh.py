import numpy as np 
import os 

from bokeh.plotting import figure, show, output_file, save,output_notebook
from bokeh.models import ColumnDataSource, Toggle, CustomJS,HoverTool, Legend
from bokeh.layouts import layout,row,column

import torch
from paths import save_folder 
from PI_object import PI_object
# ...

def generate_bokeh(trainer,data_loader,dataset,Q,args,dic_class2rpz,trial_id,trial_save,station=0,show_figure = False):
    pi,pi_cqr,p1 = plot_prediction(trainer,dataset,Q,args,station = station)
    p2 = plot_loss(trainer)
    
    if args.time_embedding:
        p3 = plot_latent_space(trainer,data_loader,args,dic_class2rpz,station)
    else:
        p3 = None

    save_dir = f'{save_folder}plot/{trial_id}/'
    combine_bokeh(p1,p2,p3,save_dir,trial_save,show_figure)

    return(pi,pi_cqr)


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

def plot_latent_space(trainer,data_loader,args,dic_class2rpz,station):
    # Get unique labels : 
    data = [[x_b,y_b,t_b] for  x_b,y_b,t_b in data_loader['test']]
    T_test = torch.cat([t_b for [_,_,t_b] in data])
    labels = T_test.unique().long().to(args.device)
    # ...

    # Plot Each Point : 
    X,Y,Legend = [],[],[]
    morning_peak_x,morning_peak_y,morning_peak_legend = [],[],[]
    evening_peak_x,evening_peak_y,evening_peak_legend =[],[],[]
    night_x,night_y,night_legend = [],[],[]
    trainer.model.eval()
    with torch.no_grad():
        for label in labels:
            # According to the version I'm using : 
            try:
                embeded_vector = trainer.model.Tembedding(label)
            except:
                embeded_vector = trainer.model.TE.Tembedding(label)
            # ...
            
            n = len(embeded_vector.size())
            embeded_vector = embeded_vector.cpu().detach().numpy()
            if n < 3:
                x,y = embeded_vector[0],embeded_vector[1]
            else: 
                x,y = embeded_vector[0,station,0],embeded_vector[0,station,1]
            rpz = dic_class2rpz[label.item()]
            X.append(x)
            Y.append(y)
            Legend.append(f"{d2day(rpz[0][0])} {hm2hour(rpz[1])}")

            h1,h2 = rpz[1][0][0],rpz[1][1][0]
            # Morning Peak
            if (h1 > 6) & (h1 < 10) & (h2 > 6) & (h2 < 10):
                morning_peak_x.append(x)
                morning_peak_y.append(y)
                morning_peak_legend.append(f"{d2day(rpz[0][0])} {hm2hour(rpz[1])}")

            # Evening Peak
            if (h1 > 15) & (h1 < 20) & (h2 > 15) & (h2 < 20):
                evening_peak_x.append(x)
                evening_peak_y.append(y)
                evening_peak_legend.append(f"{d2day(rpz[0][0])} {hm2hour(rpz[1])}")

            # Night
            if (h1 > 0) & (h1 < 6) &  (h2 > 0) & (h2 < 6):
                night_x.append(x)
                night_y.append(y)
                night_legend.append(f"{d2day(rpz[0][0])} {hm2hour(rpz[1])}")

    # ...
    

    # Assurons-nous que la sortie s'affiche dans le notebook
    # output_notebook()


    # Création du ColumnDataSource à partir des données
    source_total = ColumnDataSource(data=dict(x=X, y=Y, legend=Legend))
    source_morning = ColumnDataSource(data=dict(x=morning_peak_x, y=morning_peak_y, legend=morning_peak_legend))
    source_evening = ColumnDataSource(data=dict(x=evening_peak_x, y=evening_peak_y, legend=evening_peak_legend))
    source_night =  ColumnDataSource(data=dict(x=night_x, y=night_y, legend=night_legend))

    # Création de la figure
    p = figure(title="Visualisation of latent space", x_axis_label='dimension 1', y_axis_label='dimension 2')

    # Ajout des points à la figure
    render_total = p.scatter('x', 'y', size=15, source=source_total, visible=True)
    render_morning = p.scatter('x', 'y', size=15, source=source_morning, color='red', visible=False)
    render_evening = p.scatter('x', 'y', size=15, source=source_evening, color='yellow', visible=False)
    render_night = p.scatter('x', 'y', size=15, source=source_night, color='black', visible=False)

    # Configuration du HoverTool
    hover = HoverTool()
    hover.tooltips = [("Time Slot: ", "@legend")]
    p.add_tools(hover)

    # Ajout de ToggleButtons
    toggle_all =  Toggle(label="All labels", button_type="success", active=True)
    toggle_morning = Toggle(label="Morning Peak on working days", button_type="success", active=False)
    toggle_evening = Toggle(label="Evening Peak on working days", button_type="success", active=False)
    toggle_night = Toggle(label="Night", button_type="success", active=False)

    # CustomJS pour gérer la visibilité
    toggle_all.js_on_click(CustomJS(args=dict(render=render_total), code="""
        render.visible = !render.visible;
    """))
    toggle_morning.js_on_click(CustomJS(args=dict(render=render_morning), code="""
        render.visible = !render.visible;
    """))

    toggle_evening.js_on_click(CustomJS(args=dict(render=render_evening), code="""
        render.visible = !render.visible;
    """))

    toggle_night.js_on_click(CustomJS(args=dict(render=render_night), code="""
        render.visible = !render.visible;
    """))

    # Création d'une mise en page avec les boutons et le graphique
    l = layout([
        [toggle_all, toggle_morning, toggle_evening, toggle_night],
        [p]
    ])
    
    return(l)

def plot_loss(trainer,location = "top_right"):
    train_loss,valid_loss = trainer.train_loss, trainer.valid_loss

    # Ajout des données à la première figure
    if len(valid_loss) > 0:
        p = figure(title="Loss over Time", x_axis_label='Epochs', y_axis_label='Loss', width=900, height=400)
        p.add_layout(Legend(), 'right')

        p.line(np.arange(len(valid_loss)), valid_loss, 
            legend_label=f"Validation loss:  {'{:.4f}'.format(valid_loss[-1])}",
            line_width=2, color="blue")
        p.line(np.arange(len(train_loss)), train_loss, 
            legend_label= f"Training loss: {'{:.4f}'.format(train_loss[-1])}", 
            line_width=2, color="green")

        # Configuration des légendes
        #p.legend.location = location
    else:
        p = None
    return(p)


def plot_prediction(trainer,dataset,Q,args,station = 0, location = "top_right"):
    
    (preds,Y_true,T_labels) = trainer.testing(dataset)
    if len(preds.size()) == 2:
        preds = preds.unsqueeze(1)
    # ...

    # PI
    if preds.size(-1) > 1:
        pi = PI_object(preds,Y_true,alpha = args.alpha, type_calib = 'classic')     # PI 'classic' :
        pi_cqr = PI_object(preds,Y_true,alpha = args.alpha, Q = Q, type_calib = 'CQR',T_labels = T_labels)      # PI 'CQR' 
        # str legend
        str_picp,str_mpiw = f"{'{:.2%}'.format(pi.picp)}" , f"{'{:.2f}'.format(pi.mpiw)}"
        str_picp_cqr, str_mpiw_cqr = f"{'{:.2%}'.format(pi_cqr.picp)}" , f"{'{:.2f}'.format(pi_cqr.mpiw)}"
        str_pi_alpha = f"{'{:.2f}'.format(1-args.alpha)}%"
        # ...
        title = 'Prediction Intervals'
    
        n = len(pi_cqr.upper)

    else:
        title = 'Ponctual Prediction'
        pi,pi_cqr = None, None
    

    p = figure(title=title,x_axis_type='datetime', x_axis_label='Time', y_axis_label='Demand volume', width=900, height=400)
    p.add_layout(Legend(), 'right')
    
    if preds.size(-1)>1:
        # PI bands     
        p.line(dataset.df_verif_test.iloc[:,-1], pi_cqr.upper[:,station,0].cpu().numpy(), 
            legend_label=f"PI \n PICP: {str_picp_cqr} \n MPIW: {str_mpiw_cqr}", 
            line_dash="dashed", line_width=1, color="green")
        p.line(dataset.df_verif_test.iloc[:,-1], pi_cqr.lower[:,station,0].cpu().numpy(), line_dash="dashed", line_width=1, color="green")
        # ...
        
        # Quantile Band
        p.line(dataset.df_verif_test.iloc[:,-1], pi.upper[:,station,0].cpu().numpy(), 
            legend_label=f"Quantile  {args.alpha/2} - {1-args.alpha/2} \n PICP: {str_picp} \n MPIW: {str_mpiw}", 
            line_dash="dashed", line_width=1, color="red")
        p.line(dataset.df_verif_test.iloc[:,-1], pi.lower[:,station,0].cpu().numpy(),line_dash="dashed", line_width=1, color="red")    
        # ...

    else:
        # Predicted Values
        p.line(dataset.df_verif_test.iloc[:,-1], preds[:,station,0].cpu().numpy(), 
            legend_label=f"Prediction", 
            line_dash="dashed", line_width=1, color="red")

    
    # True Value: 
    p.line(dataset.df_verif_test.iloc[:,-1], Y_true[:,station,0].cpu().numpy(), legend_label="True Value", line_width=2, color="blue")
    # ...
    
    #p.legend.location = location

    return(pi,pi_cqr,p)


def combine_bokeh(p1,p2,p3,save_dir,trial_save,show_figure):
    # Affichage côte à côte
    if p2 is not None:
        l = column(p1, p2)
    else:
        l = p1
    if p3 is not None:
        l = row(l,p3)
    # Affichage de la figure
    if show_figure:
        show(l)
    # Pour sauvegarder en HTML (assurez-vous de mettre à jour 'name_save' avec votre nom de fichier désiré)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_file(f"{save_dir}{trial_save}.html")
    save(l)