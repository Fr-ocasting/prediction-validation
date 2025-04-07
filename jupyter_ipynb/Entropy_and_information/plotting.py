# Bokeh
from bokeh.plotting import figure,show
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar
from bokeh.transform import linear_cmap
from bokeh.colors import RGB
from bokeh.transform import transform
from bokeh.models import HoverTool

import sys
import os

current_path = notebook_dir = os.getcwd()
working_dir = os.path.abspath(os.path.join(current_path, '..','..'))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

from jupyter_ipynb.Entropy_and_information.utils import get_minmax


def visualize_TE(source,time = 't',explainable = 'x', target = 'y',TE = 'z',width=800,height=200,title = '',boolshow = True):
    # Figure Bokeh
    if boolshow: 
        output_notebook()
    p = figure(
        title= title,
        width=width,
        height=height,
        x_axis_label="Time (t)",
        y_axis_label="Amplitude"
    )

    # 1) Tracé de y(t) en rouge
    p.line(time, target, source=source, color="red", line_width=2, legend_label=f"{target}")

    # 2) ColorMapper pour x(t) en fonction de z (du plus clair au plus foncé)
    #    => plus z est grand, plus la couleur est sombre.
    min_te,max_te = get_minmax(source.data[TE])
    color_mapper = LinearColorMapper(
        palette="Viridis256",  # Palette continue
        low=min_te,
        high=max_te
    )

    # 3) Tracé de x(t) en points colorés selon z
    if type(explainable) != list:
        explainable = [explainable]
    for exp_var in explainable:
        p.scatter(
            time, exp_var,
            source=source,
            size=8,
            color=transform(TE, color_mapper),
            alpha=0.8,
            legend_label=exp_var
        )

    # 4) Ajout d'une ColorBar à droite
    color_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=12,
        width=8,
        location=(0, 0)
    )
    p.add_layout(color_bar, 'right')
    p.add_tools(HoverTool(tooltips=[(time, f"@{time}"), (TE, f"@{TE}")]))

    # (Optionnel) Aju
    # ster le comportement de la légende
    p.legend.click_policy = "hide"
    if boolshow: 
        show(p)

    return p 