import pandas as pd
import numpy as np 
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
from bokeh.plotting import figure 
from bokeh.models import ColumnDataSource, Toggle, CustomJS,HoverTool, Legend,Model
from bokeh.layouts import layout,row,column
from bokeh.resources import CDN
from bokeh.io import reset_output,show, output_file, save,output_notebook


def plot_boxplot_on_metric(df,metric_i='mse',save_path=f"MSE_distribution_per_app_and_per_fold.html"):
    sdf = df.groupby("id")[metric_i].mean().sort_values()
    sdf_ids = sdf.index.tolist()

    if not 'fold_str' in df.columns: 
        df["fold_str"] = df["fold"].astype(str)

    grp = df.groupby("id")[metric_i]
    q1 = grp.quantile(0.25)
    q2 = grp.quantile(0.50)
    q3 = grp.quantile(0.75)
    mn = grp.min()
    mx = grp.max()
    stats = pd.DataFrame({
        "id": q1.index,
        "min_v": mn.values,
        "q1": q1.values,
        "median_v": q2.values,
        "q3": q3.values,
        "max_v": mx.values
    })
    source_box = ColumnDataSource(stats)
    source_points = ColumnDataSource(df)

    sdf = df.groupby("id")[metric_i].mean().sort_values()
    sdf_ids = sdf.index.tolist()

    p = figure(
        x_range=sdf_ids, #sorted(df["id"].unique()),
        width=1200, height=400,
        title=f"{metric_i} distribution per app and per folds"
    )
    box_width = 0.2

    p.segment("id","max_v","id","q3", source=source_box, line_width=1,line_color = 'black')
    p.segment("id","min_v","id","q1", source=source_box, line_width=1,line_color = 'black')
    p.vbar("id", box_width, "median_v", "q3", source=source_box, line_width=2,fill_color = 'grey',fill_alpha = 0.3,line_color = 'black')
    p.vbar("id", box_width, "q1", "median_v", source=source_box, line_width=2,fill_color = 'grey',fill_alpha = 0.3,line_color = 'black')
    #p.rect("id","median_v", box_width, 0, source=source_box)

    palette = Category10[len(df["fold_str"].unique())]
    p.circle(
        x="id", y=metric_i,
        source=source_points,
        size=7,
        line_color="black",
        fill_color=factor_cmap("fold_str", palette=palette, factors=df["fold_str"].unique()),
        legend_group="fold_str"
    )

    p.xaxis.axis_label = "App"
    p.yaxis.axis_label = metric_i
    p.xaxis.major_label_orientation = np.pi/2
    p.legend.title = "Fold"
    output_notebook()
    show(p)

    if save_path is not None:
        reset_output()
        output_file(save_path)
        save(p)
        reset_output()