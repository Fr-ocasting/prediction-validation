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


def plot_boxplot_on_metric(df, metric_i='mse', xaxis_label="App", legend_group='fold', width=1200, height=400, 
                            save_path=None):
    # Data preparation
    sdf = df.groupby("id")[metric_i].mean().sort_values()
    sorted_ids = sdf.index.tolist()

    df[f"{legend_group}_str"] = df[legend_group].astype(str)

    grp = df.groupby("id")[metric_i]
    
    # Correct the creation of the stats DataFrame
    # Let the quantile/min/max operations define the 'id' index, then reset it.
    stats = pd.DataFrame({
        "min_v": grp.min(),
        "q1": grp.quantile(0.25),
        "median_v": grp.quantile(0.50),
        "q3": grp.quantile(0.75),
        "max_v": grp.max()
    }).reset_index() # .reset_index() turns the 'id' index into a column.

    # Create a new source for each legend group
    sources = {group: ColumnDataSource(df[df[f"{legend_group}_str"] == group]) for group in df[f"{legend_group}_str"].unique()}
    box_sources = {group: ColumnDataSource(stats[stats["id"].isin(sources[group].data['id'])]) for group in df[f"{legend_group}_str"].unique()}

    # Use FactorRange to allow for dynamic x-axis
    p = figure(
        x_range=FactorRange(factors=sorted_ids),
        width=width,
        height=height,
        title=f"{metric_i} distribution per {xaxis_label} and per {legend_group}"
    )
    box_width = 0.2
    
    palette = Category10[len(df[f"{legend_group}_str"].unique())]
    legend_groups = sorted(df[f"{legend_group}_str"].unique())
    
    renderers = []
    # Loop to create both box plot and circle renderers for each group
    for i, group_name in enumerate(legend_groups):
        # Box plot for each group
        box_renderer = p.vbar(
                x="id", top="q3", bottom="q1", width=box_width,
                source=box_sources[group_name],
                fill_color=palette[i], fill_alpha=0.3, line_color="black"
            )
        segment_top_renderer = p.segment(
                x0="id", y0="max_v", x1="id", y1="q3",
                source=box_sources[group_name],
                line_width=1, line_color="black"
            )
        segment_bottom_renderer = p.segment(
                x0="id", y0="min_v", x1="id", y1="q1",
                source=box_sources[group_name],
                line_width=1, line_color="black"
            )
        median_renderer = p.hbar(
                y="median_v", height=0, left="id", right="id",
                source=box_sources[group_name],
                line_width=2, line_color="black"
            )

        # Circles for each group, now with a legend_label
        circle_renderer = p.circle(
                x="id", y=metric_i,
                source=sources[group_name],
                size=7,
                line_color="black",
                fill_color=palette[i],
                legend_label=group_name
            )

        # We need to group all renderers related to a legend item
        # to ensure that clicking the legend hides/shows everything together.
        group_renderers = [circle_renderer, box_renderer, segment_top_renderer, segment_bottom_renderer, median_renderer]
        renderers.append(group_renderers)
        
    p.xaxis.major_label_text_font_size = "10pt"
    p.xaxis.axis_label = xaxis_label 
    p.yaxis.axis_label = metric_i
    p.xaxis.major_label_orientation = np.pi/7
    p.legend.title = legend_group
    p.legend.click_policy = "hide"
    
    # Add a JavaScript callback to handle visibility
    callback = CustomJS(args=dict(renderers=renderers, x_range=p.x_range, original_factors=p.x_range.factors), code="""
        // Wait for the next tick to ensure legend state is updated
        setTimeout(function() {
            const active_factors = new Set();
            for (const group_renderers of renderers) {
                // Check the visibility of the first renderer in the group
                // to determine if the entire group is visible
                const is_visible = group_renderers[0].visible; 
                if (is_visible) {
                    // MODIFICATION: Use the circle renderer's data source to get IDs
                    // This is robust as the circle renderer always has an ID for each point
                    const circle_renderer = group_renderers[0];
                    if (circle_renderer.data_source.data.id) {
                        circle_renderer.data_source.data.id.forEach(id => active_factors.add(id));
                    }
                }
            }
            
            const new_factors = [];
            for (const f of original_factors) {
                if (active_factors.has(f)) {
                    new_factors.push(f);
                }
            }
            x_range.factors = new_factors;
        }, 0);
    """)

    # We need to assign the callback to each renderer within its legend group.
    # The Bokeh legend item is tied to the first renderer that has a legend_label.
    # In this case, it's the circle_renderer.
    for i, item in enumerate(p.legend.items):
        # We know the first renderer for each group is the circle_renderer which has the label.
        # Attach the callback to it.
        item.renderers[0].js_on_change('visible', callback)

    output_notebook()
    show(p)

    if save_path is not None:
        reset_output()
        output_file(save_path)
        save(p)
        reset_output()

# def plot_boxplot_on_metric(df,metric_i='mse',xaxis_label = "App", legend_group = 'fold', width=1200, height=400,save_path=f"MSE_distribution_per_app_and_per_fold.html"):
#     sdf = df.groupby("id")[metric_i].mean().sort_values()
#     sdf_ids = sdf.index.tolist()

#     df[f"{legend_group}_str"] = df[legend_group].astype(str)

#     grp = df.groupby("id")[metric_i]
#     q1 = grp.quantile(0.25)
#     q2 = grp.quantile(0.50)
#     q3 = grp.quantile(0.75)
#     mn = grp.min()
#     mx = grp.max()
#     stats = pd.DataFrame({
#         "id": q1.index,
#         "min_v": mn.values,
#         "q1": q1.values,
#         "median_v": q2.values,
#         "q3": q3.values,
#         "max_v": mx.values
#     })
#     source_box = ColumnDataSource(stats)
#     source_points = ColumnDataSource(df)

#     sdf = df.groupby("id")[metric_i].mean().sort_values()
#     sdf_ids = sdf.index.tolist()

#     p = figure(
#         x_range=sdf_ids, #sorted(df["id"].unique()),
#         width=width, height=height,
#         title=f"{metric_i} distribution per {xaxis_label} and per {legend_group}"
#     )
#     box_width = 0.2

#     p.segment("id","max_v","id","q3", source=source_box, line_width=1,line_color = 'black')
#     p.segment("id","min_v","id","q1", source=source_box, line_width=1,line_color = 'black')
#     p.vbar("id", box_width, "median_v", "q3", source=source_box, line_width=2,fill_color = 'grey',fill_alpha = 0.3,line_color = 'black')
#     p.vbar("id", box_width, "q1", "median_v", source=source_box, line_width=2,fill_color = 'grey',fill_alpha = 0.3,line_color = 'black')
#     #p.rect("id","median_v", box_width, 0, source=source_box)

#     palette = Category10[len(df[f"{legend_group}_str"].unique())]
#     p.circle(
#         x="id", y=metric_i,
#         source=source_points,
#         size=7,
#         line_color="black",
#         fill_color=factor_cmap(f"{legend_group}_str", palette=palette, factors=df[f"{legend_group}_str"].unique()),
#         legend_group=f"{legend_group}_str",
        
#     )
#     p.xaxis.major_label_text_font_size = "10pt"
#     p.xaxis.axis_label = xaxis_label 
#     p.yaxis.axis_label = metric_i
#     p.xaxis.major_label_orientation = np.pi/7
#     p.legend.title = legend_group
#     # Hide legend group one by one when clicking on it
#     p.legend.click_policy="hide"  # this one hide the entire legend when clicking on it

    
#     output_notebook()
#     show(p)

#     if save_path is not None:
#         reset_output()
#         output_file(save_path)
#         save(p)
#         reset_output()