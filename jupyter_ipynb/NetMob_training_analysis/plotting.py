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
from bokeh.transform import dodge

def plot_boxplot_on_metric(df, metric_i='mse', xaxis_label="App", legend_group='fold', width=1200, height=400, 
                            save_path=None,
                            palette= None,
                            legend_groups = None,
                            title = None,
                            bool_show=True
                            ):
    
    if title is None:
        title = f"{metric_i} distribution per {xaxis_label} and per {legend_group}"
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
        "max_v": grp.max(),
        "mean_v": grp.mean()
    }).reset_index() # .reset_index() turns the 'id' index into a column.

    # Create a new source for each legend group
    sources = {group: ColumnDataSource(df[df[f"{legend_group}_str"] == group]) for group in df[f"{legend_group}_str"].unique()}
    box_sources = {group: ColumnDataSource(stats[stats["id"].isin(sources[group].data['id'])]) for group in df[f"{legend_group}_str"].unique()}

    # Use FactorRange to allow for dynamic x-axis
    p = figure(
        x_range=FactorRange(factors=sorted_ids),
        width=width,
        height=height,
        title=title
    )
    box_width = 0.2

    if palette is None:
        palette = Category10[max(3,len(df[f"{legend_group}_str"].unique()))]
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
        median_renderer = p.segment(
                x0=dodge("id", -box_width/2, range=p.x_range), y0="median_v",
                x1=dodge("id", box_width/2, range=p.x_range), y1="median_v",
                source=box_sources[group_name],
                line_width=2, line_color="black"
            )

        mean_renderer = p.segment(
                x0=dodge("id", -box_width/2, range=p.x_range), y0="mean_v",
                x1=dodge("id", box_width/2, range=p.x_range), y1="mean_v",
                source=box_sources[group_name],
                line_width=2, line_color="black", line_dash="dashed"
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
        group_renderers = [circle_renderer, box_renderer, segment_top_renderer, segment_bottom_renderer, median_renderer,mean_renderer]
        renderers.append(group_renderers)
        
    p.xaxis.major_label_text_font_size = "10pt"
    p.xaxis.axis_label = xaxis_label 
    p.yaxis.axis_label = metric_i
    p.xaxis.major_label_orientation = np.pi/3 #np.pi/7
    p.legend.title = legend_group
    p.legend.click_policy = "hide"

    p.add_layout(p.legend[0], 'right')
    
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
    for i, item in enumerate(p.legend[0].items):
        # We know the first renderer for each group is the circle_renderer which has the label.
        # Attach the callback to it.
        item.renderers[0].js_on_change('visible', callback)



    if save_path is not None:
        reset_output()
        output_file(save_path)
        save(p)
        reset_output()

    if bool_show:
        output_notebook()
        show(p)
    else:
        return p



