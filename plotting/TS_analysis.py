from bokeh.plotting import figure
from bokeh.models import Legend,RangeTool

def plot_subway_patterns(df,Metro_A_stations,palette,width=1500, height=600,title=f'Trafic Volume by stations'):

    p = figure(x_axis_type="datetime", 
               title=title,
               width=width,height=height)

    legend_it = []


    for k,station in enumerate(Metro_A_stations):
        c = p.line(x=df.index, line_width = 2.5, y=df[station], alpha=0.8,  #legend_label = f'{station}',
                   color=palette[int(k*(256-1)/(len(Metro_A_stations)+1))]
                  )#,muted_color=color, muted_alpha=0.2
        legend_it.append((station, [c]))
    
    p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
    legend = Legend(items=legend_it)
    legend.click_policy="hide"
    p.add_layout(legend, 'right')
    
    return(p)

def drag_selection_box(df,p1,p2=None,width=1500, height=150):

    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    width=width,
                    height=height, 
                    y_range=p1.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p1.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line(x=df.index, y=df.mean(axis=1))
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    
    # Sync x_range for both plots:
    if p2 is not None:
        p2.x_range = p1.x_range
    return(select)


