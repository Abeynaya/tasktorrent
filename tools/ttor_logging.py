import argparse
import pandas as pd
import networkx as nx

from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Slider, LinearColorMapper, CategoricalColorMapper
from bokeh.palettes import Viridis6, Category10_3

def make_nodes(files):
    df = pd.concat([pd.read_csv(f, header=None, names=["what", "time_start","time_end"]) for f in files], axis=0)
    df = df[df['what'].str.contains(">run>")]
    df = df[~df['what'].str.contains("intern")]
    cols = df['what'].str.split(pat=">",expand=True)
    df['task'] = cols[2]
    mint = min(df['time_end'])
    df['time'] = df['time_end'] - mint
    df['kind'] = [s.split('_')[0] for s in df['task']]
    df = df.set_index('task')
    return df

def make_edges(files):
    df = pd.concat([pd.read_csv(f, header=None, names=["start","end"]) for f in files], axis=0)
    df = df.reindex()
    return df

def make_graph(nodes, edges):
    G = nx.DiGraph()
    for index, row in nodes.iterrows():
        G.add_node(index)
    for index, row in edges.iterrows():
        G.add_edge(row['start'], row['end'])
    return G

def compute_pos(graph):
    print("Computing layout. This may take a while...")
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    print("Done with layout")
    return pos

def add_pos(nodes, edges):
    G = make_graph(nodes, edges)
    pos = compute_pos(G)
    nodes['x'] = [pos[n][0] for n in nodes.index]
    nodes['y'] = [pos[n][1] for n in nodes.index]

def make_dag_figure(nodes, edges, show_edges):

    # Add positions
    add_pos(nodes, edges)
    posx = nodes['x']
    posy = nodes['y']
    tt   = nodes['time']
    (minx,maxx) = (min(posx),max(posx))
    (miny,maxy) = (min(posy),max(posy))

    # Create figure
    plot = figure(title="Task graph", x_range=(minx-10,maxx+10), y_range=(miny-10,maxy+10))

    # Nodes
    dV = nodes
    V = ColumnDataSource(dV)

    TOOLTIPS = [
    
        ("task", "@task"),
        ("t", "@time"),
    ]   
    KINDS = list(pd.unique(nodes['kind']))

    color_mapper = CategoricalColorMapper(palette=Category10_3, factors=KINDS)
    color = {'field': 'kind', 'transform': color_mapper}
    model_nodes = plot.circle('x', 'y', source=V, color=color, alpha=1.0, size=4)

    # Edges
    if show_edges:
        dE = pd.DataFrame({
            'xs':[ [posx[e['start']],posx[e['end']]] for index, e in edges.iterrows() ],
            'ys':[ [posy[e['start']],posy[e['end']]] for index, e in edges.iterrows() ],
        })
        E = ColumnDataSource(dE)
        model_edges = plot.multi_line('xs', 'ys', source=E, color='black', line_width=1)

    # Around
    plot.add_tools(HoverTool(renderers=[model_nodes], tooltips=TOOLTIPS))
    plot.axis.visible = False
    plot.grid.visible = False
    
    # Slider
    maxt = max(tt)
    slider = Slider(start=0, end=maxt, step=maxt/25, value=maxt, title='Time (ms.)')
    def slider_callback(attr, old, new):
        new_max_t = new
        print(f"New time bound is {new_max_t}")
        new_dV = dV[dV['time'] <= new_max_t]
        model_nodes.data_source.data = new_dV
        if show_edges:
            new_dE = pd.DataFrame({
                'xs':[ [posx[e['start']],posx[e['end']]] for index, e in edges.iterrows() if tt[e[1]] <= new_max_t ],
                'ys':[ [posy[e['start']],posy[e['end']]] for index, e in edges.iterrows() if tt[e[1]] <= new_max_t ],
            })
            model_edges.data_source.data = new_dE
    slider.on_change('value', slider_callback)

    return plot, slider