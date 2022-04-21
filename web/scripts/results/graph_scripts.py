
from operator import lt
import networkx as nx
import re
import numpy as np
import pandas as pd
from pathlib import Path

####################
###   Network  ####
####################
def create_plot(table=None, lmethod=["R",], log_hr = None):
    ##Create Net From EdgeList##
    G = nx.from_pandas_edgelist(

        table, "Mir","Gene",
        edge_attr=lmethod
        )

    ##Add tag for Gene and MIR##
    patMir = re.compile("^hsa-...-*")
    MIR, GENE = "blue","red"
    CIRC, SQR = "circle", "triangle"
    for node in G.nodes(data=True):
        node_data = (MIR, SQR) if patMir.match(node[0]) else (GENE, CIRC)
        G.nodes[node[0]]['color'] = node_data[0]
        G.nodes[node[0]]['name'] = node[0]
        G.nodes[node[0]]['marker'] = node_data[1]

        if log_hr is not None:
            #print(log_hr)
            try:
                from numpy import inf
                hr = np.log2(log_hr.loc[node[0],:].tolist()[1])

                G.nodes[node[0]]['HR'] = -1 if hr == -inf else 1 if hr == inf else hr
            except Exception as error:
                print(error)
                #print(log_hr.loc[node[0],:].tolist())
                G.nodes[node[0]]['HR'] = 0
                pass
        else:
            G.nodes[node[0]]['HR'] = 0
            
    return G


def cytoscape_elements_from_table(table, lmethod=["R",], log_hr = None):
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import matplotlib.pyplot as plt

    if log_hr is not None:
        cm = plt.get_cmap('PuOr') 
        cNorm  = colors.Normalize(vmin=-1,vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    #print(nodes)

    G = create_plot(table, lmethod=lmethod, log_hr = log_hr)
    print("3")
    #pos = nx.fruchterman_reingold_layout(G, iterations=2000, threshold=1e-10)

    nodes = [
    {
        'data': {'id': node[0], 'label': node[0], 'HR':  node[1]['HR'], 'color': list(int(x) for x in scalarMap.to_rgba(node[1]['HR'],  bytes=True)[0:3]) if log_hr is not None else (0,64,255)},
        #'position': {'x': 2000*pos[node[0]][0], 'y': 2000*pos[node[0]][1]},
        'classes': node[1]['marker'] # Multiple classes
        #'locked': 'FALSE'
    }
    for node in G.nodes(data=True)
    ]


    edges = []
    #print(lmethod)
    table[["Gene","Mir",]+lmethod].apply(lambda row: \
            edges.append({'data': {'source': row["Gene"], 'target': row["Mir"], 'weight': row[lmethod[0]], 'width': "%fpx"%(float(10*abs(row[lmethod[0]]))), 
            }}),\
            axis = 1)

    elements = nodes + edges

    #print(elements)
    return elements

 
def cytoscape_network(table, lmethod=["R",], log_hr = None):
    print("1")

    elements = cytoscape_elements_from_table(table, lmethod=lmethod, log_hr = log_hr)
    print("5")

    stylesheet=[
        # Group selectors
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'text-background-color': 'grey',
                'text-background-opacity': "0.5",
                'background-color': 'data(color)',
                "border-color": "black",
                "border-opacity": "1",
                "border-width": "1px"

                #'size': 50
            }
        },

        # Class selectors
        {
            'selector': '.red',
            'style': {
                'background-color': 'red',
                'line-color': 'red'
            }
        },
        {
            'selector': '.triangle',
            'style': {
                'shape': 'triangle'
            }
        },
       # Edge Color

        {
            'selector': '[weight < 0]',
            'style': {
                'line-color': 'blue'
             }
        },
        {
            'selector': '[weight > 0]',
            'style': {
                'width':  'data(width)',
                'line-color': 'red'
            }
        }
  ]
    

    return elements, stylesheet

####################
###   HEATMAPS  ####
####################
#https://plotly.com/python/builtin-colorscales/
def plotly_heat(matrix, colorscale = 'RdBu_r', zmin = None, zmax = None, zmid = 0):
    #https://stackoverflow.com/questions/36846395/embedding-a-plotly-chart-in-a-django-template
    import plotly.offline as opy
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import pandas as pd
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
    }}

    ### get data
    data = matrix
    #print(data)
    #data = data.iloc[0:5,0:8]
    data_array = pd.DataFrame(data)
    labels_mir = data.index.tolist()
    labels_gene = data.columns.tolist()

    if data.shape[0] == 1 or data.shape[1] == 1:
        data_array = pd.DataFrame(data.sort_values(by=data.columns.tolist()[0]))
        fig = go.Figure(go.Heatmap(
                x = labels_gene,
                y = labels_mir,
                z = data_array,
                colorscale = colorscale,
                zmin = zmin,
                zmax = zmax,
                zmid = zmid,
                colorbar = {'x': -.18, 'len': 0.8}))

    else:
        # Initialize figure by creating upper dendrogram
        fig = ff.create_dendrogram(data_array.transpose(), orientation='bottom', labels=labels_gene)
        for i in range(len(fig['data'])):
            fig['data'][i]['yaxis'] = 'y2'

        # Create Side Dendrogram
        dendro_side = ff.create_dendrogram(data_array, orientation='right',labels=labels_mir)
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'

        # Add Side Dendrogram Data to Figure
        for data in dendro_side['data']:
            fig.add_trace(data)

        # Create Heatmap
        mir = dendro_side['layout']['yaxis']['ticktext']
        gene = fig['layout']['xaxis']['ticktext']
        data_dist = data_array.loc[mir,gene]

        heatmap = [
            go.Heatmap(
                x = gene,
                y = mir,
                z = data_dist,
                colorscale = colorscale,
                zmin = zmin,
                zmax = zmax,
                zmid = zmid,
                colorbar = {'x': -.1, 'len': 0.8})
        ]

        heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
        heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']


        # Add Heatmap Data to Figure
        for data in heatmap:
            fig.add_trace(data)

        # Edit Layout
        
        fig.update_layout({'width':1240, 'height':1240,
                                'showlegend':False, 'hovermode': 'closest',
                                })
                                
        # Edit xaxis
        fig.update_layout(xaxis={'domain': [.15, 1],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'ticks':""})
        # Edit xaxis2
        fig.update_layout(xaxis2={'domain': [0, .15],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks':""})

        # Edit yaxis
        fig.update_layout(yaxis={'domain': [0, .85],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'ticktext': mir,
                                        'tickvals': heatmap[0]['y'],
                                        'showticklabels': True,
                                        'side':'right',})
        # Edit yaxis2
        fig.update_layout(yaxis2={'domain':[.825, .975],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks':"",
                                        'automargin': True})

    fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)',
					plot_bgcolor = 'rgba(0,0,0,0)')
    
    div = opy.plot(fig, auto_open=False, config = config, output_type='div')

    return div

####################
###   Scatter   ####
####################
def infiltrated_score_plot(ExprDf = None, lCell = None, lMir = None):
    import plotly.graph_objs as go
    import plotly.express as px
    import plotly.offline as opy

    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
    }}


    # Rename the lists of columns

    fig = go.Figure()
    i = 0

    for cell in lCell:  
        if i == 0:
            fig.add_scatter(x=ExprDf[lMir[0]], y=ExprDf[cell], mode='markers', visible = True)
        else:
            fig.add_scatter(x=ExprDf[lMir[0]], y=ExprDf[cell], mode='markers', visible = False)
        i += 1

        
    #The trace restyling  to be performed at an option selection in the first/second dropdown menu
    # is defined within  buttons1/buttons2 below:

    buttons1 = [dict(method = "restyle",
                    args = [{'x': [ExprDf[lMir[k]] for cell in lCell],
                            'y': [ExprDf[cell] for cell in lCell]
                            }], 
                    label = lMir[k])   for k in range(0, len(lMir))]  


    buttons2 = [dict(method = "restyle",
                    args = [{'visible':[cell == lCell[k] for cell in lCell]}],
                    label = lCell[k])   for k in range(0, len(lCell))]  



    fig.update_layout(title_text='Module Score',

                    
                    updatemenus=[dict(active=0,
                                        buttons=buttons1,
                                        x=1.15,
                                        y=1,
                                        xanchor='left',
                                        yanchor='top'),
                                
                                dict(buttons=buttons2,
                                        x=1.15,
                                        y=0.85,
                                        xanchor='left',
                                        yanchor='top')

                                

                                ]); 

    #Add annotations for the two dropdown menus:


    # Set x-axis title
    fig.update_xaxes(title_text="<b>miRNA Expression</b>")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Module score</b>")

    fig.add_annotation(
                x=1.05,
                y=1,
                xref='paper',
                yref='paper',
                showarrow=False,
                xanchor='left',
                text="Select<br>miRNA")

    fig.add_annotation(
                x=1.05,
                y=0.85,
                showarrow=False,
                xref='paper',
                yref='paper',
                xanchor='left',
                #yanchor='top',
                text="Select<br>Geneset");

    div = opy.plot(fig, auto_open=False, output_type='div', config = config)

    return div

def module_score_plot(ExprDf = None, dfDict = None, lMir = None):
    import plotly.graph_objs as go
    import plotly.express as px
    import plotly.offline as opy

    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
    }}

    # Rename the lists of columns
    lGeneSet = list(dfDict.keys())

    # Rename the lists of columns

    fig = go.Figure()
    i = 0

    for geneset in lGeneSet:  
        if i == 0:
            fig.add_scatter(x=ExprDf[lMir[0]], y=dfDict[geneset][lMir[0]], mode='markers', visible = True)
        else:
            fig.add_scatter(x=ExprDf[lMir[0]], y=dfDict[geneset][lMir[0]], mode='markers', visible = False)
        i += 1

        
    #The trace restyling  to be performed at an option selection in the first/second dropdown menu
    # is defined within  buttons1/buttons2 below:

    buttons1 = [dict(method = "restyle",
                    args = [{'x': [ExprDf[lMir[k]] for df in dfDict.values()],
                            'y': [df[lMir[k]] for df in dfDict.values()]
                            }], 
                    label = lMir[k])   for k in range(0, len(lMir))]  


    buttons2 = [dict(method = "restyle",
                    args = [{'visible':[name == lGeneSet[k] for name in lGeneSet]}],
                    label = lGeneSet[k])   for k in range(0, len(lGeneSet))]  



    fig.update_layout(title_text='Module Score',

                    
                    updatemenus=[dict(active=0,
                                        buttons=buttons1,
                                        x=1.15,
                                        y=1,
                                        xanchor='left',
                                        yanchor='top'),
                                
                                dict(buttons=buttons2,
                                        x=1.15,
                                        y=0.85,
                                        xanchor='left',
                                        yanchor='top')

                                

                                ]); 

    #Add annotations for the two dropdown menus:


    # Set x-axis title
    fig.update_xaxes(title_text="<b>miRNA Expression</b>")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Module score</b>")

    fig.add_annotation(
                x=1.05,
                y=1,
                xref='paper',
                yref='paper',
                showarrow=False,
                xanchor='left',
                text="Select<br>miRNA")

    fig.add_annotation(
                x=1.05,
                y=0.85,
                showarrow=False,
                xref='paper',
                yref='paper',
                xanchor='left',
                #yanchor='top',
                text="Select<br>Geneset");

    div = opy.plot(fig, auto_open=False, output_type='div', config = config)

    return div


def ips_score_plot(ExprDf = None, dfIps = None, lMir = None):
    import plotly.graph_objs as go
    import plotly.express as px
    import plotly.offline as opy
    import pandas as pd
    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
    }}

    # Rename the lists of columns

    fig = go.Figure()
    i = 0
    ExprDf = pd.concat((ExprDf[lMir], dfIps), axis = 1)
    for mir in lMir:  
        if i == 0:
            fig.add_scatter(x=ExprDf[mir], y=ExprDf["AZ"], mode='markers', visible = True)
        else:
            fig.add_scatter(x=ExprDf[mir], y=ExprDf["AZ"], mode='markers', visible = False)
        i += 1

        
    #The trace restyling  to be performed at an option selection in the first/second dropdown menu
    # is defined within  buttons1/buttons2 below:

    buttons1 = [dict(method = "restyle",
                    args = [{'x': [ExprDf[lMir[k]]],
                            'y': [ExprDf["AZ"]]
                            }], 
                    label = lMir[k])   for k in range(0, len(lMir))]  





    fig.update_layout(title_text='Immunophenoscore',

                    
                    updatemenus=[dict(active=0,
                                        buttons=buttons1,
                                        x=1.15,
                                        y=1,
                                        xanchor='left',
                                        yanchor='top'),
                            

                                

                                ]); 

    #Add annotations for the two dropdown menus:


    # Set x-axis title
    fig.update_xaxes(title_text="<b>miRNA Expression</b>")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Immunophenoscore</b>")

    fig.add_annotation(
                x=1.05,
                y=1,
                xref='paper',
                yref='paper',
                showarrow=False,
                xanchor='left',
                text="Select<br>miRNA")


    div = opy.plot(fig, auto_open=False, output_type='div', config = config)

    return div
#################
### ROC #####3###
#################

def roc_cruve_plotly (results):
    import plotly.offline as opy

    import plotly.graph_objects as go
    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
    }}

    c_fill      = 'rgba(52, 152, 219, 0.2)'
    c_line      = 'rgba(52, 152, 219, 0.5)'
    c_line_main = 'rgba(41, 128, 185, 1.0)'
    c_grid      = 'rgba(189, 195, 199, 0.5)'
    fpr_mean    = np.linspace(0, 1, 100)
    interp_tprs = []
    k = len(results['fpr'])
    print(k)
    for i in range(k):
        fpr           = results['fpr'][i]
        tpr           = results['tpr'][i]
        interp_tpr    = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    if k > 1:
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(results['auc'])
        
        fig = go.Figure([
            go.Scatter(
                x          = fpr_mean,
                y          = tpr_upper,
                line       = dict(color=c_line, width=1),
                showlegend = False,
                name       = 'upper'),
            go.Scatter(
                x          = fpr_mean,
                y          = tpr_lower,
                fill       = 'tonexty',
                fillcolor  = c_fill,
                line       = dict(color=c_line, width=1),
                showlegend = False,
                name       = 'lower'),
            go.Scatter(
                x          = fpr_mean,
                y          = tpr_mean,
                line       = dict(color=c_line_main, width=2),
                showlegend = True,
                name       = f'AUC: {auc:.3f}')
        ])

    else:
        import plotly.express as px

        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC=%.4f)'%(results['auc'][0]),
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        #fig.show()

    fig.add_shape(
        type ='line', 
        line =dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        template    = 'plotly_white', 
        title_x     = 0.5,
        xaxis_title = "1 - Specificity",
        yaxis_title = "Sensitivity",
        width       = 800,
        height      = 800,
        legend      = dict(
            yanchor="bottom", 
            xanchor="right", 
            x=0.95,
            y=0.01,
        )
    )
    fig.update_yaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        scaleanchor = "x", 
        scaleratio  = 1,
        linecolor   = 'black')
    fig.update_xaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        constrain   = 'domain',
        linecolor   = 'black')

    div = opy.plot(fig, auto_open=False, config = config, output_type='div')

    return div

####################
## Feature Weigth ##
####################

def feature_weigth(df):
    import plotly.express as px
    import plotly.offline as opy
    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
    }}

    fig = px.bar(df, orientation='h')
    fig.update_layout(template = "plotly_white")
    div = opy.plot(fig, config = config, auto_open=False, output_type='div')

    return div


def survival_weigth_plot(matrix = None):
    import plotly.graph_objs as go
    import plotly.express as px
    import plotly.offline as opy

    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
    }}

    # Rename the lists of columns
    lModels = matrix.columns.tolist()

    # Rename the lists of columns

    fig = go.Figure()
    i = 0

    for model in lModels:

        dfModel = matrix[model] 
        dfModel = dfModel[dfModel.fillna(0)!=0]
        dfModel.sort_values(ascending = False, inplace = True)  
        if i == 0:
            fig.add_bar( x = dfModel.tolist(), y = dfModel.index.tolist(),  orientation='h', visible = True)
        else:
            fig.add_bar(x = dfModel.tolist(), y = dfModel.index.tolist(), orientation='h', visible = False)
        i += 1

        
    #The trace restyling  to be performed at an option selection in the first/second dropdown menu
    # is defined within  buttons1/buttons2 below:

    buttons2 = [dict(method = "restyle",
                    args = [{'title_text': lModels[k], 'visible':[name == lModels[k] for name in lModels]}],
                    label = lModels[k])   for k in range(0, len(lModels))]  



    fig.update_layout(title_text='Feature Importance',

                    
                    updatemenus=[dict(active=0,
                                        buttons=buttons2,
                                        x=1.15,
                                        y=1,
                                        xanchor='left',
                                        yanchor='top'),
                                ]); 

    fig.update_layout(template = "plotly_white")

    #Add annotations for the two dropdown menus:


    # Set x-axis title
    fig.update_xaxes(title_text="<b>Feature name</b>")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Feature weight</b>")

    fig.add_annotation(
                x=1.05,
                y=1,
                xref='paper',
                yref='paper',
                showarrow=False,
                xanchor='left',
                text="Select<br>model")

    div = opy.plot(fig, auto_open=False, config = config, output_type='div')

    return div

#########
## PCA ##
#########

def pca_plotly(df, lFeature, label):
    import plotly.express as px
    from sklearn.decomposition import PCA
    import plotly.offline as opy
    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
    }}
    X = df[lFeature]

    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    total_var = pca.explained_variance_ratio_.sum() * 100
    print(df)
    print(df[label])
    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=df[label].map(str),
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )

    fig.update_layout(height =1240,
                    width=1240)
    fig.update_yaxes(automargin=True)


    div = opy.plot(fig, auto_open=False, output_type='div', config = config)
    return div


#########
## Kaplan MEIAR ##
#########

def closet_value(given_value, a_list):
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    closet = min(a_list, key=absolute_difference_function)

    return closet


def df_to_kaplan_meier_plot(df, timestamp="time", status="event", conditions="exprs", translations={1:"High", 0:"Low"}, q=None, target = None, treshold = None, dataset = None, optimal = False):
    import pandas as pd
    from lifelines import KaplanMeierFitter, CoxPHFitter
    import numpy as np
    from multiprocessing import Pool
    import numpy as np
    import plotly
    import plotly.offline as opy
    from miopy.survival import get_hazard_ratio
    from lifelines.plotting import add_at_risk_counts
    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
    }}


    """
    Creates a Plotly Figure object from a CSV file which has survival data
    param timestamp: str, the column name which is used for time x-axis
    param status: str, the column name which stores the condition used for calculating the Kaplan-Meier survival rate
    conditions: str, the column name where the different conditions are used, i.e. which generates multiple traces
    translations None or dict, used to store translations, e.g. for numerical encoded sexes, e.g. {1: "male", 2: 'female'}
    """
    #df = pd.read_csv(filename)

    print(df)
    fig = plotly.tools.make_subplots(rows=2, cols=1, shared_xaxes=False, print_grid=False)
    kmfs = []

    steps = 8 # the number of time points where number of patients at risk which should be shown

    x_min = 0 # min value in x-axis, used to make sure that both plots have the same range
    x_max = 0 # max value in x-axis

    lTemp = []
    for rx in df[conditions].unique():
        T = df[df[conditions] == rx][timestamp]
        E = df[df[conditions] == rx][status]
        kmf = KaplanMeierFitter()

        kmf.fit(T, event_observed=E)
        kmfs.append(kmf)
        x_max = max(x_max, max(kmf.event_table.index))
        x_min = min(x_min, min(kmf.event_table.index))
        if translations is not None and translations.get(rx) is not None:
            rx = translations[rx]
        else:
            rx = str(rx)

        fig.append_trace(plotly.graph_objs.Scatter(x=[0,] + list(kmf.survival_function_.index),
                                                   y=[1,] + list(kmf.survival_function_.values.flatten()),
                                                   name="High (>=%.2f)"%q if rx=="High" else "Low (<%.2f)"%q,
                                                   line_color= "#FF0000" if rx=="High" else "#0000FF",
                                                   line_shape = "hv"), 
                         1, 1)

    x = list(range(0, int(x_max), int(x_max / (steps - 1))))
            

    #print(lTemp)
    j = 0


    for rx in df[conditions].unique():

        if translations is not None and translations.get(rx) is not None:
            rx = translations[rx]
        else:
            rx = str(rx)


        lIndex = kmfs[j].event_table.index.tolist()
        even_table = pd.DataFrame(kmfs[j].event_table)
        fig.append_trace(plotly.graph_objs.Scatter(x=x, 
                                                   y=[rx] * len(x), 
                                                   text=[0 if lIndex[-1] < t else  even_table.loc[closet_value(t,lIndex),"at_risk"] for t in x], 
                                                   mode='text', 
                                                   showlegend=False,
                                                   textfont=dict(color="#FF0000" if rx=="High" else "#0000FF")), 
                         2, 1)
        j += 1
    # just a dummy line used as a spacer/header
    fig.append_trace(plotly.graph_objs.Scatter(x=x, 
                                               y=[''] * len(x), 
                                               mode='text', 
                                               showlegend=False), 
                     2, 1)

    # Statics
    log_hr, pval, hr_high, hr_low = get_hazard_ratio(df)
    if optimal:
        adj_pval = -1.63 * pval * (1 + 2.35 * np.log(pval))
    #print(log_hr), print(hr_high), print(hr_low)
    fig.append_trace(plotly.graph_objs.Scatter(x=(x_max*0.65,), 
                                            y=(1,),
                                            text = "HR: %.3f [%.3f,%.3f]"%(log_hr,hr_low, hr_high), 
                                            mode='text', 
                                            showlegend=False), 
                     1, 1)
    fig.append_trace(plotly.graph_objs.Scatter(x=(x_max*0.65,), 
                                            y=(0.95,),
                                            text = "Quantile = %.3f   Cutpoint = %.3f"%(q, float(treshold)), 
                                            mode='text', 
                                            showlegend=False), 
                     1, 1)
    fig.append_trace(plotly.graph_objs.Scatter(x=(x_max*0.65,), 
                                            y=(0.9,),
                                            text = "p.value = %.3f "%(pval) if pval >= 0.001 else "p.value < 0.001" , 
                                            mode='text', 
                                            showlegend=False), 
                     1, 1)
    
    if optimal:
        fig.append_trace(plotly.graph_objs.Scatter(x=(x_max*0.65,), 
                                                y=(0.85,),
                                                text = "padj = %.3f "%(adj_pval) if adj_pval >= 0.001 else "padj < 0.001" , 
                                                mode='text', 
                                                showlegend=False), 
                        1, 1)
    
    # prettier layout
    x_axis_range = [x_min - x_max * 0.05, x_max * 1.05]
    fig['layout']['xaxis2']['visible'] = False
    fig['layout']['xaxis2']['range'] = x_axis_range
    fig['layout']['xaxis']['range'] = x_axis_range
    fig['layout']['yaxis']['domain'] = [0.3, 1]
    fig['layout']['yaxis2']['domain'] = [0.0, 0.15]
    fig['layout']['yaxis2']['showgrid'] = False
    fig['layout']['yaxis']['showgrid'] = False
    fig.layout.template = "simple_white"
    fig.update_layout(
        title="Kaplan-Meier estimate by %s expression in %s"%(target, dataset),
        xaxis_title="Survival time in months",
        yaxis_title="Survival probability",
        legend_title="Expression level",
        font=dict(
            family="Arial, monospace",
            size=18,
        )
        )
    fig.update_layout({'width':900, 'height':750,})

    div = opy.plot(fig, auto_open=False, config = config, output_type='div')

    return div

