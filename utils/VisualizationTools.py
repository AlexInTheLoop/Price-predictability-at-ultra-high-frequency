import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import calendar
import plotly.express as px



def plot_block_frequencies(df):
    blocks = df["block"].astype(str)
    abs_freq = df["absolute frequency"]
    rlt_freq = df["relative frequency"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Blocks absolute frequency", "Blocks relative frequency"),
        shared_yaxes=False
    )

    fig.add_trace(
        go.Bar(x=blocks, y=abs_freq, name="Blocks absolute frequency", marker_color="gray"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=blocks, y=rlt_freq, name="Blocks relative frequency", marker_color="black"),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Blocks Analysis",
        xaxis_title="Blocks",
        yaxis_title="Absolute Frequency",
        xaxis2_title="Blocks",
        yaxis2_title="Relative Frequency",
        height=600,
        width=1200,
        showlegend=False
    )
    fig.show()

def plot_3D(test_results,test='Entropy Bias'):
    bias_surface = test_results[:, :, 0] 
    quantile_99_surface = test_results[:, :, 1]

    aggregation_levels = np.arange(1, 51)
    block_sizes = np.arange(1, 11)
    agg_grid, block_grid = np.meshgrid(aggregation_levels, block_sizes, indexing='ij')

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=bias_surface,
        x=agg_grid,
        y=block_grid,
        colorscale='Viridis',
        name=test
    ))

    fig.add_trace(go.Surface(
        z=quantile_99_surface,
        x=agg_grid,
        y=block_grid,
        colorscale='Cividis',
        name='Quantile 99',
        showscale=False
    ))

    fig.update_layout(
        title=test + " by Aggregation Level and Block Size",
        scene=dict(
            xaxis_title="Aggregation Level",
            yaxis_title="Block Size",
            zaxis_title=test
        ),
        height=700,
        width=900
    )

    fig.show()
        
def plot_predictability(aggregation_levels, predictability, x_label='Aggregation Level'):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=aggregation_levels,
        y=predictability,
        mode='lines',
        name='Predictability',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title=f"Predictability by {x_label}",
        xaxis_title=x_label,
        yaxis_title="Predictability",
        height=700,
        width=900
    )
    fig.show()

def plot_test(x_values, 
              y1_values, y2_values, 
              test='Entropy Bias',
              x_label='aggregation Level',
              pair=''):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y1_values,
        mode='lines',
        name='Test statistic',
        line=dict(color="dark red", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y2_values,
        mode='lines',
        name='Quantile 99%',
        line=dict(color='dark green', width=2)
    ))

    fig.update_layout(
        title=dict(
            text=test + " by " + x_label + " for " + pair,
            x=0.5,
            xanchor="center",
            font=dict(size=20)
        ),
        xaxis_title=dict(
            text="By " + x_label,
            font=dict(size=16)
        ),
        yaxis_title=dict(
            text= test + " statistic and quantile 99%",
            font=dict(size=16)
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.99
        ),
        height=700,
        width=900,
        template="plotly_white"
    )
    
    fig.show()


def plot_all_models(x_data, 
                         y_data, 
                         x_label='Aggregation Level', 
                         y_label='Fraction of Predictable Intervals', 
                         title='Fraction of predictable simulated intervals'):
    fig = go.Figure()

    for i, (model,y) in enumerate(y_data.items()):
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y,
            mode='lines',
            name=model,
            line=dict(width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=700,
        width=900
    )

    fig.show()


def sub_plots_comp(result, pairs, year, month, day, test):
    is_daily = day is not None
    global_title = f"Fraction of predictable {'hourly' if is_daily else 'daily'} intervals ({test})"
    y_axis_title = "Fraction of predictable hours" if is_daily else "Fraction of predictable days"
    x_axis_title = "Aggregation level"
    num_pairs = len(pairs)
    cols = 3
    rows = -(-num_pairs // cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.03
    )
    periods = day if is_daily else month
    if is_daily:
        legend_labels = [f"{d}/{month:02d}/{year}" for d in day]
    else:
        legend_labels = [calendar.month_name[m] for m in month]
    colors = px.colors.qualitative.Plotly[:len(periods)]
    
    annotations = []
    
    for idx, pair in enumerate(pairs):
        row = idx // cols + 1
        col = idx % cols + 1
        
        for period_idx, period in enumerate(periods):
            y_values = [result[period][level][idx] for level in range(len(result[period]))]
            x_values = list(range(1, len(y_values) + 1))
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines',
                    name=legend_labels[period_idx],
                    line=dict(color=colors[period_idx], width=2),
                    legendgroup=legend_labels[period_idx],
                    showlegend=(idx == 0)
                ),
                row=row,
                col=col
            )
    
    annotations.append(dict(
        text=x_axis_title,
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=17)
    ))
    
    annotations.append(dict(
        text=y_axis_title,
        x=-0.07,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=dict(size=17)
    ))
    
    fig.update_layout(
        title=dict(
            text=global_title,
            x=0.5,
            xanchor="center",
            font=dict(size=21)
        ),
        height=300 * rows,
        width=1000,
        legend_title="Periods",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=100, l=100, r=50),
        annotations=annotations
    )
    
    for i in range(num_pairs):
        row = i // cols + 1
        col = i % cols + 1
        
        xref = f"x{i+1}" if i > 0 else "x"
        yref = f"y{i+1}" if i > 0 else "y"
        
        fig.add_annotation(
            text=pairs[i],
            x=0.5,
            y=-0.15,
            xref=xref + " domain",
            yref=yref + " domain",
            showarrow=False,
            font=dict(size=14),
            xanchor="center",
            yanchor="top"
        )
    
    fig.show()

