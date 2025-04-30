import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np



def plot_block_frequencies(df):
    blocks = df["block"].astype(str)
    abs_freq = df["absolute frequency"]
    rlt_freq = df["relative frequency"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Absolute frequency", "Relative frequency"),
        shared_yaxes=False
    )

    fig.add_trace(
        go.Bar(x=blocks, y=abs_freq, name="Absolute frequency", marker_color="blue"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=blocks, y=rlt_freq, name="Relative frequency", marker_color="red"),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Blocks Analysis",
        xaxis_title="Blocks",
        yaxis_title="Absolute Frequency",
        xaxis2_title="Blocks",
        yaxis2_title="Relative Frequency",
        height=800,
        width=1500,
        showlegend=False
    )
    fig.show()

def plot_3D_entropy_bias(test_results,test='Entropy Bias'):
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