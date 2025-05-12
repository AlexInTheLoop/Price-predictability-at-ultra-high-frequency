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

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import plotly.express as px

def sub_plots_comp(result, pairs, year, month, day, test):
    # Determine if we are working with days or months
    is_daily = day is not None
    global_title = f"Fraction of predictable {'hourly' if is_daily else 'daily'} intervals ({test})"
    y_axis_title = "Fraction of predictable hours" if is_daily else "Fraction of predictable days"
    x_axis_title = "Aggregation level"

    # Create a subplot grid with up to 3 plots per row
    num_pairs = len(pairs)
    cols = 3
    rows = -(-num_pairs // cols)  # Ceiling division
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=pairs,
        shared_xaxes=True,  # Share x-axis across all subplots
        shared_yaxes=True,  # Share y-axis across all subplots
        vertical_spacing=0.05,  # Reduce vertical spacing between subplots
        horizontal_spacing=0.05  # Reduce horizontal spacing between subplots
    )

    # Generate colors for the legend
    periods = day if is_daily else month
    if is_daily:
        legend_labels = [f"{d}/{month:02d}/{year}" for d in day]
    else:
        legend_labels = [calendar.month_name[m] for m in month]
    colors = px.colors.qualitative.Plotly[:len(periods)]

    # Add traces for each pair
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
                    mode='lines+markers',
                    name=legend_labels[period_idx],
                    line=dict(color=colors[period_idx]),
                    legendgroup=legend_labels[period_idx],  # Group traces for shared legend interaction
                    showlegend=(idx == 0)  # Show legend only for the first subplot
                ),
                row=row,
                col=col
            )

    # Update layout for global titles, axis labels, and legend
    fig.update_layout(
        title=dict(
            text=global_title,
            x=0.5,  # Center the title
            xanchor="center",
            font=dict(size=21)  # Increase title font size
        ),
        height=300 * rows,  # Adjust height dynamically based on the number of rows
        width=1000,
        legend_title="Periods",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,  # Position the legend below the plots
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=100, l=100, r=50),  # Adjust margins for axis titles
        annotations=[
            dict(
                text=x_axis_title,
                x=0.5,
                y=-0.15,  # Position the x-axis title between the last row and the legend
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=17)  # Increase x-axis title font size
            ),
            dict(
                text=y_axis_title,
                x=-0.07,
                y= 0.18 + (0.05 * (rows - 1)),  # Dynamically adjust the y position based on the number of rows
                xref="paper",
                yref="paper",
                showarrow=False,
                textangle=-90,
                font=dict(size=17)  # Increase y-axis title font size
            )
        ]
    )

    # Remove duplicate x-axis titles
    for i in range(1, cols + 1):
        fig.update_xaxes(title_text=None, row=rows, col=i)

    # Show the figure
    fig.show()