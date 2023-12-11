from dash import html, Dash, dcc
import pandas as pd
from collections.abc import Iterable
from . import ids
import sys

sys.path.append("..")
from functions import plot_subgroups_px


def render(
    app: Dash,
    dataset_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    target: Iterable,
    subgroups: pd.DataFrame,
) -> html.Div:
    fig = plot_subgroups_px(
        data=dataset_df,
        x_column=x_column,
        y_column=y_column,
        target=target,
        subgroups=subgroups,
    )
    # First plot should create html.Div with no plot
    if subgroups is None:
        return html.Div(
            className="subgroups-2d-plot",
            children=[
                html.H3("Subgroups 2D plot", style={"textAlign": "center"}),
                dcc.Graph(
                    id=ids.SUBGROUPS_PLOT_ID, figure=fig, style={"textAlign": "center"}
                ),
            ],
        )

    # Every subsequent render call should just return the figure to update the plot
    return fig
