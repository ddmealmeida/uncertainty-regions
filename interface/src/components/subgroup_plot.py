from dash import html, Dash, dcc
import pandas as pd
from collections.abc import Iterable
from . import ids
from plotly.tools import mpl_to_plotly
from io import BytesIO
import base64
import sys
import plotly.express as px

sys.path.append("..")
from functions import plot_subgroups, plot_subgroups_px


def render(
    app: Dash,
    dataset_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    target: Iterable,
    subgroups: pd.DataFrame,
) -> None:
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
            [html.H4("Subgroups"), dcc.Graph(id=ids.SUBGROUPS_PLOT_ID)]
        )

    # Every subsequent render call should just return the figure to update the plot
    return fig
