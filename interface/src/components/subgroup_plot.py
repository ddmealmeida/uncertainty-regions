from dash import html, Dash
import dash_core_components
import pandas as pd
from collections.abc import Iterable
import matplotlib
from plotly.tools import mpl_to_plotly

import sys

sys.path.append("..")
from functions import plot_subgroups


def create_2d_plot(
    app: Dash,
    dataset_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    target: Iterable,
    subgroups: pd.DataFrame,
) -> None:
    fig = plot_subgroups(
        data=dataset_df,
        x_column=x_column,
        y_column=y_column,
        target=target,
        subgroups=subgroups,
    )

    return html.Div(
        [dash_core_components.Graph(id="2d_plot", figure=mpl_to_plotly(fig))]
    )
