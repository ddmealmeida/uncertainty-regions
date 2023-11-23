from dash import html, Dash
import dash_core_components
import pandas as pd
from collections.abc import Iterable
from . import ids
from plotly.tools import mpl_to_plotly
from io import BytesIO
import base64
import sys
import plotly.express as px

sys.path.append("..")
from functions import plot_subgroups


def render(
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

    buf = BytesIO()
    fig.savefig(buf, format="png")
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    fig_bar_matplotlib = f"data:image/png;base64,{fig_data}"

    return html.Div(
        style={"width": "50%", "margin": "auto"},
        children=[html.Img(id="example", src=fig_bar_matplotlib)],
    )
