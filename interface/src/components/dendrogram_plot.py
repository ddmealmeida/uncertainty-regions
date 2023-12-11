from dash import html, Dash, dcc
import pandas as pd
from collections.abc import Iterable
from . import ids
import sys

sys.path.append("..")
from functions import plot_dendrogram


def render(app: Dash, subgroups_df: pd.DataFrame) -> html.Div:
    return html.Div(
        className="dendogram-plot",
        children=[
            html.H3("Hierarchical Clustering Dendogram", style={"textAlign": "center"}),
            dcc.Graph(
                id=ids.DENDOGRAM_PLOT_ID, figure=plot_dendrogram(df_regras=subgroups_df)
            ),
        ],
    )
