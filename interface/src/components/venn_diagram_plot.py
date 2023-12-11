from dash import html, Dash, dcc
import pandas as pd
from . import ids
import sys

sys.path.append("..")
from functions import venn_diagram


def render(app: Dash, rules_df: pd.DataFrame) -> html.Div:
    return html.Div(
        className="venn-diagram",
        children=[dcc.Graph(figure=venn_diagram(df_regras=rules_df))],
    )
