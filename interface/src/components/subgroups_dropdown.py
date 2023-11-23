from dash import Dash, html, dcc
import pandas as pd
from . import ids


def render(app: Dash, df: pd.DataFrame) -> html.Div:
    all_subgroups: list[str] = df.subgroup_str.tolist()
    return html.Div(
        children=[
            html.H6("subgroups"),
            dcc.Dropdown(
                id=ids.SUBGROUPS_DROPDOWN_ID,
                options=[{"label": rule, "value": rule} for rule in all_subgroups],
                value=None,
                multi=True,
            ),
        ]
    )
