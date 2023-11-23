from dash import html, Dash, dash_table, Input, Output, callback
from . import subgroups_dropdown
import pandas as pd

def create_dataframe_table(app: Dash, df: pd.DataFrame) -> None:
    return html.Div(
        children=[
            html.H1("Tabela de subgrupos", style={"textAlign": "center"}),
            dash_table.DataTable(
                id="rules_table",
                data=df.to_dict("records"),
                columns=[{"id": c, "name": c} for c in df.columns],
                style_as_list_view=True,
                style_cell={
                    "textAlign": "center",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "maxWidth": 0,
                },
                style_cell_conditional=[
                    {
                        "if": {"column_id": "subgroup"},
                        "textAlign": "left",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "maxWidth": 0,
                    }
                ],
                style_table={"height": "600px", "overflowY": "auto"},
                style_data={
                    "height": "auto",
                    "line_height": "30px",
                    "whiteSpace": "normal",
                },
                style_header={
                    "backgroundColor": "white",
                    "fontWeight": "bold",
                    "textAlign": "center",
                },
                page_size=min(df.shape[0], 20),
            ),
            html.Div(
                className="dropdown-container",
                children=[subgroups_dropdown.render(app=app, df=df)],
            ),
        ]
    )
