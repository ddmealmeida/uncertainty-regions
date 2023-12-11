from dash import html, Dash, dash_table
import pandas as pd


def render(app: Dash, table_subgroups_df: pd.DataFrame) -> dash_table.DataTable:
    return dash_table.DataTable(
        id="rules_table",
        data=table_subgroups_df.to_dict("records"),
        columns=[{"id": c, "name": c} for c in table_subgroups_df.columns],
        style_table={"height": "600px", "overflowY": "auto"},
        style_cell={
            "textAlign": "center",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "maxWidth": 0,
        },
        style_data={
            "height": "auto",
            "line_height": "30px",
            "whiteSpace": "normal",
            "color": "black",
        },
        style_header={
            "backgroundColor": "white",
            "fontWeight": "bold",
            "textAlign": "center",
        },
        page_size=min(table_subgroups_df.shape[0], 20),
    )
