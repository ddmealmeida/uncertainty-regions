from dash import html, Dash, dash_table, Input, Output, callback
import pandas as pd


def create_dataframe_table(app: Dash, df: pd.DataFrame) -> None:
    return dash_table.DataTable(df.to_dict("records"))