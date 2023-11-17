from dash import html, Dash
import plotly.graph_objects as go
import pandas as pd


def create_dataframe_table(app: Dash, df: pd.DataFrame) -> None:
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns), fill_color="paleturquoise", align="left"
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    
    fig.show()