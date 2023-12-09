from dash import html, Dash, dash_table, Input, Output, callback
from . import subgroups_dropdown, subgroup_plot
from . import ids
import pandas as pd


def create_layout(
    app: Dash, dataset_df: pd.DataFrame, subgroups_df: pd.DataFrame
) -> None:
    table_subgroups_df: pd.Dataframe = subgroups_df[
        ["subgroup_str", "size_sg", "mean_sg"]
    ].rename(
        columns={
            "subgroup_str": "Subgrupo",
            "size_sg": "Tamanho",
            "mean_sg": "Erro médio do subgrupo",
        }
    )
    return html.Div(
        id=ids.MAIN_LAYOUT_ID,
        children=[
            html.H1("Tabela de subgrupos", style={"textAlign": "center"}),
            dash_table.DataTable(
                id="rules_table",
                data=table_subgroups_df.to_dict("records"),
                columns=[{"id": c, "name": c} for c in table_subgroups_df.columns],
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
                    "color": "black",
                },
                style_header={
                    "backgroundColor": "white",
                    "fontWeight": "bold",
                    "textAlign": "center",
                },
                page_size=min(subgroups_df.shape[0], 20),
            ),
            html.Div(
                className="dropdown-container",
                children=[
                    subgroups_dropdown.render(
                        app=app, dataset_df=dataset_df, subgroups_df=subgroups_df
                    )
                ],
            ),
            html.Div(
                id=ids.SUBGROUPS_PLOT_ID,
                className="subgroups-plot",
                children=[
                    subgroup_plot.render(
                        app=app,
                        dataset_df=dataset_df,
                        x_column="petal width (cm)",
                        y_column="petal length (cm)",
                        target=dataset_df.target.tolist(),
                        subgroups=None,
                    )
                ],
            ),
        ],
    )
