from dash import html, Dash, dash_table, Input, Output, callback
from . import (
    subgroups_dropdown,
    subgroup_plot,
    dendrogram_plot,
    subgroups_table,
    venn_diagram_plot,
)
from . import ids
import pandas as pd


def create_layout(
    app: Dash, dataset_df: pd.DataFrame, subgroups_df: pd.DataFrame
) -> None:
    table_subgroups_df: pd.Dataframe = subgroups_df[
        ["subgroup_str", "size_sg", "mean_sg", "class"]
    ].rename(
        columns={
            "subgroup_str": "Subgrupo",
            "size_sg": "Tamanho",
            "mean_sg": "Erro médio do subgrupo",
        }
    )
    # rounding mean_sg (Erro médio do subgrupo) to 3 decimal cases
    table_subgroups_df["Erro médio do subgrupo"] = table_subgroups_df[
        "Erro médio do subgrupo"
    ].round(3)

    return html.Div(
        id=ids.MAIN_LAYOUT_ID,
        style={"margin": "auto", "width": "70%"},
        children=[
            html.H1("Subgroups table", style={"textAlign": "center"}),
            html.Div(
                className="subgroups-datatable",
                children=[
                    subgroups_table.render(
                        app=app,
                        table_subgroups_df=table_subgroups_df[
                            table_subgroups_df["class"] == 1
                        ],
                    )
                ],
            ),
            html.Br(),
            html.Div(
                className="dropdown-container",
                children=[
                    subgroups_dropdown.render(
                        app=app,
                        dataset_df=dataset_df,
                        subgroups_df=subgroups_df[subgroups_df["class"] == 1],
                    ),
                    subgroup_plot.render(
                        app=app,
                        dataset_df=dataset_df,
                        x_column="petal width (cm)",
                        y_column="petal length (cm)",
                        target=dataset_df.target.tolist(),
                        subgroups=None,
                    ),
                ],
            ),
            html.Div(
                className="venn-diagram-plot",
                children=[venn_diagram_plot.render(app=app, rules_df=subgroups_df)],
            ),
        ],
    )
