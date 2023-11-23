from dash import Dash, html, dcc, Input, Output
import pandas as pd
from . import ids
from . import subgroup_plot


def render(app: Dash, dataset_df: pd.DataFrame, subgroups_df: pd.DataFrame) -> html.Div:
    all_subgroups: list[str] = subgroups_df.subgroup_str.tolist()
    @app.callback(
        Output(ids.SUBGROUPS_PLOT_ID, "children"),
        Input(ids.PLOT_SUBGROUPS_BUTTON_ID, "n_clicks"),
        Input(ids.SUBGROUPS_DROPDOWN_ID, "value"),
    )
    def plot_subgroups(_: int, selected_subgroups: list[str]) -> html.Div:
        df_rows: list[int] = []
        for subgroup in selected_subgroups:
            df_rows.append(subgroups_df.index[subgroups_df.subgroup_str == subgroup].tolist()[0])
        return subgroup_plot.render(
            app=app,
            dataset_df=dataset_df,
            x_column="petal width (cm)",
            y_column="petal length (cm)",
            target=dataset_df.target.tolist(),
            subgroups=subgroups_df.loc[df_rows, ["subgroup", "mean_sg", "mean_dataset"]],
        )

    return html.Div(
        children=[
            html.H6("Subgrupos", style={"textAlign": "center"}),
            dcc.Dropdown(
                id=ids.SUBGROUPS_DROPDOWN_ID,
                options=[{"label": rule, "value": rule} for rule in all_subgroups],
                value=[],
                multi=True,
            ),
            html.Button(
                className="dropdown-button",
                children=["Plot subgroups"],
                id=ids.PLOT_SUBGROUPS_BUTTON_ID,
            ),
        ]
    )
