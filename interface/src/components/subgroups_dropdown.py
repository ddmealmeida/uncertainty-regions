from dash import Dash, html, dcc, Input, Output, State, exceptions
import pandas as pd
from . import ids
from . import subgroup_plot


def render(app: Dash, dataset_df: pd.DataFrame, subgroups_df: pd.DataFrame) -> html.Div:
    all_subgroups: list[str] = subgroups_df.subgroup_str.tolist()

    @app.callback(
        Output(ids.SUBGROUPS_PLOT_ID, "figure"),
        Input(ids.PLOT_SUBGROUPS_BUTTON_ID, "n_clicks"),
        State(ids.SUBGROUPS_DROPDOWN_ID, "value"),
    )
    def plot_subgroups(n_clicks: int, selected_subgroups: list[str]) -> html.Div:
        if n_clicks is None:
            raise exceptions.PreventUpdate

        df_rows: list[int] = []
        for subgroup in selected_subgroups:
            df_rows.append(
                subgroups_df.index[subgroups_df.subgroup_str == subgroup].tolist()[0]
            )
        return subgroup_plot.render(
            app=app,
            dataset_df=dataset_df,
            x_column="petal width (cm)",
            y_column="petal length (cm)",
            target=dataset_df.target.tolist(),
            subgroups=subgroups_df.loc[
                df_rows, ["subgroup", "mean_sg", "mean_dataset"]
            ],
        )

    @app.callback(
        Output(ids.SUBGROUPS_DROPDOWN_ID, "options"),
        Input(ids.SUBGROUPS_DROPDOWN_ID, "value"),
    )
    def filter_subgroups(selected_subgroups: list[str]) -> list[str]:
        if len(selected_subgroups) == 0:
            return all_subgroups

        if len(selected_subgroups) > 1:
            raise exceptions.PreventUpdate

        # get first cause it is the only one
        first_subgroup = subgroups_df[
            subgroups_df["subgroup_str"] == selected_subgroups[0]
        ]

        # case where only one dimension defines rule
        if first_subgroup.y_column.iloc[0] == "":
            return subgroups_df[
                (subgroups_df.x_column == first_subgroup.x_column.iloc[0])
                ^ (subgroups_df.y_column == first_subgroup.x_column.iloc[0])
            ].subgroup_str.tolist()
        else:
            return subgroups_df[
                (
                    (subgroups_df.x_column == first_subgroup.x_column.iloc[0])
                    & (subgroups_df.y_column == first_subgroup.y_column.iloc[0])
                )
                ^ (
                    (subgroups_df.x_column == first_subgroup.y_column.iloc[0])
                    & (subgroups_df.y_column == first_subgroup.x_column.iloc[0])
                )
            ].subgroup_str.tolist()

    return html.Div(
        className="subgroups-dropdown",
        children=[
            dcc.Dropdown(
                id=ids.SUBGROUPS_DROPDOWN_ID,
                options=[{"label": rule, "value": rule} for rule in all_subgroups],
                value=[],
                multi=True,
                placeholder="Select a subgroup",
            ),
            html.Button(
                className="dropdown-button",
                children=["Plot subgroups"],
                id=ids.PLOT_SUBGROUPS_BUTTON_ID,
                style={"width": "auto", "align": "center"},
            ),
        ],
    )
