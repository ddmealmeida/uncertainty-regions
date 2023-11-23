from dash import Dash, html, Output, Input
from dash_bootstrap_components.themes import BOOTSTRAP
from src.components.layout import create_layout
from src.components.table import create_dataframe_table
from src.components.subgroup_plot import create_2d_plot
import plotly.graph_objects as go
import pandas as pd
import sys

sys.path.append("..")
from functions import *


def main() -> None:
    config_file_path: str = sys.argv[1]

    with open(config_file_path, "r") as f:
        dataset_file_path = f.readline()[:-1]  # remove \n
        errors_file_path = f.readline()
        dataset_df: pd.DataFrame = pd.read_csv(dataset_file_path)
        errors_df: pd.DataFrame = pd.read_csv(errors_file_path)
        errors_df.rename(columns={"0": 0, "1": 1, "2": 2}, inplace=True)

    # apply subgroup discovery
    df_dict: dict = subgroup_discovery(
        dataset_df=dataset_df, errors_df=errors_df, number_of_classes=3
    )

    # changing column subgroup to be string
    for cls in range(3):
        df_dict[str(cls)]["subgroup_str"]= df_dict[str(cls)].subgroup.astype(str)

    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "Uncertainty Regions"
    app.layout = create_dataframe_table(
        app=app, df=df_dict["0"][["subgroup_str", "size_sg"]]
    )

    """ app.layout = create_2d_plot(
        app=app,
        dataset_df=dataset_df,
        x_column="petal width (cm)",
        y_column="petal length (cm)",
        target=dataset_df.target.tolist(),
        subgroups=df_dict["0"].loc[
            [1, 2, 3, 4], ["subgroup", "mean_sg", "mean_dataset"]
        ],
    ) """

    app.run(debug=True)


if __name__ == "__main__":
    main()
