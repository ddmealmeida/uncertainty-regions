from dash import Dash, html
from dash_bootstrap_components.themes import BOOTSTRAP
from src.components.layout import create_layout
from src.components.table import create_dataframe_table
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

    # apply subgroup discovery
    df_dict: dict = subgroup_discovery(
        dataset_df=dataset_df, errors_df=errors_df, number_of_classes=3
    )

    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "Uncertainty Regions"
    app.layout = create_layout(app=app)
    app.run(debug=True)


if __name__ == "__main__":
    main()
