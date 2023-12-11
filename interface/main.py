from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP
from src.components.layout import create_layout
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
    
    # subgroups_df = pd.concat(df_dict, ignore_index=True)
    subgroups_df = remove_redundant_subgroups(df_dict=df_dict)

    # adding string column with rules
    subgroups_df["subgroup_str"] = subgroups_df.subgroup.astype(str)

    # adding two extra columns for each coordinate of each rule. If rule contain only one coordinate, second column will be null
    subgroups_df["x_column"] = ""
    subgroups_df["y_column"] = ""

    for idx, subgroup in enumerate(subgroups_df.subgroup):
        rules = subgroup.selectors
        if len(rules) < 2:
            subgroups_df.x_column.at[idx] = rules[0].attribute_name
        else:
            subgroups_df.x_column.at[idx] = rules[0].attribute_name
            subgroups_df.y_column.at[idx] = rules[1].attribute_name

    # app = Dash(external_stylesheets=[BOOTSTRAP])
    app = Dash()
    app.title = "Uncertainty Regions"

    # print(df_dict["1"]["subgroup"].iloc[0].selectors[0].attribute_name)

    app.layout = create_layout(
        app=app, dataset_df=dataset_df, subgroups_df=subgroups_df
    )

    app.run(debug=True)


if __name__ == "__main__":
    main()
