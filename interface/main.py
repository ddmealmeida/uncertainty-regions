from dash import Dash, html
from dash_bootstrap_components.themes import BOOTSTRAP
from src.components.layout import create_layout
from src.components.table import create_dataframe_table
import plotly.graph_objects as go
import pandas as pd

def main() -> None:
    app = Dash()
    app.title = "Uncertainty Regions"
    app.layout = create_layout(app=app)

    df: pd.DataFrame = pd.read_csv("../data/iris.csv")
    create_dataframe_table(app=app, df=df)
    app.run()


if __name__ == "__main__":
    main()
