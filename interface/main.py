from dash import Dash, html
from dash_bootstrap_components.themes import BOOTSTRAP
from src.components.layout import create_layout
import plotly.graph_objects as go

def main() -> None:
    app = Dash()
    app.title = "Uncertainty Regions"
    app.layout = create_layout(app=app)
    app.run()


if __name__ == "__main__":
    main()
