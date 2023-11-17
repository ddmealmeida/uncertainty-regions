from dash import Dash, html
from src.components.layout import create_layout

def main() -> None:
    app = Dash()
    app.title = "Uncertainty Regions"
    app.layout = create_layout(app=app)
    app.run()


if __name__ == "__main__":
    main()
