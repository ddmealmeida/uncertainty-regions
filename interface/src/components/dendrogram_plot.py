from dash import html, Dash, dcc
import pandas as pd
from collections.abc import Iterable
from . import ids
import sys

sys.path.append("..")
from functions import plot_dendrogram


def render(app: Dash, subgroups_df: pd.DataFrame):
  plot_dendrogram(df_regras=subgroups_df)