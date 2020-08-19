"""Methods called from subplots"""

from typing import Iterable

import pandas as pd


def get_plot_values(
    df, sort_column: str, groupby_columns: Iterable, value_column: str
) -> dict:
    """Get values to plot from a groupby"""

    dict_plot_values = {}
    for index, sub_df in df.sort_values(sort_column).groupby(groupby_columns):
        dict_plot_values[frozenset(index)] = sub_df[value_column].values
    return dict_plot_values


def map_teacher_colors() -> dict:
    """
    Map each of the teachers to a color
    Warning: assumes the teachers
    """

    return {
        "leitner": "blue",
        "threshold": "orange",
        "sampling": "green",
    }
