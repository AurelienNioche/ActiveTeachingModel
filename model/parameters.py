#%%
import pandas as pd

def make_param_learners_df() -> pd.DataFrame:
    """Get parameter names for all learners"""

    exp_decay = pd.Series(
        ("alpha", "beta"),
        name="exp_decay",
    )

    walsh2018 = pd.Series(
        ("tau", "s", "b", "m", "c", "x"),
        name="walsh2018",
    )

    return pd.concat((exp_decay, walsh2018), axis=1)

