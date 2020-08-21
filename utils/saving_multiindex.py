def get_saving_dict() -> dict:
    """
    Make dictionary that can appended to df.
    Remember to change the parameters according to the model.

    Example to add to df:
        saving_df.loc[last_idx+1] = this_dict
    """

    return {
    ("Agent", "Human"): None,  # bool
    ("Agent", "ID"): None,    # int
    ("Bounds", "b"): None,    # np.ndarray(float)
    ("Bounds", "c"): None,    # np.ndarray(float)
    ("Bounds", "m"): None,    # np.ndarray(float)
    ("Bounds", "s"): None,    # np.ndarray(float)
    ("Bounds", "tau"): None,  # np.ndarray(float)
    ("Bounds", "x"): None,    # np.ndarray(float)
    ("Model", "Learner"): None,  # str
    ("Model", "Psychologist"): None,  # str
    ("Model", "Teacher"): None,  # str
    ("Parameter inferred", "b"): None,   # float
    ("Parameter inferred", "c"): None,   # float
    ("Parameter inferred", "m"): None,   # float
    ("Parameter inferred", "s"): None,   # float
    ("Parameter inferred", "tau"): None, # float
    ("Parameter inferred", "x"): None,   # float
    ("Parameter real", "b"): None,  # float
    ("Parameter real", "c"): None,  # float
    ("Parameter real", "m"): None,  # float
    ("Parameter real", "s"): None,  # float
    ("Parameter real", "tau"): None,# float
    ("Parameter real", "x"): None,  # float
    ("Session", "Item ID"): None,  # int
    ("Session", "Iteration"): None,  # int
    ("Session", "Number"): None,  # int
    ("Session", "Omniscient"): None,  # bool
    ("Session", "P recall error"): None,  # float
    ("Session", "Seed"): None,  # int
    ("Session", "Success"): None,  # bool
    ("Session", "Time delta"): None  # pd.Timedelta; pd.Timedelta(now, unit="s")
    }
