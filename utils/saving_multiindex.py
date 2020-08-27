# def get_saving_dict() -> dict:
#     """
#     Make dictionary that can appended to df.
#     Remember to change the parameters according to the model.
#
#     Example to add to df:
#         saving_df.loc[last_idx+1] = this_dict
#     """
#
#     return {
#     ("Agent", "Human"): None,  # bool
#     ("Agent", "ID"): None,    # int
#     ("Bounds", "b"): None,    # np.ndarray(float)
#     ("Bounds", "c"): None,    # np.ndarray(float)
#     ("Bounds", "m"): None,    # np.ndarray(float)
#     ("Bounds", "s"): None,    # np.ndarray(float)
#     ("Bounds", "tau"): None,  # np.ndarray(float)
#     ("Bounds", "x"): None,    # np.ndarray(float)
#     ("Model", "Learner"): None,  # str
#     ("Model", "Psychologist"): None,  # str
#     ("Model", "Teacher"): None,  # str
#     ("Parameter inferred", "b"): None,   # float
#     ("Parameter inferred", "c"): None,   # float
#     ("Parameter inferred", "m"): None,   # float
#     ("Parameter inferred", "s"): None,   # float
#     ("Parameter inferred", "tau"): None, # float
#     ("Parameter inferred", "x"): None,   # float
#     ("Parameter real", "b"): None,  # float
#     ("Parameter real", "c"): None,  # float
#     ("Parameter real", "m"): None,  # float
#     ("Parameter real", "s"): None,  # float
#     ("Parameter real", "tau"): None,# float
#     ("Parameter real", "x"): None,  # float
#     ("Session", "Item ID"): None,  # int
#     ("Session", "Iteration"): None,  # int
#     ("Session", "Number"): None,  # int
#     ("Session", "Omniscient"): None,  # bool
#     ("Session", "P recall error"): None,  # float
#     ("Session", "Seed"): None,  # int
#     ("Session", "Success"): None,  # bool
#     ("Session", "Time delta"): None  # pd.Timedelta; pd.Timedelta(now, unit="s")
#     }


def get_saving_dict() -> dict:
    """
    Make dictionary that can appended to df.
    Remember to change the parameters according to the model.

    Example to add to df:
        saving_df.loc[last_idx+1] = this_dict
    """

    """
    "is_human"       true if human                                      bool
    "agent"          id of the the agent                                int
    "bounds"         bounds used by the psychologist                    np.ndarray((n_param, 2)) dtype=float)
    "md_learner"     class of the learner                               str
    "md_psy"         class of the psychologist                          str
    "md_teacher"     class of the teacher                               str
    "omni"           true if the psychologist know the true parameter   bool
    "n_item"         number of item to (possibly) learn                 int
    "task_pr_lab"    task param labels                                  np.ndarray(n_param_task, dtype=string)
    "task_pr_val"    task param values                                  np.ndarray(n_param_task, dtype=float)
    "teacher_pr_lab" teacher parameters labels                          np.ndarray(n_param_teacher, dtype=string)
    "teacher_pr_val" teacher param values                               np.ndarray(n_param_teacher, dtype=float)
    "pr_lab"         learner parameters labels                          np.ndarray(n_param_learner, dtype=string)
    "pr_inf"          inferred learner parameters                       np.ndarray(n_param_learner, dtype=float)
    "pr_real"        true learner parameters                            np.ndarray(n_param_learner, dtype=float)
    "item"           index of the item                                  int
    "iter"           index of the iteration                             int
    "ss_iter"        session relative index of the iteration            int
    "ss_idx"         index of the session                               int
    "p_err_mean"     average error of prediction per item               float
    "p_err_std"      std error of prediction per item                   float
    "success"        true if recall                                     bool
    "n_learnt"       number of item such a p > learnt threshold         int
    "timestamp"      fake timestamp using second as a unit for task     int
    "timestamp_cpt"  real timestamp using second as a unit              int
    "config"         name of the config file                            str
    
    """

    return {
    # "is_human": None,
    # "agent": None,
    # "bounds": None,
    # "md_learner": None,
    # "md_psy": None,
    # "md_teacher": None,
    # "omni": None,
    # "param_labels": None,
    # "param_inf": None,
    # "param_real": None,
    # "item": None,
    # "iter": None,
    # "ss_iter": None,
    # "ss_idx": None,
    # "p_err": None,
    # "seed": None,
    # "success": None,
    # "timestamp": None
}
