Config:
    "config"         name of the config file                            str
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
    "psy_pr_lab" teacher parameters labels                              np.ndarray(n_param_teacher, dtype=string)
    "psy_pr_val" teacher param values                                   np.ndarray(n_param_teacher, dtype=float)
    "pr_lab"         learner parameters labels                          np.ndarray(n_param_learner, dtype=string)
    "pr_val"         true learner parameters                            np.ndarray(n_param_learner, dtype=float)


Results:
	"config"         name of the config file                            str
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
    "psy_pr_lab" teacher parameters labels                              np.ndarray(n_param_teacher, dtype=string)
    "psy_pr_val" teacher param values                                   np.ndarray(n_param_teacher, dtype=float)
    "pr_lab"         learner parameters labels                          np.ndarray(n_param_learner, dtype=string)
    "pr_val"         true learner parameters                            np.ndarray(n_param_learner, dtype=float)
 ----------------------------------------------------------------------------------------------------------
	
   "pr_inf"         inferred learner parameters                        np.ndarray(n_param_learner, dtype=float)
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
    "is_human"       true if human                                      bool
	"agent"          id of the the agent                                str
	
	
	
