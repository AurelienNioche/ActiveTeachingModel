import os
import json
import numpy as np

from model.teacher.leitner import Leitner
from model.teacher.myopic import Myopic
from model.teacher.conservative import Conservative
from model.teacher.robust import Robust

from model.psychologist.psychologist_grid import PsyGrid

from model.learner.exponential import Exponential


TEACHER = {
    Myopic.__name__: Myopic,
    Leitner.__name__: Leitner,
    Conservative.__name__: Conservative,
    Robust.__name__: Robust
}

LEARNER = {
    Exponential.__name__: Exponential
}

PSYCHOLOGIST = {
    PsyGrid.__name__: PsyGrid
}


class Config:
    def __init__(
        self,
        data_folder,
        config_file,
        config_dic,
        seed,
        agent,
        bounds,
        md_learner,
        md_psy,
        md_teacher,
        omni,
        n_item,
        is_item_specific,
        ss_n_iter,
        time_between_ss,
        n_ss,
        learnt_threshold,
        time_per_iter,
        cst_time,
        teacher_pr_lab,
        teacher_pr_val,
        psy_pr_lab,
        psy_pr_val,
        pr_lab,
        pr_val
    ):

        self.seed = seed
        self.agent = agent

        self.teacher_cls = TEACHER[md_teacher]
        self.learner_cls = LEARNER[md_learner]
        self.psy_cls = PSYCHOLOGIST[md_psy] \
            if md_psy is not None else None

        self.bounds = np.asarray(bounds)

        self.param = np.asarray(pr_val)
        self.param_labels = np.asarray(pr_lab)

        self.n_item = n_item
        self.omniscient = omni

        self.is_item_specific = is_item_specific
        self.time_between_ss = time_between_ss
        self.time_per_iter = time_per_iter
        self.n_ss = n_ss
        self.ss_n_iter = ss_n_iter
        self.learnt_threshold = learnt_threshold
        self.cst_time = cst_time

        if psy_pr_lab is not None and len(psy_pr_lab):
            self.psy_pr = {k: v
                           for k, v in zip(psy_pr_lab, psy_pr_val)}
        else:
            self.psy_pr = None
        self.teacher_pr = {k: v
                           for k, v in zip(teacher_pr_lab, teacher_pr_val)}

        self.config_file = config_file
        self.config_dic = config_dic
        self.data_folder = data_folder

    @classmethod
    def get(cls, file):

        with open(file) as f:
            cf_from_file = json.load(f)
            cf = cls(
                config_file=os.path.basename(file),
                config_dic=cf_from_file,
                **cf_from_file,
            )
        return cf
