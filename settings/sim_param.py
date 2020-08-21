"""
To generate the config files
"""

task_param = {
    "is_item_specific": True,
    "n_item": 100,
    "ss_n_iter": 100,
    "ss_n_iter_between": 86200,
    "n_ss": 10,
    "thr": 0.9,
    "time_per_iter": 2,
}

sampling_cst = {"horizon": 100, "n_sample": 500}

leitner_cst = {"delay_factor": 2, "delay_min": 2}

walsh_cst = {
    "param_labels": ["tau", "s", "b", "m", "c", "x"],
    "bounds": [
        [0.5, 1.5],
        [0.005, 0.10],
        [0.005, 0.2],
        [0.005, 0.2],
        [0.1, 0.1],
        [0.6, 0.6],
    ],
    "learner_model": "walsh2018",
    "grid_size": 10,
}

exp_decay_cst = {
    "param_labels": ["alpha", "beta"],
    "bounds": [[0.001, 0.2], [0.00, 0.5]],
    "learner_model": "exp_decay",
    "grid_size": 20,
}
