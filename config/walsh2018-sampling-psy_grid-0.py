{
    "param": [
        1.0729407452992574,
        0.04118499354267331,
        0.15658792031547936,
        0.058873323045820214,
        0.1,
        0.6
    ],
    "param_labels": [
        "tau",
        "s",
        "b",
        "m",
        "c",
        "x"
    ],
    "is_item_specific": false,
    "n_item": 100,
    "ss_n_iter": 100,
    "ss_n_iter_between": 86200,
    "n_ss": 10,
    "thr": 0.9,
    "bounds": [
        [
            0.8,
            1.2
        ],
        [
            0.03,
            0.05
        ],
        [
            0.005,
            0.2
        ],
        [
            0.005,
            0.2
        ],
        [
            0.1,
            0.1
        ],
        [
            0.6,
            0.6
        ]
    ],
    "init_guess": null,
    "learner_model": "walsh2018",
    "psychologist_model": "psy_grid",
    "teachers": [
        "sampling"
    ],
    "omniscient": [
        false
    ],
    "time_per_iter": 2,
    "grid_size": 10,
    "leitner": {
        "delay_factor": 2,
        "delay_min": 2
    },
    "mcts": {
        "iter_limit": 500,
        "time_limit": null,
        "horizon": 50
    },
    "seed": 0
}