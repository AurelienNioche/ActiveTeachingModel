import settings.paths as paths

from plot import fake_data
from plot.subplot import chocolate, box, efficiency_cost, p_recall_error


def main() -> None:
    """Plot figures with fake data"""

    NUM_AGENTS = 30
    T_TOTAL = 300

    LEARNERS = frozenset({"exponential", "walsh2018"})
    PSYCHOLOGISTS = frozenset({"bayesian", "black_box"})
    TEACHERS = frozenset({"leitner", "threshold", "sampling"})

    BKP_DIR = paths.BKP_DIR

    # Fake data
    models = (LEARNERS, PSYCHOLOGISTS, TEACHERS)
    cumulative_df = fake_data.prepare_data_chocolate(
        1.3, 1.2, models, NUM_AGENTS, paths.BKP_DIR, force_save=False,
    )

    primary_df = fake_data.make_primary_df(
        models, NUM_AGENTS, T_TOTAL, BKP_DIR, force_save=False,
    )

    # Individual plots
    chocolate.plot(TEACHERS, LEARNERS, PSYCHOLOGISTS, cumulative_df, paths.FIG_DIR)
    box.plot(cumulative_df, paths.FIG_DIR)
    efficiency_cost.plot(cumulative_df, paths.FIG_DIR)
    p_recall_error.plot(TEACHERS, LEARNERS, PSYCHOLOGISTS, primary_df, paths.FIG_DIR)


main()

#%%
if __name__ == "__main__":
    main()

# MODELS = frozenset({"Learner", "Psychologist", "Teacher"})
# NUM_CONDITIONS = NUM_TEACHERS + NUM_LEARNERS + NUM_PSYCHOLOGISTS
# NUM_OBSERVATIONS = NUM_CONDITIONS * NUM_AGENTS
# NUM_TEACHERS = len(TEACHERS)
# NUM_LEARNERS = len(LEARNERS)
# NUM_PSYCHOLOGISTS = len(PSYCHOLOGISTS)
