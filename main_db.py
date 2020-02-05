import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from tqdm import tqdm

from simulation_data.models import Simulation, Post

from model.teacher.leitner import Leitner
from model.simplified.teacher import Teacher

# from main import run_n_days
from model.settings import BKP_FOLDER
import glob
import pickle

from model.constants import \
    POST_MEAN, POST_SD, N_SEEN, HIST, SUCCESS, TIMESTAMP


def main():

    def load(f_name):
        return pickle.load(open(f_name, 'rb'))

    def file_name(suffix):
        return os.path.join(BKP_FOLDER, "run_n_days", f"{suffix}.p")

    def extract_id(f_name):
        return int("".join([s for s in
                            os.path.basename(f_name) if s.isdigit()]))

    print(f"Look for {file_name('*_info')}")

    info_files = glob.glob(file_name("*_info"))

    print(info_files)

    for info_file in tqdm(info_files):

        info_loaded = load(info_file)
        idx = extract_id(info_file)
        data = load(file_name(f"{idx}_data"))

        n_session = info_loaded["n_day"]
        learner_model = info_loaded["learner"]
        n_item = info_loaded["n_item"]
        grid_size = info_loaded["grid_size"]
        param = info_loaded["param"]
        seed = info_loaded["seed"]

        n_iteration_per_session = info_loaded["n_iter_session"]
        bounds = info_loaded["bounds"]
        if bounds is None:
            bounds = info_loaded["learner"].bounds
        param_labels = info_loaded["param_labels"]
        if param_labels is None:
            param_labels = info_loaded["learner"].param_labels

        try:
            sec_per_iter = info_loaded["sec_per_iter"]
        except KeyError:
            sec_per_iter = 2

        n_iteration_between_session = \
            int((60 ** 2 * 24) /
                sec_per_iter - n_iteration_per_session)

        condition_label = info_loaded["condition"]

        if condition_label == "Leitner":
            teacher_model = Leitner
        elif condition_label == "Teacher":
            teacher_model = Teacher
        else:
            raise ValueError(f"'{condition_label}' not recognized")

        n_seen = data[N_SEEN]
        post_means = data[POST_MEAN]
        post_sds = data[POST_SD]
        success = data[SUCCESS]
        hist = data[HIST]
        timestamp = data[TIMESTAMP]

        sim_entry = Simulation.objects.create(
            n_item=n_item,
            n_session=n_session,
            n_iteration_per_session=n_iteration_per_session,
            n_iteration_between_session=n_iteration_between_session,
            teacher_model=teacher_model.__name__,
            learner_model=learner_model.__name__,
            param_labels=list(param_labels),
            param_values=list(param),
            param_upper_bounds=[b[0] for b in bounds],
            param_lower_bounds=[b[1] for b in bounds],
            grid_size=grid_size,
            timestamp=list(timestamp),
            hist=list(hist),
            success=list(success),
            n_seen=list(n_seen),
            seed=seed)

        post_entries = []

        for i, pr in enumerate(param_labels):
            post_entries.append(
                Post(
                    simulation=sim_entry,
                    param_label=pr,
                    std=list(post_sds[pr]),
                    mean=list(post_means[pr])))

        Post.objects.bulk_create(post_entries)


if __name__ == "__main__":

    main()
