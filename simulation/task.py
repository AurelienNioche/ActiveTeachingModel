import os
import pickle
import uuid

import numpy as np

import similarity_graphic.measure
import similarity_semantic.measure
from task.parameters import N_POSSIBLE_REPLIES

BKP_FOLDER = os.path.join('bkp', 'learning_material')

os.makedirs(BKP_FOLDER, exist_ok=True)


class Task:

    def __init__(self, t_max=100, n_kanji=20, grades=(1, ),
                 verbose=False, seed=123, compute_similarity=True,
                 normalize_similarity=False,
                 generate_full_task=False, force=False):

        # Seed
        np.random.seed(seed)

        self.n_item = n_kanji
        self.n_possible_replies = N_POSSIBLE_REPLIES
        self.t_max = t_max
        self.grades = grades

        self.kanji, self.meaning = \
            self.get_kanji_and_meaning()

        if compute_similarity:
            self.c_semantic = \
                self.get_semantic_connection(
                    normalize=normalize_similarity,
                    verbose=verbose,
                    force=force
                )
            self.c_graphic = \
                self.get_graphic_connection(
                    normalize=normalize_similarity,
                    verbose=verbose,
                    force=force
                )
        if verbose:
            print(f"Kanjis used are: {self.kanji}\n")
            print(f"Meanings used are: {self.meaning}\n")

        if generate_full_task:
            self.question_list, self.correct_answer_list,\
                self.possible_replies_list = self.generate_full_task()

    def get_kanji_and_meaning(self):

        # Backup file
        bkp_file = f"{BKP_FOLDER}/kanji_meaning_n{self.n_item}_" \
            f"g{'_'.join([str(i) for i in self.grades])}.p"
        if os.path.exists(bkp_file):
            print(f"I will use the backup file '{bkp_file}'.")
            kanji, meaning = pickle.load(open(bkp_file, 'rb'))
            return kanji, meaning

        print(f"I was unable to find the backup file '{bkp_file}'.")
        print("I will produce the data and create it.")

        # Django specific settings
        os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                              "ActiveTeachingModel.settings")
        # Ensure settings are read
        from django.core.wsgi import get_wsgi_application
        application = get_wsgi_application()

        # Your application specific imports
        from task.models import Kanji

        # Seed
        np.random.seed(123)

        # Select n kanji among first grade
        k = list(Kanji.objects.filter(grade__in=self.grades).order_by('id'))

        while True:

            # Select randomly n kanji
            kanji_idx = np.random.choice(np.arange(len(k)), size=self.n_item,
                                         replace=False)

            # Get the meaning
            meaning = [k[kanji_idx[i]].meaning for i in range(self.n_item)]

            # Ensure that each kanji has a different meaning
            if len(np.unique(meaning)) == len(meaning):
                break

        # Get the kanji
        kanji = [k[kanji_idx[i]].kanji for i in range(self.n_item)]

        kanji = np.asarray(kanji)
        meaning = np.asarray(meaning)

        pickle.dump((kanji, meaning), open(bkp_file, 'wb'))

        return kanji, meaning

    def get_graphic_connection(self, normalize=True, force=False,
                               verbose=False):

        # Create ID for kanji list
        list_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{self.kanji}"))

        # Backup file
        bkp_file = os.path.join(
            BKP_FOLDER,
            f"connect_graphic_norm_{normalize}_{len(self.kanji)}_{list_id}.p")

        if os.path.exists(bkp_file) and not force:
            print(f"I will load graphic connection from '{bkp_file}'.")
            return pickle.load(file=open(bkp_file, 'rb'))

        print(f"I will compute the graphic connections "
              f"and save it to '{bkp_file}'.")

        sg = similarity_graphic.measure.get(
                self.kanji, normalize_similarity=normalize,
                verbose=verbose)

        pickle.dump(sg, open(bkp_file, 'wb'))
        return sg

    def get_semantic_connection(self, normalize=True, force=False,
                                verbose=False):

        # Create ID for meaning list
        word_list = [i.lower() for i in self.meaning]
        list_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{word_list}"))

        # Backup file
        bkp_file = os.path.join(
            BKP_FOLDER,
            f"connect_semantic_norm_{normalize}_{len(word_list)}_{list_id}.p")

        if os.path.exists(bkp_file) and not force:
            print(f"I will load semantic connection from '{bkp_file}'.")
            return pickle.load(file=open(bkp_file, 'rb'))

        print(f"I will compute the semantic connections "
              f"and save it to '{bkp_file}'.")

        ss = similarity_semantic.measure.get(
                self.meaning, normalize_similarity=normalize,
                verbose=verbose)

        pickle.dump(ss, open(bkp_file, 'wb'))
        return ss

    def generate_full_task(self):

        # Define probability for a kanji to be selected
        p = np.random.random(self.n_item)
        p /= p.sum()

        q_idx = np.random.choice(np.arange(self.n_item), size=self.t_max,
                                 p=p, replace=True)

        question_list = []
        correct_answer_list = []
        possible_replies_list = []

        for t in range(self.t_max):

            # Get question and correct answer
            question = self.kanji[q_idx[t]]
            correct_answer = self.meaning[q_idx[t]]

            # Select possible replies
            meaning_without_correct = list(self.meaning.copy())
            meaning_without_correct.remove(correct_answer)

            possible_replies = \
                [correct_answer, ] + \
                list(np.random.choice(meaning_without_correct,
                                      size=self.n_possible_replies - 1,
                                      replace=False))

            # Randomize the order of possible replies
            np.random.shuffle(possible_replies)

            question_list.append(question)
            correct_answer_list.append(correct_answer)
            possible_replies_list.append(possible_replies)

        return question_list, correct_answer_list, possible_replies_list
