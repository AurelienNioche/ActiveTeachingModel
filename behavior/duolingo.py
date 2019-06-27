import datetime
import os
from time import time

import numpy as np

from duolingo.models import Item
from utils.utils import dump, load

from behavior.data_structure import Task


class UserData:

    def __init__(self, user_id, verbose=False):

        self.user_id = user_id

        lexeme_id, lexeme_string = \
            np.asarray(Item.objects.filter(user_id=user_id)
                       .order_by('timestamp')
                       .values_list('lexeme_id', 'lexeme_string')).T

        h_seen, h_correct, s_seen, s_correct, time_stamp = \
            np.asarray(Item.objects.filter(user_id=user_id)
                       .order_by('timestamp')
                       .values_list('history_seen', 'history_correct',
                                    'session_seen', 'session_correct',
                                    'timestamp')).T

        ui_language, learning_language = \
            np.asarray(Item.objects.filter(user_id=user_id).values_list(
                'ui_language', 'learning_language'
            )).T

        self.ui_language = None
        self.learning_language = None

        if len(np.unique(ui_language)) > 1:
            if verbose:
                print('ui_language > 1!')
        elif len(np.unique(learning_language)) > 1:
            if verbose:
                print('learning_language > 1!')
        else:
            it = \
                Item.objects.filter(user_id=user_id)[0]

            self.ui_language = it.ui_language
            self.learning_language = it.learning_language

            unq_lex_id, lex_idx = np.unique(
                lexeme_id, return_inverse=True)

            n_entry = len(lexeme_id)

            self.n_item = len(unq_lex_id)

            meaning = []
            for i in range(self.n_item):
                lex_str = Item.objects\
                    .filter(lexeme_id=unq_lex_id[i])[0].lexeme_string
                m = lex_str.split('/')[0]
                if '<' in m:
                    m = lex_str.split('/')[1].split('<')[0]
                meaning.append(m)

            self.meaning = np.asarray(meaning)

            self.questions = np.zeros(n_entry, dtype=int)
            self.replies = np.zeros(n_entry, dtype=int)
            self.successes = np.zeros(n_entry, dtype=bool)
            self.times = np.zeros(n_entry, dtype=float)
            self.first_presentation = np.zeros(n_entry, dtype=bool)

            self.questions[:] = lex_idx

            self.replies[:] = -1
            self.successes[:] = s_seen == s_correct
            self.replies[self.successes] = self.questions[self.successes]
            self.times[:] = time_stamp[:]

            self.t_max = n_entry

            self.tk = Task(n_item=self.n_item,
                           t_max=n_entry+np.sum(self.first_presentation))

    @classmethod
    def load(cls, user_id, force=False, verbose=True):

        folder_path = os.path.join("data", "duolingo_user")
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"data_u{user_id}.p")

        if force or not os.path.exists(file_path):
            if verbose:
                t = time()
                print("Loading data from file...", end=" ", flush=True)
            data = cls(user_id=user_id, verbose=verbose)
            if verbose:
                print(f"Done! [time elapsed "
                      f"{datetime.timedelta(seconds=time()-t)}]")
            dump(data, file_path, verbose=verbose)
            return data

        else:
            return load(file_path, verbose=verbose)
