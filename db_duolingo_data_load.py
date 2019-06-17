import os

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Your application specific imports
from duolingo.models import Item

from time import time
import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.utils import AskUser

RAW_DATA = "./data/duolingo.csv"


@AskUser
def main():
    # Load Data
    print('Loading raw data...', end=" ", flush=True)
    a = time()
    df_duo = pd.read_csv(RAW_DATA)
    b = time()
    print(f"Done in {str(datetime.timedelta(seconds=b-a))}!")

    n_lines = len(df_duo["p_recall"])
    n_subjects = len(np.unique(df_duo["user_id"]))
    print(f'Number of lines: {n_lines}\n'
          f'Number of subjects: {n_subjects}')

    # items = []
    for i in tqdm(range(n_lines)):

        it = Item(
            p_recall=df_duo['p_recall'][i],
            timestamp=df_duo['timestamp'][i],
            delta=df_duo['delta'][i],
            user_id=df_duo['user_id'][i],
            learning_language=df_duo['learning_language'][i],
            ui_language=df_duo['ui_language'][i],
            lexeme_id=df_duo['lexeme_id'][i],
            lexeme_string=df_duo['lexeme_string'][i],
            history_seen=df_duo['history_seen'][i],
            history_correct=df_duo['history_correct'][i],
            session_seen=df_duo['session_seen'][i],
            session_correct=df_duo['session_correct'][i],
        )
        # items.append(it)
        it.save(force_insert=True)

    # Call bulk_create to create records in a single call
    # Item.objects.bulk_create(items)


if __name__ == "__main__":
    main()
