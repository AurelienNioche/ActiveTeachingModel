import os


def backup_user_data():
    os.system(
        'pg_dump --data-only  --table question --table user ActiveTeaching --inserts '
        '> data/user_and_question_tables2.sql')


if __name__ == "__main__":
    backup_user_data()
