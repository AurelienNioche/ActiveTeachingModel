import os


def dump():
    os.system(
        'pg_dump --data-only  --table item --verbose ActiveTeaching '
        '--inserts > data/duolinguo.sql')


if __name__ == "__main__":
    dump()
