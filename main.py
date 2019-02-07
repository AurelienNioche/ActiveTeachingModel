from task.run import run
from graph import plot


def main():

    success = run()
    plot.success(success)


if __name__ == "__main__":

    main()
