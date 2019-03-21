class Param:

    def __init__(self, **kwargs):
        print(kwargs)
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def main():

    p = Param(**{"k": 1, "b": 2})
    print(p.k)
    print(p.b)

if __name__ == "__main__":

    main()