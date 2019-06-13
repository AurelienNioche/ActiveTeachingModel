import googletrans

TR = googletrans.Translator()


def translate(word, src='ja', dest='en'):

    return TR.translate(word, src=src, dest=dest).text


def main():

    print(translate('leer', 'es'))


if __name__ == "__main__":

    main()
