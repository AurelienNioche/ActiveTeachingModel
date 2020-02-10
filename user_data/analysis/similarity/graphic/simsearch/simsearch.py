"""
Adapted from https://github.com/larsyencken/simsearch
Related paper: https://www.aclweb.org/anthology/C08-1131
"""

import os

import numpy as np

# static data files needed for building
DATA_DIR = os.path.join('user_data', 'bkp', 'simsearch')

# The source of stroke data for each character
STROKE_SOURCE = os.path.join(DATA_DIR, 'stroke_ulrich')

assert os.path.exists(STROKE_SOURCE), \
    "Error of configuration for the 'stroke ulrich' file..."


class StrokeSimilarity:
    """The similarity between two kanji depending on the edit distance
    between stroke sequences for both kanji."""

    def __init__(self):
        self.stroke_types = {}
        self.n_stroke_types = 0

        self.signatures = {}
        with open(STROKE_SOURCE, 'rb') as i_stream:
            for i, line in enumerate(i_stream):
                line = line.decode()
                kanji, raw_strokes = line.rstrip().split()

                # Decode from bytes
                # kanji = kanji.decode()
                # raw_strokes = raw_strokes.decode()

                raw_strokes = raw_strokes.split(',')
                strokes = list(map(self.get_stroke_type, raw_strokes))
                self.signatures[kanji] = strokes

    def get_stroke_type(self, stroke):
        try:
            return self.stroke_types[stroke]
        except KeyError:
            pass

        self.stroke_types[stroke] = self.n_stroke_types
        self.n_stroke_types = self.n_stroke_types + 1

        return self.n_stroke_types - 1

    def raw_distance(self, kanji_a, kanji_b):
        s_py = self.signatures[kanji_a]
        t_py = self.signatures[kanji_b]

        return self.edit_distance(s_py, t_py)

    def __call__(self, kanji_a, kanji_b):
        s_py = self.signatures[kanji_a]
        t_py = self.signatures[kanji_b]
        result = self.edit_distance(s_py, t_py)
        return 1 - float(result) / max(len(s_py), len(t_py))

    def __contains__(self, kanji):
        return kanji in self.signatures

# ----------------------------------------------------------------------------#

    @classmethod
    def edit_distance(cls, s_py, t_py):

        table = np.zeros((100, 100))
        s = np.zeros(100)
        t = np.zeros(100)

        s_len = len(s_py)
        t_len = len(t_py)
        if s_len > 99 or t_len > 99:
            raise (ValueError, "stroke sequences too long")

        for i in range(s_len):
            table[i, 0] = i
            s[i] = s_py[i]
        table[s_len, 0] = s_len

        for j in range(t_len):
            table[0, j] = j
            t[j] = t_py[j]
        table[0, t_len] = t_len

        for i in range(1, s_len + 1):
            for j in range(1, t_len + 1):
                if s[i - 1] == t[j - 1]:
                    cost = 0
                else:
                    cost = 1

                up = table[i - 1, j] + 1
                left = table[i, j - 1] + 1
                diag = table[i - 1, j - 1] + cost
                if up <= left:
                    if up <= diag:
                        table[i, j] = up
                    else:
                        table[i, j] = diag
                else:
                    if left <= diag:
                        table[i, j] = left
                    else:
                        table[i, j] = diag

        return table[s_len, t_len]

#
# def get_similarity():
#
#     return StrokeSimilarity()


# def demo():
#
#     s = StrokeSimilarity()
#
#     for i, j in combinations(['八', '花', '虫', '中',
#     '王', '足', '生', '力', '七', '二'], 2):
#         print(i, j, s(i, j))
#
#
# if __name__ == "__main__":
#
#     demo()
