#%%
"""
Test the values for Walsh 2018
"""

import numpy as np

eps

tau = 0.5, 1.5
s = 0.0, 0.1
b = 0.0, 0.2
m = 0.0, 0.2
c = 0.1, 0.1
x = 0.6, 0.6

p_n = 1 / (1 + np.exp((tau - m_n) / s))
m_n = n ** c * t_n ** -d
t_n = sum(w * t)
w = t ** -x / (sum(t ** -x))
d = b + m * (1 / (n - 1) * sum(1 / np.log(lag + np.e)))

print(f"Time elapsed since practice: {t}")

    def p(self, item, param, now, is_item_specific, cst_time):

        if is_item_specific:
            tau, s, b, m, c, x = param[item]

        else:
            tau, s, b, m, c, x = param

        relevant = self.hist == item
        rep = self.ts[relevant]

        rep *= cst_time
        now *= cst_time

        n = len(rep)
        delta = now - rep

        if n == 0:
            return 0
        elif np.min(delta) == 0:
            return 1
        else:

            w = delta ** -x
            w /= np.sum(w)

            _t_ = np.sum(w * delta)
            if n > 1:
                lag = rep[1:] - rep[:-1]
                d = b + m * np.mean(1 / np.log(lag + np.e))
            else:
                d = b

            _m_ = n ** c * _t_ ** -d
            with np.errstate(divide="ignore", invalid="ignore"):
                v = (-tau + _m_) / s
            p = expit(v)
            return p
