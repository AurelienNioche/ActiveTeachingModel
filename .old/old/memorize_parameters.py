import os

import numpy as np
import pandas as pd


def main():

    results_hlr = pd.read_csv(
        os.path.join(".old/old/data", "hlr.duolingo.weights"), index_col=1)

    print(results_hlr)

    n0 = 2**(-(results_hlr['value'][3:]+results_hlr['value'].loc['bias']))
    duo_alpha = -2**(-results_hlr['value'].loc['right'])+1
    duo_beta = 2**(-results_hlr['value'].loc['wrong'])-1

    print("alpha", duo_alpha)
    print("beta", duo_beta)
    print("n0 mean", np.mean(n0))
    print("n0 std", np.std(n0))


if __name__ == '__main__':

    main()
