import os
import numpy as np
import itertools
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


def calculate_prob(alpha, beta, delta, n_pres):
    """A function to compute the probability of a positive response."""
    p_obs = np.exp(-delta*alpha*(n_pres-1)*beta)
    return p_obs


def main():

    np.random.seed(123)

    n_design = 1000
    bounds_design = 1, n_design
    design = np.linspace(*bounds_design, n_design)

    bounds = ((2e-07, 0.025), (0.0001, 0.9999))

    n_grid_param = 10
    methods = np.geomspace, np.linspace
    grid = np.array(list(itertools.product(
        *(m(*b, n_grid_param) for m, b in zip(methods, bounds)))))

    n_trial = 100

    param = [0.01, 0.02]

    n_param_set, n_param = grid.shape

    engine = ('random', 'ado')
    n_engine = len(engine)

    means = {e: np.zeros((n_param, n_trial)) for e in engine}
    stds = {e: np.zeros((n_param, n_trial)) for e in engine}

    # random ------------------------------------------------------------

    lp = np.ones(n_param_set)
    lp -= logsumexp(lp)

    for t in tqdm(range(1, n_trial), file=sys.stdout):

        # Select design
        d_idx = np.random.randint(n_design)
        d = design[d_idx]

        # Get response
        p_r = calculate_prob(delta=d,
                             alpha=param[0],
                             beta=param[1],
                             n_pres=t)
        resp = p_r > np.random.random()

        # Compute likelihood; shape: n param set
        p_one = calculate_prob(delta=d,
                               alpha=grid[:, 0],
                               beta=grid[:, 1],
                               n_pres=t)
        p = p_one if resp else 1 - p_one
        log_lik_r = np.log(p + np.finfo(np.float).eps)

        # update prior
        lp += log_lik_r
        lp -= logsumexp(lp)

        # Compute expected value
        ep = np.dot(np.exp(lp), grid)

        # Compute sd
        delta = grid - ep
        post_cov = np.dot(delta.T, delta * np.exp(lp).reshape(-1, 1))
        sdp = np.sqrt(np.diag(post_cov))

        means['random'][:, t] = ep
        stds['random'][:, t] = sdp

    # ado ----------------------------------------------------------------

    lp = np.ones(n_param_set)
    lp -= logsumexp(lp)

    for t in tqdm(range(1, n_trial), file=sys.stdout):

        post = np.exp(lp)

        # Compute likelihood
        p_one = np.array([calculate_prob(delta=d,
                                         alpha=grid[:, 0],
                                         beta=grid[:, 1],
                                         n_pres=t)
                          for d in design])  # shape: (n design, n param set)
        p_zero = 1 - p_one
        p = np.zeros((n_design, n_param_set, 2))
        p[:, :, 0] = p_zero
        p[:, :, 1] = p_one
        log_lik = np.log(p + np.finfo(np.float).eps)

        # Compute entropy of LL
        ent_obs = -np.multiply(np.exp(log_lik), log_lik).sum(-1)

        # Calculate the marginal log likelihood
        # shape (num_design, num_response)
        extended_lp = np.expand_dims(np.expand_dims(lp, 0), -1)
        mll = logsumexp(log_lik + extended_lp, axis=1)

        # Calculate the marginal entropy and conditional entropy.
        ent_marg = -np.sum(np.exp(mll) * mll, -1)  # shape (num_designs,)
        ent_cond = np.sum(post * ent_obs, axis=1)  # shape (num_designs,)

        # Calculate the mutual information.
        mutual_info = ent_marg - ent_cond  # shape (num_designs,)

        # Select design
        d_idx = np.argmax(mutual_info)
        d = design[d_idx]

        # Produce response
        p_r = calculate_prob(delta=d,
                             alpha=param[0],
                             beta=param[1],
                             n_pres=t)
        resp = p_r > np.random.random()

        # Get likelihood response
        log_lik_r = log_lik[d_idx, :, int(resp)].flatten()

        # Update prior
        lp += log_lik_r
        lp -= logsumexp(lp)

        # Expected value
        ep = np.dot(np.exp(lp), grid)

        # Sd
        delta = grid - ep
        post_cov = np.dot(delta.T, delta * np.exp(lp).reshape(-1, 1))
        sdp = np.sqrt(np.diag(post_cov))

        means['ado'][:, t] = ep
        stds['ado'][:, t] = sdp

    # Figure ----------------------------------------------

    fig, axes = plt.subplots(ncols=n_param, figsize=(12, 6))

    colors = [f'C{i}' for i in range(n_engine)]

    for i, ax in enumerate(axes):

        for j, e in enumerate(engine):

            _means = means[e][i, :]
            _stds = stds[e][i, :]

            true_p = param[i]
            ax.axhline(true_p, linestyle='--', color='black',
                       alpha=.2)

            c = colors[j]

            ax.plot(_means, color=c, label=e)
            ax.fill_between(range(n_trial), _means - _stds,
                            _means + _stds, alpha=.2, color=colors[j])

            ax.set_title(f'b{i}')
            ax.set_xlabel("trial")
            ax.set_ylabel(f"value")

    plt.legend()
    plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig(os.path.join("fig", "ado_vs_random_exp_decay_with_rep.pdf"))


if __name__ == "__main__":
    main()

