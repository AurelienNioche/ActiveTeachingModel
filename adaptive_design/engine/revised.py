import numpy as np

from scipy.special import logsumexp

from . classic import AdaptiveClassic

EPS = np.finfo(np.float).eps


class AdaptiveRevised(AdaptiveClassic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_lik_t0_t1 = \
            np.zeros((len(self.possible_design),
                      len(self.grid_param), 4))

        self.y = np.arange(2)

    @staticmethod
    def log_lik_bernoulli(y, p):
        """Log likelihood for a Bernoulli random variable"""
        return y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)

    def _compute_log_lik(self):
        """Compute the log likelihood."""

        for j, param in enumerate(self.grid_param):

            for i, x in enumerate(self.possible_design):

                learner = self.learner_model(
                    t=len(self.hist),
                    hist=self.hist,
                    param=param)
                p = learner.p_recall(item=x)
                learner.learn(item=x)
                p_t2 = learner.p_recall(item=x)

                log_p = self.log_lik_bernoulli(self.y, p)
                log_p_t2 = self.log_lik_bernoulli(self.y, p_t2)

                self.log_lik[i, j, :] = log_p
                self.log_lik_t0_t1[i, j, :] = np.hstack((log_p, log_p_t2))

    def _update_mutual_info(self):

        # Get likelihood
        # shape (num_designs, num_params, num_responses, )
        ll = self.log_lik_t0_t1

        # Calculate the marginal log likelihood.
        # shape (num_designs, num_responses, )
        lp = self.log_post.reshape((1, len(self.log_post), 1))
        mll = logsumexp(ll + lp, axis=1)

        # Calculate the marginal entropy and conditional entropy.
        # shape (num_designs,)
        ent_mrg = - np.sum(np.exp(mll) * mll, -1)

        # Compute entropy obs -------------------------

        # shape (num_designs, num_params, num_responses, )
        # shape (num_designs, num_params, )
        ent_obs = - np.multiply(np.exp(ll), ll).sum(-1)

        # Compute conditional entropy -----------------

        # shape (num_designs,)
        ent_cond = np.sum(self.post * ent_obs, axis=1)

        # Calculate the mutual information. -----------
        # shape (num_designs,)
        self.mutual_info = ent_mrg - ent_cond
