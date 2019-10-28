import numpy as np

from scipy.special import logsumexp

from . classic import AdaptiveClassic

EPS = np.finfo(np.float).eps


class AdaptiveRevised(AdaptiveClassic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _update_mutual_info(self):

        # If there is no need to update mutual information, it ends.
        if not self.flag_update_mutual_info:
            return

        # Update prior
        self.log_prior = self.post.copy()  # Shape (len_grid,)

        # Calculate the marginal entropy and conditional entropy.
        self.ent_marg = - np.sum(
            np.exp(self.log_prior) * self.log_prior)  # shape 1

        lp = self.log_prior.reshape((1, len(self.log_prior), 1))
        self.new_log_post = self.log_lik + lp
        self.new_log_post -= logsumexp(self.new_log_post)
        # shape (num_designs, num_params, num_replies

        self.ent_cond = - np.sum(np.sum(
            np.exp(self.new_log_post) * self.new_log_post, axis=1), axis=-1)
        # shape (num_designs,)

        # Calculate the mutual information.
        self.mutual_info = self.ent_marg - \
            self.ent_cond  # shape (num_designs,)

        # Flag that there is no need to update mutual information again.
        self.flag_update_mutual_info = False
