from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition
from autodp import mechanism_zoo, transformer_zoo
from autodp import rdp_bank
from autodp import rdp_acct
from autodp import utils

import math

from autodp.privacy_calibrator import subsample_epsdelta

from scipy.optimize import minimize_scalar

import numpy as np
import argparse


def rdp_to_approxdp(rdp, alpha_max=np.inf, BBGHS_conversion=True):
    # from RDP to approx DP
    # alpha_max is an optional input which sometimes helps avoid numerical issues
    # By default, we are using the RDP to approx-DP conversion due to BBGHS'19's Theorem 21
    # paper: https://arxiv.org/pdf/1905.09982.pdf
    # if you need to use the simpler RDP to approxDP conversion for some reason, turn the flag off

    def approxdp(delta):
        """
        approxdp outputs eps as a function of delta based on rdp calculations
        :param delta:
        :return: the \epsilon with a given delta
        """

        if delta < 0 or delta > 1:
            print("Error! delta is a probability and must be between 0 and 1")
        if delta == 0:
            return rdp(np.inf)
        else:

            def fun(x):  # the input the RDP's alpha
                if x <= 1:
                    return np.inf
                else:
                    if BBGHS_conversion:
                        return np.maximum(
                            rdp(x)
                            + np.log((x - 1) / x)
                            - (np.log(delta) + np.log(x)) / (x - 1),
                            0,
                        )
                    else:
                        return np.log(1 / delta) / (x - 1) + rdp(x)

            results = np.min([fun(alpha) for alpha in range(1, alpha_max)])
            return results

            # results = minimize_scalar(fun, method='Bounded', bounds=(1, alpha_max) )
            # if results.success:
            #     return results.fun
            # else:
            #     # There are cases when certain \delta is not feasible.
            #     # For example, let p and q be uniform the privacy R.V. is either 0 or \infty and unless all \infty
            #     # events are taken cared of by \delta, \epsilon cannot be < \infty
            #     return np.inf

    return approxdp


def approxRDP_to_approxDP(
    delta, delta0, rdp_func, alpha_max=np.inf, BBGHS_conversion=True
):
    if delta < delta0:
        return np.inf

    delta1 = delta - delta0

    approxdp = rdp_to_approxdp(rdp_func, alpha_max, BBGHS_conversion)

    return approxdp(delta1)


# eps is the privacy parameter for EM
# add Gumbel(sensitivity/eps), where sensitivity is the sensitivity of utility function
class EM(Mechanism):
    def __init__(self, eps, name="EM", monotone=False):
        Mechanism.__init__(self)
        self.name = name
        self.params = {"eps": eps}

        if monotone:

            def privloss(t, alpha):
                return (
                    np.exp(alpha * (eps - t))
                    - np.exp(-alpha * t)
                    - (
                        np.exp(alpha * eps - (alpha + 1) * t)
                        - np.exp(eps - (alpha + 1) * t)
                    )
                ) / (np.exp(eps - t) - np.exp(-t))

            def RDP_EM(alpha):
                if alpha == np.infty:
                    return eps
                enegt = ((alpha - 1) * (np.exp(alpha * eps) - 1)) / (
                    (alpha) * (np.exp(alpha * eps) - np.exp(eps))
                )
                return np.log(privloss(np.log(1 / enegt), alpha)) / (alpha - 1)

        else:

            def RDP_EM(alpha):
                if alpha == np.infty:
                    return eps * 2
                temp = (np.sinh(alpha * eps) - np.sinh((alpha - 1) * eps)) / np.sinh(
                    eps
                )
                return min(1 / 2 * alpha * eps**2, np.log(temp) / (alpha - 1))

        self.propagate_updates(RDP_EM, "RDP")


class compose_subsampled_EM(Mechanism):
    def __init__(self, eps, prob, niter, name="compose_subsampled_EM", monotone=False):
        Mechanism.__init__(self)
        self.name = name

        subsample = transformer_zoo.AmplificationBySampling()
        compose = transformer_zoo.Composition()

        mech = EM(eps, monotone=monotone)

        if prob < 1:
            mech = subsample(mech, prob, improved_bound_flag=False)

        mech = compose([mech], [niter])
        rdp_total = mech.RenyiDP

        self.propagate_updates(rdp_total, type_of_update="RDP")


# params = {eps, prob, niter}
def compose_subsampled_EM_to_approxDP(params, delta):
    mech = compose_subsampled_EM(
        params["eps"], params["prob"], params["niter"], monotone=params["monotone"]
    )
    rdp_func = mech.RenyiDP

    approxdp = rdp_to_approxdp(rdp_func, alpha_max=200, BBGHS_conversion=True)

    return approxdp(delta)


# Function to perform binary search for epsilon
def find_best_epsilon(
    target_param, privacy_params, delta, prob, eps_min=0.01, eps_max=10, tolerance=5e-3
):
    while eps_max - eps_min > tolerance:
        eps_mid = (eps_min + eps_max) / 2

        # Compute param with the current mid epsilon
        privacy_params["eps"] = eps_mid
        param_mid = compose_subsampled_EM_to_approxDP(privacy_params, delta=delta)

        # Compare the mid value to the target parameter
        if param_mid < target_param:
            eps_min = eps_mid  # If param is too small, increase epsilon
        else:
            eps_max = eps_mid  # If param is too large, decrease epsilon

    # Return the best epsilon found
    return (eps_min + eps_max) / 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find epsilon for target parameter.")
    parser.add_argument(
        "--tp",
        type=float,
        required=True,
        help="Target parameter for epsilon search.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Extract target_param from arguments
    target_param = args.tp
    delta = 1 / 3900
    prob = 400 / 3900

    privacy_params = {
        "eps": 0.237,
        "sigma": 1,
        "prob": prob,
        "niter": 100,
        "monotone": False,
    }

    epsilon = find_best_epsilon(target_param, privacy_params, delta, prob)
    print(epsilon)
