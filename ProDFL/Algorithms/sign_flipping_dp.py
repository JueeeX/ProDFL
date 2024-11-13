import numpy as np


def sign_flipping_dp(delta_w, gamma):
    """
    Applies the Sign-Flipping Differential Privacy mechanism to the model update.

    :param delta_w: The update vector (Δω_j^t = ω_j^t - ω_j^{t-1}) for client j.
    :param gamma: The probability with which the sign is retained (0.5 < γ < 1).
    :return: The updated vector after applying the Sign-Flipping DP mechanism.
    """
    # Ensure that gamma is between 0.5 and 1
    if not (0.5 < gamma < 1):
        raise ValueError("Gamma must be between 0.5 and 1.")

    # Get the sign of the delta update
    sign_delta_w = np.sign(delta_w)

    # Flip the sign of each element with probability 1 - gamma
    flipped_signs = np.where(np.random.rand(*sign_delta_w.shape) < (1 - gamma),
                             -sign_delta_w, sign_delta_w)

    return flipped_signs
