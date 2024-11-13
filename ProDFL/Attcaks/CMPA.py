import numpy as np
import torch


def calculate_lambda(M_b_i, neighbors, init_lambda=1.0, threshold=0.01, max_iter=50):
    """
    Compute the optimal lambda (scaling factor) for constructing the malicious model.
    M_b_i: The benign model of the current malicious participant.
    neighbors: List of benign neighbor models.
    init_lambda: Initial scaling factor.
    threshold: Convergence threshold for lambda.
    max_iter: Maximum iterations for convergence.
    """
    lambda_i = init_lambda
    for _ in range(max_iter):
        malicious_model = M_b_i + lambda_i * -torch.sign(M_b_i)
        max_distance = max([torch.norm(M_b_i - M_b_j, p=2).item() for M_b_j in neighbors])

        # Adjust lambda to satisfy the L2 constraint
        if torch.norm(malicious_model - M_b_i, p=2) <= max_distance:
            break
        lambda_i *= 0.5  # reduce lambda
    return lambda_i


def construct_malicious_model(M_b_i, lambda_i):
    """
    Construct the malicious model using the optimal lambda.
    M_b_i: The benign model of the malicious participant.
    lambda_i: Optimal scaling factor.
    """
    perturbation = -torch.sign(M_b_i)
    M_malicious = M_b_i + lambda_i * perturbation
    return M_malicious


def collude_and_publish(malicious_models):
    """
    Collusion step: Select the model with the highest toxicity score as the collusive model.
    malicious_models: List of all malicious models constructed by colluding participants.
    """
    scores = [torch.norm(M_m - M_b, p=2).item() for M_m, M_b in malicious_models]
    max_index = np.argmax(scores)
    return malicious_models[max_index]

