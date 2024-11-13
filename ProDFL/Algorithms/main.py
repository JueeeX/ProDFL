# main.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loader import load_mnist_data  # Only loading MNIST for now
from models import LeNet5  # Only using LeNet5 for MNIST
import numpy as np
from sign_flipping_dp import sign_flipping_dp  # Import the Sign-Flipping DP function

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
epochs = 5
learning_rate = 0.01
gamma = 0.8  # Sign flipping probability parameter

def compute_cosine_similarity(vec_a, vec_b):
    """
    Compute the cosine similarity between two vectors.
    """
    dot_product = torch.dot(vec_a, vec_b)
    norm_a = torch.norm(vec_a)
    norm_b = torch.norm(vec_b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else torch.tensor(0.0)


def compute_confidence_coulomb_force_coefficient(M_i_t, M_j_t, resultant_force_i, m=2):
    """
    Compute the Confidence Coulomb Force Coefficient (FC) based on the formula provided.
    """
    # Compute the Outlier Coulomb Force
    outlier_force, outlier_force_magnitude = compute_outlier_coulomb_force(M_i_t, M_j_t, m)

    # Compute the cosine similarity
    cos_theta = compute_cosine_similarity(outlier_force, resultant_force_i)

    # Compute the Confidence Coulomb Force Coefficient
    if outlier_force_magnitude != 0:  # Ensure the magnitude is non-zero
        fc = (1 + cos_theta) / (2 * (1 + abs(outlier_force_magnitude)))
    else:
        fc = torch.tensor(0.0)
    return fc


def compute_outlier_coulomb_force(M_i_t, M_j_t, m=2):
    """
    Compute the Outlier Coulomb Force between two confidence matrices.
    """
    # Calculate the Euclidean distance between the matrices
    distance = torch.norm(M_i_t - M_j_t)

    # Calculate the unit vector r
    r = (M_i_t - M_j_t) / distance if distance != 0 else torch.zeros_like(M_i_t)

    # Compute the electric field intensity E and the magnitude of the Coulomb force
    E = distance ** (m - 1)
    k_m = 1  # Assume k_m = 1 (constant)
    force_magnitude = k_m * E

    # Calculate the Coulomb force vector
    force = force_magnitude * r
    return force, force_magnitude


def rsa_with_fc(batch_size=64, epochs=10, lr=0.01, attack=None, save_model=False, save_results=False):
    """
    Perform RSA-based aggregation with Confidence Coulomb Force and Sign-Flipping DP.
    """
    # Load data for MNIST
    trainloader, testloader = load_mnist_data(batch_size=batch_size)  # Only MNIST data loading here

    # Initialize models and optimizer
    model = LeNet5().to(device)  # Use LeNet5 for MNIST
    model0 = LeNet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        model.train()

        # Training phase
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # Implement RSA aggregation with Confidence Coulomb Force and Sign-Flipping DP
            worker_grad = [torch.zeros_like(para) for para in model.parameters()]
            worker_model = [torch.zeros_like(para) for para in model.parameters()]

            for id in range(len(model.parameters())):
                for para, grad in zip(model.parameters(), worker_grad):
                    # Calculate FC (Confidence Coulomb Force) between workers
                    M_i_t = torch.flatten(para)
                    M_j_t = torch.flatten(model0.parameters()[id])
                    resultant_force_i = torch.zeros_like(M_i_t)
                    FC_ij = compute_confidence_coulomb_force_coefficient(M_i_t, M_j_t, resultant_force_i)

                    # Update model parameters with FC coefficient
                    para.data.add_(grad, alpha=-lr * FC_ij)

                    # Apply Sign-Flipping DP to the model updates (flip the sign of the model delta)
                    delta_w = para.data - model0.parameters()[id].data
                    flipped_sign_update = sign_flipping_dp(delta_w.cpu().numpy(), gamma)

                    # Update the model parameter with the flipped sign updates
                    para.data.add_(torch.tensor(flipped_sign_update).to(device), alpha=-lr * FC_ij)

        # Evaluate test accuracy after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Save model and results
    if save_model:
        torch.save(model.state_dict(), "rsa_mnist_model.pth")

    if save_results:
        print("Results saved.")


if __name__ == "__main__":
    rsa_with_fc(batch_size=batch_size, epochs=epochs, lr=learning_rate, save_model=True, save_results=True)
