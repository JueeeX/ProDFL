import torch
import torch.nn as nn
import torch.optim as optim


# --------- Confidence Matrix Generation --------- #
def compute_confidence_matrix(model, dataloader, loss_fn, s_t, batch_size, E, xi):
    """
    Function to compute the Confidence Matrix for a client based on the optimization problem in the prompt.

    model: The trained model (LeNet-5, ResNet-18, etc.)
    dataloader: The data loader for the dataset (MNIST, CIFAR-10, CIFAR-100)
    loss_fn: The loss function used for training (CrossEntropyLoss)
    s_t: The scaling factor to amplify differences between models
    batch_size: Batch size used in training
    E: Total number of local training rounds
    xi: Mini-batch size used during the SGD optimization
    """

    # Ensure model is in evaluation mode
    model.eval()

    # Initialize an empty list to store the gradients or differences (for confidence matrix)
    confidence_matrix = []

    # Record the initial model parameters to track the change over training epochs
    initial_model_params = {name: param.clone() for name, param in model.named_parameters()}

    # Iterate over the training data for E epochs
    for epoch in range(E):
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()  # Move to GPU if necessary

            # Perform a forward pass
            outputs = model(inputs)

            # Compute the loss and perform a backward pass to get gradients
            loss = loss_fn(outputs, targets)
            model.zero_grad()
            loss.backward()

            # Store gradients after the backward pass (for confidence matrix computation)
            gradients = {name: param.grad.clone() for name, param in model.named_parameters()}

            # Now compute the model parameter differences between this epoch and the initial epoch
            model_diff = {}
            for name, param in model.named_parameters():
                initial_param = initial_model_params[name]
                # Compute the parameter difference
                model_diff[name] = s_t * (param - initial_param) / (E * len(dataloader.dataset) / xi)

            # Store the model differences (this can be used as a "confidence matrix" or for optimization)
            confidence_matrix.append(model_diff)

        # After each epoch, update the initial_model_params for the next iteration
        initial_model_params = {name: param.clone() for name, param in model.named_parameters()}

    # Convert the confidence matrix into a tensor format (for easier further processing)
    # Here we convert each gradient or model difference into a tensor and stack them
    stacked_conf_matrix = []
    for model_diff in confidence_matrix:
        stacked_model_diff = torch.cat([param.view(-1) for param in model_diff.values()])
        stacked_conf_matrix.append(stacked_model_diff)

    # Return the stacked confidence matrix
    return torch.stack(stacked_conf_matrix)
