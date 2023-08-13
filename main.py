import numpy as np

from tinygrad.jit import TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.state import get_parameters
from tinygrad.tensor import Tensor
from extra.datasets import fetch_mnist

from src import FGSM_attack
from src import MNISTNet, train_model
from src import sparse_cross_entropy

# Sets the training flag to True


if __name__ == "__main__":

    # Sets variables
    n_epochs = 1000
    n_valid  = 10000

    # Retrieve dataset
    X_train, Y_train, X_test, Y_test = fetch_mnist()

    # Normalize
    X_train /= 255.
    X_test  /= 255.

    # Split training and validation sets
    indexes_valid = np.random.choice(range(len(X_train)), 
                                     n_valid, 
                                     replace=False)
    mask = np.ones(Y_train.shape, bool)
    mask[indexes_valid] = False

    X_val, Y_val = X_train[indexes_valid], Y_train[indexes_valid]
    X_train, Y_train = X_train[mask], Y_train[mask]

    # Generate network
    network    = MNISTNet()
    net_params = get_parameters(network)
    optimizer  = SGD(net_params, lr=3e-3)

    # Train model
    network = train_model(
        network, 
        sparse_cross_entropy, 
        optimizer,
        X_train, Y_train, 
        X_val, Y_val,
        64, n_epochs
    )

    # Fix model
    network.training = False

    # 
    perturbation = FGSM_attack(
        network, 
        sparse_cross_entropy, 
        X_test, 
        Y_test
    )