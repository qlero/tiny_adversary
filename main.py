import numpy as np

# from tinygrad.jit import TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.state import get_parameters
from tinygrad.tensor import Tensor
from extra.datasets import fetch_mnist

from src import attack_network
from src import TinyCNN, train_model
from src import sparse_cross_entropy

# Sets the training flag to True


if __name__ == "__main__":

    # Sets variables
    n_epochs = 1000
    n_valid  = 10000

    # Retrieve dataset
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

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
    network    = TinyCNN()
    net_params = get_parameters(network)
    optimizer  = SGD(net_params, lr=0.001)

    # Train model
    print("\nTrains a 4-layer CNN (2-conv2d, 2 linear) network (for attack purposes)")
    network = train_model(
        network, 
        sparse_cross_entropy, 
        optimizer,
        X_train, 
        Y_train, 
        X_val, 
        Y_val,
        32,
        n_epochs
    )

    # Runs FGSM attack
    print("\nRuns a FGSM attack with 2-norm, 0.3-eps, trg=3")
    perturbation = attack_network(
        network, 
        sparse_cross_entropy, 
        X_test, 
        Y_test,
        n_iterations = 1,
        random_initialization = False,
        norm = 2,
        epsilon = 0.3,
        target_class = 3,
        print_example = [True, "FGSM.png"]
    )

    # Runs R+FGSM attack
    print("\nRuns a R+FGSM attack with 2-norm, 0.3-eps, trg=3")
    perturbation = attack_network(
        network, 
        sparse_cross_entropy, 
        X_test, 
        Y_test,
        n_iterations = 1,
        random_initialization = True,
        norm = 2,
        epsilon = 0.3,
        target_class = 3,
        print_example = [True, "R_FGSM.png"]
    )

    # Runs PGD attack
    print("\nRuns a PGD attack with 40-iter, 2-norm, 0.3-eps, trg=3")
    perturbation = attack_network(
        network, 
        sparse_cross_entropy, 
        X_test, 
        Y_test,
        n_iterations = 40,
        random_initialization = True,
        norm = 2,
        epsilon = 0.3,
        target_class = 3,
        print_example = [True, "PGD_l2.png"]
    )

    # Runs test norm=inf attack
    print("\nRuns a PGD attack with 40-iter, inf-norm, 0.3-eps, trg=3")
    perturbation = attack_network(
        network, 
        sparse_cross_entropy, 
        X_test, 
        Y_test,
        n_iterations = 40,
        random_initialization = True,
        norm = float("inf"),
        epsilon = 0.3,
        target_class = 3,
        print_example = [True, "PGD_linf.png"]
    )

    