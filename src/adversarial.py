import matplotlib.pyplot as plt
import numpy as np

from tinygrad.tensor import Tensor

def tensor_norm(x: Tensor, p: float):
    """Computes the p-norm of a tensor.
    """
    assert p in [2., float("inf")]
    shape = x.shape
    if p == 2:
        norm = x.pow(2).reshape(x.shape[0], -1).sum(axis=1).sqrt().squeeze()
    elif p == float("inf"):
        norm = x.reshape(x.shape[0], -1).max(axis=1)
    norm = norm.reshape(norm.shape[0], *[1]*len(shape[1:]))
    return norm

class Adversarial_Buffer:

    def __init__(
            self, 
            features,
            random_initialization: bool = False,
            norm: float = None,
            epsilon: float = None
        ):
        """Initialization method for adversarial perturbation buffer.
        """
        # Declares the placeholder perturbation
        if random_initialization:
            self.perturbation = getattr(Tensor, "rand")(*features).sub(1/2).mul(2)
        else:
            self.perturbation = getattr(Tensor, "zeros")(*features)
        
        # If indicated, initializes the placeholder perturbation
        if random_initialization and \
           norm is not None and \
           epsilon is not None and epsilon > 0:
            normalizing_const = tensor_norm(self.perturbation, norm)
            self.perturbation = self.perturbation.div(normalizing_const).mul(epsilon)

        self.perturbation.requires_grad = True

    def __call__(self, x):
        """Method that allows modularization
        """
        x = x.add(self.perturbation)
        return x
    
    def __str__(self):
        return str(self.perturbation.numpy())
    
class Adversarial_Attack():

    def __init__(self: object,
                 loss_function: object,
                 n_iterations: int = 1,
                 random_initialization: bool = False,
                 norm: float = None,
                 epsilon: float = None):
        """Initialization method for Adversarial Attack class.
        """
        
        # Basic checks
        assert type(n_iterations)==int and n_iterations > 0 #<n_iterations> must be positive int
        assert type(random_initialization)==bool # <random_initialization> must be a bool
        assert norm in [2., float("inf")]
        assert epsilon is None or type(epsilon)==float and epsilon > 0 # <epsilon> is a positive, real-valued normalizing constant
        
        # Prints the attack type
        if n_iterations == 1:
            if random_initialization:
                print("\tR+FGSM Adversarial attack declared.")
            else:
                print("\tFGSM Adversarial attack declared.")
        else:
            print("\tProjected Gradient Descent (PGD) Adversarial attack declared.")

        self.loss = loss_function
        self.iter = n_iterations
        self.rand = random_initialization
        self.norm = norm
        self.cons = epsilon

    def run(self: object,
            X: Tensor, 
            Y: Tensor,
            network: object,
            target_class: int = 0):
        """Runs the declared adversarial attack.
        """
        # Sets flags
        Tensor.train = True
        network.training = False

        # Declares array of targets (needs to be numpy for loss func)
        targets = Tensor.full(Y.shape, fill_value=target_class).numpy()

        # Declares adversarial buffer
        perturbations = Adversarial_Buffer(X.shape, self.rand, self.norm, self.cons)

        if self.iter == 1:
            # Retrieves loss
            out = network(perturbations(X))
            loss = self.loss(out, targets).mul(-1.)
            loss.backward()
            # Compute perturbation
            perturbation = perturbations.perturbation
            perturbation = perturbation.grad.sign().mul(self.cons)
            # Clamps perturbation
            perturbation = X.add(perturbation).maximum(0).minimum(1).sub(X)

            return perturbation

        # Iterates the attack
        for _ in range(self.iter):
            # Retrieves loss
            out = network(perturbations(X))
            loss = self.loss(out, targets).mul(-1.)
            loss.backward()
            # Compute perturbation update
            perturbation = perturbations.perturbation
            update = perturbation.grad
            norm = tensor_norm(update, self.norm)
            update = update.div(norm).mul(self.cons/self.iter if self.norm!=2 else self.cons*0.2)
            perturbation = perturbation.add(update).realize()
            # Clamps perturbation
            perturbation = X.add(perturbation).maximum(0).minimum(1).sub(X)
            # Feed update back into buffer
            perturbations.perturbation = perturbation.realize()
        
        return perturbations.perturbation
            
def attack_network(
        network: object, 
        loss_fn: object,
        X_test: np.array, 
        Y_test: np.array,
        n_iterations: int = 1,
        random_initialization: bool = False,
        norm: float = 2,
        epsilon: float = 0.3,
        target_class: int = 3,
        print_example: list = None
    ):
    """Function to test an attack on a network using first 128 elements of given set
    """

    X_test = Tensor(X_test[:128])
    Y_test = Y_test[:128]

    out      = network(X_test)
    preds    = np.argmax(out.softmax().numpy(), axis=-1)
    test_acc = np.sum(preds == Y_test)/len(Y_test)

    print(f"\tClean test accuracy: {100 * test_acc:.2f}%")

    attack = Adversarial_Attack(loss_fn, n_iterations, random_initialization, norm, epsilon)
    perturbations = attack.run(X_test, Y_test, network, target_class)

    targets = Tensor.full(Y_test.shape, fill_value=target_class).numpy()
    inputs  = X_test.add(perturbations).minimum(1).maximum(0)
    outputs = network(inputs)
    preds   = np.argmax(outputs.numpy(), axis=-1)
    # print(preds, targets)
    adv_acc = np.sum(preds == Y_test)/len(Y_test)
    asr     = np.sum(preds == targets)/len(Y_test)

    print(f"\tAdversarial test accuracy: {100 * adv_acc:.2f}% | ASR: {100 * asr:.2f}%")

    if print_example is not None and print_example[0]:
        plt.imshow(inputs.numpy()[2].reshape(28, 28), cmap="gray")
        plt.savefig(print_example[1])

    return None