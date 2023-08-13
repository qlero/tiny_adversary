import numpy as np

from tinygrad.tensor import Tensor

class Adversarial:
    def __init__(
            self, 
            features,
            initialization: str = "zeros",
        ):
        """Initialization method for adversarial layer
        """
        self.perturbation = getattr(Tensor, initialization)(features)
        self.perturbation.requires_grad = True
    
    def __call__(self, x):
        """Method that allows modularization
        """
        # self.perturbation = self.epsilon * self.perturbation.grad.sign()
        x = x.add(self.perturbation)
        # x = x.minimum(1.)
        # x = x.maximum(-1)
        return x
    
    def __str__(self):
        return str(self.perturbation.numpy())

def FGSM_attack(
        network, 
        loss_fn,
        X_test, Y_test,
        target_class: int = 3
    ):

    Tensor.train = True
    network.training = False

    X_test  = Tensor(X_test)
    targets = Tensor.full(Y_test.shape, fill_value = target_class).numpy()

    adversarial = Adversarial(X_test.shape[:2])

    out      = network(X_test)
    preds    = np.argmax(out.softmax().numpy(), axis=-1)
    test_acc = np.sum(preds == Y_test)/len(Y_test)

    print(f"Clean test accuracy: {100 * test_acc:.2f}%")

    outputs = network(adversarial(X_test))
    loss    = -1 * loss_fn(outputs, targets)

    # Backward propagation
    loss.backward()

    # Compute accuracy
    perturbation = adversarial.perturbation
    perturbation = 0.15 * perturbation.grad.sign().numpy()
    import matplotlib.pyplot as plt
    plt.imshow((X_test+Tensor(perturbation)).numpy()[0].reshape(28, 28), cmap="gray")
    plt.savefig("test.png")
    outputs = network((X_test + Tensor(perturbation)).minimum(1).maximum(0))
    preds   = np.argmax(outputs.numpy(), axis=-1)
    adv_acc = np.sum(preds == Y_test)/len(Y_test)
    asr     = np.sum(preds == targets)/len(targets)

    print(f"Adversarial test accuracy: {100 * adv_acc:.2f}% | ASR: {100 * asr:.2f}")

    return adversarial.perturbation