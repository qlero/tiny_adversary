import numpy as np

from tinygrad.nn import Linear
from tinygrad.tensor import Tensor 

class MNISTNet:
    def __init__(self):
        self.l1 = Linear(784, 128, bias=True)
        self.l2 = Linear(128, 10, bias=True)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        logits = self.l2(x)
        return logits
    
def train_model(
        network, 
        loss_fn, 
        optimizer,
        X_train, Y_train, 
        X_val, Y_val, 
        n_batch, n_epochs
    ):
    Tensor.train = True
    X_val        = Tensor(X_val, requires_grad = False)

    for ep in range(n_epochs):
        # Retrieves data
        indexes = np.random.randint(0, X_train.shape[0], size=(n_batch,))
        data    = Tensor(X_train[indexes], requires_grad = False)
        labels  = Y_train[indexes]
        # Sets gradients to 0
        optimizer.zero_grad()
        # Forward pass
        outputs = network(data)
        loss    = loss_fn(outputs, labels)
        # Backward propagation
        loss.backward()
        optimizer.step()
        # Compute accuracy
        preds    = np.argmax(outputs.numpy(), axis=-1)
        accuracy = np.sum(preds == labels)/len(labels)
        if ep % 100 == 0:
            Tensor.training = False
            out     = network(X_val)
            preds   = np.argmax(out.softmax().numpy(), axis=-1)
            val_acc = np.sum(preds == Y_val)/len(Y_val)
            Tensor.training = True
            print(f"Epoch {ep+1} | " \
                  f"Loss: {loss.numpy()} | " \
                  f"Train acc: {100 * accuracy:.2f}% | "\
                  f"Val. acc: {100 * val_acc:.2f}%")
    
    return network