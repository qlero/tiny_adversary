import numpy as np

from tinygrad.nn import Linear, Conv2d
from tinygrad.tensor import Tensor 

class TinyCNN:
    def __init__(self):
        # self.cnn1 = Conv2d(1, 32, 3)
        # self.cnn2 = Conv2d(32, 32, 7)
        self.l0 = Linear(784, 128, bias=True)
        self.l1 = Linear(128, 64, bias=True)
        self.l2 = Linear(64, 10, bias=False)

    def __call__(self, x):
        # x = self.cnn1(x).relu().max_pool2d(kernel_size=(3,3))
        # x = self.cnn2(x).relu()
        x = x.reshape(x.shape[0], -1)
        x = self.l0(x)
        x = x.relu()
        x = self.l1(x)
        x = x.relu()
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
        if ep % 50 == 0:
            Tensor.training = False
            out     = network(X_val)
            preds   = np.argmax(out.softmax().numpy(), axis=-1)
            val_acc = np.sum(preds == Y_val)/len(Y_val)
            Tensor.training = True
            print(f"\tEpoch {ep+1} | " \
                  f"Loss: {loss.numpy()} | " \
                  f"Train acc: {100 * accuracy:.2f}% | "\
                  f"Val. acc: {100 * val_acc:.2f}%")
    
    return network