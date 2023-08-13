import numpy as np

from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor

def one_hot_encoding(labels: np.array, n_classes: int):
    """One-hot encodes a set of integer labels (e.g. [0, ..., 9])
    """
    flat_labels       = labels.flatten().astype(np.int32)
    sparsified_labels = np.zeros((labels.shape[0], n_classes), np.float32)
    sparsified_labels[range(sparsified_labels.shape[0]), flat_labels] = -1.
    target_shape      = list(labels.shape) + [n_classes]
    sparsified_labels = sparsified_labels.reshape(target_shape)
    return Tensor(sparsified_labels)

def sparse_cross_entropy(logits: Tensor, labels: np.array, reduction: str = "sum"):
    """Implements the cross entropy loss 
    """
    n_classes = logits.shape[-1]
    ohe       = one_hot_encoding(labels, n_classes)
    if reduction == "sum":
        return logits.log_softmax().mul(ohe).sum()
    elif reduction == "mean":
        return logits.log_softmax().mul(ohe).mean()
    else:
        return logits.log_softmax().mul(ohe)