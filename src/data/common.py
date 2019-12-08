import torch
import numpy as np

from scipy.io import loadmat
from skimage.io import imread

def default_loader(path_):
    return imread(path_)

def mat_loader(path_):
    return loadmat(path_)

def make_onehot(index_map, n):
    # Only deals with tensors with no batch dim
    old_size = index_map.size()
    z = torch.zeros(n, *old_size[-2:]).type_as(index_map)
    z.scatter_(0, index_map, 1)
    return z
    
def to_tensor(arr):
    if arr.ndim < 3:
        return torch.from_numpy(arr)
    elif arr.ndim == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(arr, (2,0,1))))
    else:
        raise NotImplementedError

def to_array(tensor):
    if tensor.ndimension() < 3:
        return tensor.data.cpu().numpy()
    elif tensor.ndimension() in (3, 4):
        return np.ascontiguousarray(np.moveaxis(tensor.data.cpu().numpy(), -3, -1))
    else:
        raise NotImplementedError