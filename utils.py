import numpy as np 
import torch 



def orthogonal(tensor, gain = 1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are suppored")
    
    rows = tensor.size(0)
    cols = tensor[0].numel()

    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()


    q, r = torch.qr(flattened)
    d = torch.diag(r, 0)
    ph = d.sign()

    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor

'''
def flatten_env_vec(arr):
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
'''

def flatten_env_vec(arr):
    s = list(arr.size())
    # return torch.transpose(arr, 0, 1).view(s[0] * s[1], *s[2:])
    return arr.view(s[0] * s[1], *s[2:])